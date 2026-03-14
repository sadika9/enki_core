mod workspace;

use crate::agent::workspace::AgentWorkspace;
use crate::llm::{
    ChatMessage, LlmConfig, LlmProvider, MessageRole, UniversalLLMClient,
};
use crate::message::message::{IndexedValue, Message, next_request_id};
use crate::tooling::tool_calling::{RegistryToolExecutor, ToolCallRegistry, ToolExecutor};
use crate::tooling::types::ToolContext;
use crate::build_tools;
use serde_json::Value;
use std::path::PathBuf;
use tokio::fs::File;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

// Config
const DEFAULT_MAX_ITERATIONS: usize = 20;

pub struct AgentDefinition {
    pub name: String,
    pub system_prompt_preamble: String,
    pub model: String,
    pub max_iterations: usize,
}

impl Default for AgentDefinition {
    fn default() -> Self {
        Self {
            name: "Personal Assistant".to_string(),
            system_prompt_preamble: "You are a helpful Personal Assistant agent.".to_string(),
            model: String::new(),
            max_iterations: DEFAULT_MAX_ITERATIONS,
        }
    }
}

pub struct Agent {
    definition: AgentDefinition,
    tool_registry: ToolCallRegistry,
    tool_executor: Box<dyn ToolExecutor>,
    workspace: AgentWorkspace,
    llm: Box<dyn LlmProvider>,
}

enum StepOutcome {
    Continue,
    Final(String),
}

struct ToolInvocation {
    name: String,
    args: Value,
    call_id: Option<String>,
}

impl Agent {
    fn resolve_model(definition: &AgentDefinition) -> Result<String, String> {
        if !definition.model.trim().is_empty() {
            return Ok(definition.model.clone());
        }

        std::env::var("ENKI_MODEL")
            .map_err(|_| "Missing model. Set AgentDefinition.model or ENKI_MODEL.".to_string())
    }

    pub(crate) async fn new() -> Result<Self, String> {
        Self::with_definition(AgentDefinition::default()).await
    }

    pub(crate) async fn with_definition(definition: AgentDefinition) -> Result<Self, String> {
        Self::with_definition_executor_and_workspace(
            definition,
            Box::new(RegistryToolExecutor),
            None,
        )
        .await
    }

    pub(crate) async fn with_definition_and_executor(
        definition: AgentDefinition,
        tool_executor: Box<dyn ToolExecutor>,
    ) -> Result<Self, String> {
        Self::with_definition_executor_and_workspace(definition, tool_executor, None).await
    }

    pub(crate) async fn with_definition_executor_and_workspace(
        definition: AgentDefinition,
        tool_executor: Box<dyn ToolExecutor>,
        workspace_home: Option<PathBuf>,
    ) -> Result<Self, String> {
        let model = Self::resolve_model(&definition)?;
        let workspace = AgentWorkspace::new(&definition.name, workspace_home);
        workspace.ensure_dirs().await?;

        Ok(Self {
            llm: Box::new(UniversalLLMClient::new(&model).map_err(|e| e.to_string())?),
            definition,
            tool_registry: ToolCallRegistry::new(build_tools()),
            tool_executor,
            workspace,
        })
    }

    fn system_prompt(&self, ctx: &ToolContext) -> String {
        format!(
            r#"You are {}.
{} Use tools via JSON calls when needed.
- Process each incoming user message as a loop:
  1. Receive the message.
  2. Interpret it.
  3. Choose the next action.
  4. Either reply immediately, call a tool, or ask a follow-up question.
  5. If you call a tool, read the result and continue the loop.
  6. Stop only when a final reply is ready.
- One user message may require multiple internal iterations before the final answer.
- If a tool is needed, prefer native tool calls. If native tool calling is unavailable, respond with ONLY {{"tool": "tool_name", "args": {{...}}}}.
- When done, respond with plain text.
Available tools: {}
Agent workspace: {}
Current task workspace: {}"#,
            self.definition.name,
            self.definition.system_prompt_preamble,
            self.tool_registry.catalog_json(),
            ctx.agent_dir.display(),
            ctx.workspace_dir.display()
        )
    }

    async fn load_session(&self, session_id: &str, ctx: &ToolContext) -> Vec<Message> {
        let session_file = self.workspace.session_file(session_id);

        if tokio::fs::try_exists(&session_file).await.unwrap_or(false) {
            if let Ok(file) = File::open(&session_file).await {
                let reader = BufReader::new(file);
                let mut lines = reader.lines();
                let mut messages = Vec::new();
                let mut index = 0usize;

                while let Ok(Some(line)) = lines.next_line().await {
                    if let Ok(value) = serde_json::from_str::<Value>(&line) {
                        if let Ok(message) = Message::try_from(IndexedValue { index, value }) {
                            messages.push(message);
                        }
                    }
                    index += 1;
                }

                if !messages.is_empty() {
                    return messages;
                }
            }
        }

        vec![Message::system(self.system_prompt(ctx))]
    }

    async fn save_messages(&self, session_id: &str, messages: &[Message]) -> Result<(), String> {
        let session_file = self.workspace.session_file(session_id);
        let start = messages.len().saturating_sub(10);

        let mut file = File::create(&session_file)
            .await
            .map_err(|e| format!("Failed to open session file for writing: {e}"))?;

        for msg in &messages[start..] {
            let line = serde_json::to_string(&Value::from(msg.clone()))
                .map_err(|e| format!("Failed to serialize message: {e}"))?;
            file.write_all(line.as_bytes())
                .await
                .map_err(|e| format!("Failed to write session file: {e}"))?;
            file.write_all(b"\n")
                .await
                .map_err(|e| format!("Failed to write session file: {e}"))?;
        }

        Ok(())
    }

    fn to_llm_messages(&self, messages: &[Message]) -> Vec<ChatMessage> {
        messages
            .iter()
            .filter_map(|message| {
                let value = Value::from(message);
                let role = match value.get("role").and_then(Value::as_str)? {
                    "system" => MessageRole::System,
                    "user" => MessageRole::User,
                    "assistant" => MessageRole::Assistant,
                    "tool" => MessageRole::Tool,
                    _ => return None,
                };

                Some(ChatMessage {
                    role,
                    content: value
                        .get("content")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string(),
                    tool_call_id: value
                        .get("tool_call_id")
                        .and_then(Value::as_str)
                        .map(str::to_string),
                })
            })
            .collect()
    }

    async fn call_llm(&self, messages: &[Message]) -> Result<Value, String> {
        let response = self
            .llm
            .complete(&self.to_llm_messages(messages), &LlmConfig::default())
            .await
            .map_err(|e| e.to_string())?;

        Ok(serde_json::json!({
            "role": "assistant",
            "content": response.content,
        }))
    }

    fn parse_content_tool_call(&self, assistant_message: &Value) -> Option<(String, Value)> {
        let content = assistant_message.get("content")?.as_str()?;
        let parsed: Value = serde_json::from_str(content).ok()?;
        let tool_name = parsed.get("tool")?.as_str()?.to_string();
        let args = parsed.get("args").cloned().unwrap_or(Value::Null);
        Some((tool_name, args))
    }

    async fn persist(&self, session_id: &str, messages: &[Message]) {
        let _ = self.save_messages(session_id, messages).await;
    }

    fn push_out_message(messages: &mut Vec<Message>, value: Value) {
        let prev_message_id = messages.last().map(|message| message.message_id.clone());
        messages.push(Message::out(value, prev_message_id));
    }

    fn extract_tool_invocations(&self, assistant_message: &Value) -> Vec<ToolInvocation> {
        let native_calls = assistant_message
            .get("tool_calls")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();

        if !native_calls.is_empty() {
            return native_calls
                .into_iter()
                .map(|tool_call| ToolInvocation {
                    name: tool_call
                        .get("function")
                        .and_then(|function| function.get("name"))
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string(),
                    args: tool_call
                        .get("function")
                        .and_then(|function| function.get("arguments"))
                        .cloned()
                        .unwrap_or(Value::Null),
                    call_id: tool_call
                        .get("id")
                        .and_then(Value::as_str)
                        .map(str::to_string),
                })
                .collect();
        }

        self.parse_content_tool_call(assistant_message)
            .map(|(name, args)| {
                vec![ToolInvocation {
                    name,
                    args,
                    call_id: None,
                }]
            })
            .unwrap_or_default()
    }

    async fn build_tool_result_messages(
        &self,
        invocations: Vec<ToolInvocation>,
        ctx: &ToolContext,
        parent_message_id: Option<String>,
    ) -> Vec<Message> {
        let mut messages = Vec::with_capacity(invocations.len());

        for invocation in invocations {
            let tool_message = self
                .tool_executor
                .build_tool_message(
                    &self.tool_registry,
                    &invocation.name,
                    &invocation.args,
                    ctx,
                    invocation.call_id.as_deref(),
                )
                .await;
            messages.push(Message::out(tool_message, parent_message_id.clone()));
        }

        messages
    }

    async fn step(
        &self,
        messages: &mut Vec<Message>,
        ctx: &ToolContext,
    ) -> Result<StepOutcome, String> {
        let assistant_message = self.call_llm(messages).await?;
        Self::push_out_message(messages, assistant_message.clone());

        let parent_message_id = messages.last().map(|message| message.message_id.clone());
        let invocations = self.extract_tool_invocations(&assistant_message);

        if !invocations.is_empty() {
            let tool_messages = self
                .build_tool_result_messages(invocations, ctx, parent_message_id)
                .await;
            messages.extend(tool_messages);
            return Ok(StepOutcome::Continue);
        }

        let content = assistant_message
            .get("content")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();

        Ok(StepOutcome::Final(content))
    }

    pub(crate) async fn run(&self, session_id: &str, user_message: &str) -> String {
        let ctx = self.workspace.tool_context(session_id);
        if let Err(e) = tokio::fs::create_dir_all(&ctx.workspace_dir).await {
            return format!("Workspace error: {e}");
        }

        let mut messages = self.load_session(session_id, &ctx).await;
        let request_id = next_request_id();
        let prev_message_id = messages.last().map(|message| message.message_id.clone());

        messages.push(Message::user(
            user_message.to_string(),
            request_id,
            prev_message_id,
            None,
        ));

        for _ in 0..self.definition.max_iterations {
            match self.step(&mut messages, &ctx).await {
                Ok(StepOutcome::Continue) => {
                    self.persist(session_id, &messages).await;
                }
                Ok(StepOutcome::Final(content)) => {
                    self.persist(session_id, &messages).await;
                    return content;
                }
                Err(e) => return format!("LLM error: {e}"),
            }
        }

        self.persist(session_id, &messages).await;
        "Max iterations reached.".to_string()
    }
}
