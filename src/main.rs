#[macro_use]
mod macros;
pub mod tooling;

use std::collections::BTreeMap;
use crate::tooling::builtin_tools::{ExecTool, ReadFileTool, WriteFileTool};
use crate::tooling::types::*;

use reqwest::blocking::Client;
use serde_json::{Map, Value, json};
use std::env;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::time::Duration;

// Config
const LLM_URL: &str = "http://localhost:11434/api/chat";
const MODEL: &str = "qwen3.5:latest";
const MAX_ITERATIONS: usize = 20;

fn api_key() -> Option<String> {
    None
    // env::var("ANTHROPIC_API_KEY").ok()
}

fn home_dir() -> PathBuf {
    env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
}

fn build_context() -> ToolContext {
    let workspace_dir = home_dir().join(".atomiagent");
    let sessions_dir = workspace_dir.join("sessions");

    ToolContext {
        workspace_dir,
        sessions_dir,
    }
}

fn ensure_dirs(ctx: &ToolContext) -> Result<(), String> {
    fs::create_dir_all(&ctx.sessions_dir).map_err(|e| format!("Failed to create dirs: {e}"))
}

fn build_tools() -> ToolRegistry {
    register_tools![ReadFileTool, WriteFileTool, ExecTool]
}

fn tool_catalog_json(tools: &ToolRegistry) -> Value {
    let mut map = Map::new();

    for tool in tools.values() {
        map.insert(tool.name().to_string(), tool.as_catalog_entry());
    }

    Value::Object(map)
}

fn tools_payload(tools: &ToolRegistry) -> Vec<Value> {
    tools.values().map(|tool| tool.as_tool_payload()).collect()
}

fn post_json(url: &str, payload: &Value, timeout_secs: u64) -> Result<Value, String> {
    let client = Client::builder()
        .timeout(Duration::from_secs(timeout_secs))
        .build()
        .map_err(|e| format!("Client error: {e}"))?;

    let mut req = client.post(url).json(payload);

    if let Some(key) = api_key() {
        req = req.bearer_auth(key);
    }

    let resp = req.send().map_err(|e| format!("Connection error: {e}"))?;
    let status = resp.status();
    let body = resp.text().map_err(|e| format!("Read error: {e}"))?;

    if !status.is_success() {
        return Err(format!("HTTP {}: {}", status.as_u16(), body));
    }

    serde_json::from_str(&body).map_err(|e| format!("Invalid JSON response: {e}"))
}

struct Agent {
    tools: ToolRegistry,
    ctx: ToolContext,
}

impl Agent {
    fn new() -> Result<Self, String> {
        let ctx = build_context();
        ensure_dirs(&ctx)?;

        Ok(Self {
            tools: build_tools(),
            ctx,
        })
    }

    fn system_prompt(&self) -> String {
        format!(
            r#"You are a helpful Personal Assistant agent. Use tools via JSON calls when needed.
- To use a tool: respond with ONLY {{"tool": "tool_name", "args": {{...}}}}
- When done, respond with plain text answer.
Available tools: {}
Workspace: {}"#,
            tool_catalog_json(&self.tools),
            self.ctx.workspace_dir.display()
        )
    }

    fn load_session(&self, session_id: &str) -> Vec<Value> {
        let session_file = self.ctx.sessions_dir.join(format!("{session_id}.jsonl"));

        if session_file.exists() {
            if let Ok(file) = File::open(&session_file) {
                let reader = BufReader::new(file);
                let messages: Vec<Value> = reader
                    .lines()
                    .filter_map(Result::ok)
                    .filter_map(|line| serde_json::from_str::<Value>(&line).ok())
                    .collect();

                if !messages.is_empty() {
                    return messages;
                }
            }
        }

        vec![json!({
            "role": "system",
            "content": self.system_prompt()
        })]
    }

    fn save_messages(&self, session_id: &str, messages: &[Value]) -> Result<(), String> {
        let session_file = self.ctx.sessions_dir.join(format!("{session_id}.jsonl"));
        let start = messages.len().saturating_sub(10);

        let mut file = File::create(&session_file)
            .map_err(|e| format!("Failed to open session file for writing: {e}"))?;

        for msg in &messages[start..] {
            let line = serde_json::to_string(msg)
                .map_err(|e| format!("Failed to serialize message: {e}"))?;
            writeln!(file, "{line}").map_err(|e| format!("Failed to write session file: {e}"))?;
        }

        Ok(())
    }

    fn call_llm(&self, messages: &[Value]) -> Result<Value, String> {
        let payload = json!({
        "model": MODEL,
        "messages": messages,
        "stream": false,
        "tools": tools_payload(&self.tools),
    });

        let data = post_json(LLM_URL, &payload, 60)?;
        data.get("message")
            .cloned()
            .ok_or_else(|| "Missing `message` in LLM response.".to_string())
    }

    fn execute_tool(&self, tool_call: &Value) -> String {
        let tool_name = tool_call
            .get("tool")
            .and_then(Value::as_str)
            .unwrap_or_default();

        let args = tool_call.get("args").unwrap_or(&Value::Null);

        match self.tools.get(tool_name) {
            Some(tool) => tool.execute(args, &self.ctx),
            None => "Unknown tool.".to_string(),
        }
    }

    fn run(&self, session_id: &str, user_message: &str) -> String {
        let mut messages = self.load_session(session_id);

        messages.push(json!({
        "role": "user",
        "content": user_message
    }));

        for _ in 0..MAX_ITERATIONS {
            let assistant_message = match self.call_llm(&messages) {
                Ok(msg) => msg,
                Err(e) => return format!("LLM error: {e}"),
            };

            messages.push(assistant_message.clone());

            let tool_calls = assistant_message
                .get("tool_calls")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();

            if tool_calls.is_empty() {
                let content = assistant_message
                    .get("content")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();

                let _ = self.save_messages(session_id, &messages);
                return content;
            }

            for tc in tool_calls {
                let tool_name = tc
                    .get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(Value::as_str)
                    .unwrap_or("");

                let args = tc
                    .get("function")
                    .and_then(|f| f.get("arguments"))
                    .cloned()
                    .unwrap_or(Value::Null);

                let result = match self.tools.get(tool_name) {
                    Some(tool) => tool.execute(&args, &self.ctx),
                    None => format!("Unknown tool: {tool_name}"),
                };

                messages.push(json!({
                "role": "tool",
                "tool_name": tool_name,
                "content": result
            }));
            }

            let _ = self.save_messages(session_id, &messages);
        }

        "Max iterations reached.".to_string()
    }
}

fn main() {
    let agent = match Agent::new() {
        Ok(agent) => agent,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };

    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        eprintln!("Usage: {} <session_id> '<message>'", args[0]);
        std::process::exit(1);
    }

    let session_id = &args[1];
    let message = args[2..].join(" ");

    let response = agent.run(session_id, &message);
    println!("{response}");
}
