//! Universal LLM Client using graniet/llm library.
//!
//! Provides a unified interface for 13+ LLM providers using the `provider::model` format.
//!
//! # Supported Providers
//!
//! - **OpenAI** - `openai::gpt-4o`, `openai::gpt-3.5-turbo`
//! - **Anthropic** - `anthropic::claude-3-opus-20240229`, `anthropic::claude-3-sonnet-20240229`
//! - **Ollama** - `ollama::llama3.2`, `ollama::gemma3:latest`, `ollama::mistral`
//! - **Google** - `google::gemini-pro`
//! - **DeepSeek** - `deepseek::deepseek-chat`
//! - **X.AI** - `xai::grok`
//! - **Groq** - `groq::llama3-70b-8192`
//! - **Mistral** - `mistral::mistral-large`
//! - **Cohere** - `cohere::command`
//! - **OpenRouter** - `openrouter::anthropic/claude-3-opus`
//!
//! # Examples
//!
//! ```rust,no_run
//! use enki_llm::UniversalLLMClient;
//!
//! // Ollama local model (no API key needed)
//! let client = UniversalLLMClient::new("ollama::gemma3:latest").unwrap();
//!
//! // OpenAI with env var API key (OPENAI_API_KEY)
//! let client = UniversalLLMClient::new("openai::gpt-4o").unwrap();
//!
//! // With explicit API key
//! let client = UniversalLLMClient::with_api_key(
//!     "anthropic::claude-3-sonnet-20240229",
//!     "sk-ant-..."
//! ).unwrap();
//! ```

use async_trait::async_trait;
use futures::Stream;
use llm::LLMProvider;
use llm::builder::{LLMBackend, LLMBuilder};
use llm::chat::ChatMessage as LlmChatMessage;
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use std::sync::Arc;
use tracing::debug;

/// Error type for LLM operations in this module.
#[derive(Debug, Clone)]
pub enum LlmError {
    Config(String),
    Provider(String),
}

impl std::fmt::Display for LlmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Config(msg) => write!(f, "Config error: {}", msg),
            Self::Provider(msg) => write!(f, "Provider error: {}", msg),
        }
    }
}

impl std::error::Error for LlmError {}

/// Convenience result type for this module.
pub type Result<T> = std::result::Result<T, LlmError>;

/// Message role for chat interactions.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum MessageRole {
    System,
    User,
    Assistant,
    Tool,
}

/// Minimal chat message representation for this module.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: MessageRole,
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// Optional usage stats.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmUsage {
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
    pub total_tokens: Option<u32>,
}

/// LLM response wrapper.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmResponse {
    pub content: String,
    pub usage: Option<LlmUsage>,
    pub tool_calls: Vec<String>,
    pub model: String,
    pub finish_reason: Option<String>,
}

/// Configuration placeholder for completion calls.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LlmConfig {}

/// Tool definition placeholder.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: Option<String>,
}

/// Stream type for streaming responses.
pub type ResponseStream = Pin<Box<dyn Stream<Item = Result<LlmResponse>> + Send>>;

/// Minimal provider trait for this module.
#[async_trait]
pub trait LlmProvider: Send + Sync {
    async fn complete(&self, messages: &[ChatMessage], config: &LlmConfig) -> Result<LlmResponse>;
    async fn complete_stream(
        &self,
        messages: &[ChatMessage],
        config: &LlmConfig,
    ) -> Result<ResponseStream>;
    async fn complete_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        config: &LlmConfig,
    ) -> Result<LlmResponse>;
    fn name(&self) -> &'static str;
    fn available_models(&self) -> Vec<&'static str>;
}

/// Configuration for Universal LLM Client.
///
/// Extended configuration supporting all graniet/llm options.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniversalConfig {
    /// Model in "provider::model" format
    pub model: String,

    /// Optional API key (auto-detected from env if not provided)
    pub api_key: Option<String>,

    /// Optional base URL override
    pub base_url: Option<String>,

    /// Max tokens to generate
    pub max_tokens: Option<u32>,

    /// Sampling temperature
    pub temperature: Option<f32>,

    /// Top-p sampling
    pub top_p: Option<f32>,

    /// Top-k sampling
    pub top_k: Option<u32>,

    /// System prompt
    pub system: Option<String>,

    /// Timeout in seconds
    pub timeout_seconds: Option<u64>,

    /// Enable resilient mode with retries
    pub resilient: Option<bool>,

    /// Number of retry attempts
    pub resilient_attempts: Option<usize>,
}

impl Default for UniversalConfig {
    fn default() -> Self {
        Self {
            model: String::new(),
            api_key: None,
            base_url: None,
            max_tokens: Some(4096),
            temperature: None,
            top_p: None,
            top_k: None,
            system: None,
            timeout_seconds: None,
            resilient: None,
            resilient_attempts: None,
        }
    }
}

impl UniversalConfig {
    /// Create new config with model in "provider::model" format.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            ..Default::default()
        }
    }

    /// Set API key.
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Set base URL.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Set max tokens.
    pub fn with_max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = Some(tokens);
        self
    }

    /// Set temperature.
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }

    /// Set system prompt.
    pub fn with_system(mut self, prompt: impl Into<String>) -> Self {
        self.system = Some(prompt.into());
        self
    }

    /// Enable resilience with retry count.
    pub fn with_resilience(mut self, attempts: usize) -> Self {
        self.resilient = Some(true);
        self.resilient_attempts = Some(attempts);
        self
    }

    /// Parse provider name from model string.
    pub fn provider(&self) -> Option<&str> {
        self.model.split("::").next()
    }

    /// Parse model name from model string.
    pub fn model_name(&self) -> &str {
        self.model.split("::").nth(1).unwrap_or(&self.model)
    }
}

/// Get environment variable name for a provider's API key.
fn get_api_key_env_var(provider: &str) -> Option<&'static str> {
    match provider.to_lowercase().as_str() {
        "ollama" => None, // Ollama doesn't require API key
        "anthropic" | "claude" => Some("ANTHROPIC_API_KEY"),
        "openai" | "gpt" => Some("OPENAI_API_KEY"),
        "deepseek" => Some("DEEPSEEK_API_KEY"),
        "xai" | "x.ai" => Some("XAI_API_KEY"),
        "phind" => Some("PHIND_API_KEY"),
        "google" | "gemini" => Some("GOOGLE_API_KEY"),
        "groq" => Some("GROQ_API_KEY"),
        "azure" | "azureopenai" | "azure-openai" => Some("AZURE_OPENAI_API_KEY"),
        "elevenlabs" | "11labs" => Some("ELEVENLABS_API_KEY"),
        "cohere" => Some("COHERE_API_KEY"),
        "mistral" => Some("MISTRAL_API_KEY"),
        "openrouter" => Some("OPENROUTER_API_KEY"),
        _ => None,
    }
}

/// Get API key from environment variable for the provider.
fn get_api_key_from_env(provider: &str) -> Result<Option<String>> {
    match get_api_key_env_var(provider) {
        None => Ok(None), // Provider doesn't need API key
        Some(env_var) => match std::env::var(env_var) {
            Ok(key) => Ok(Some(key)),
            Err(_) => Err(LlmError::Config(format!(
                "API key required for provider '{}'. Please set {} environment variable.",
                provider, env_var
            ))),
        },
    }
}

/// Parse provider string to LLMBackend.
fn parse_backend(provider: &str) -> Result<LLMBackend> {
    match provider.to_lowercase().as_str() {
        "ollama" => Ok(LLMBackend::Ollama),
        "anthropic" | "claude" => Ok(LLMBackend::Anthropic),
        "openai" | "gpt" => Ok(LLMBackend::OpenAI),
        "deepseek" => Ok(LLMBackend::DeepSeek),
        "xai" | "x.ai" => Ok(LLMBackend::XAI),
        "phind" => Ok(LLMBackend::Phind),
        "google" | "gemini" => Ok(LLMBackend::Google),
        "groq" => Ok(LLMBackend::Groq),
        "azure" | "azureopenai" => Ok(LLMBackend::AzureOpenAI),
        "elevenlabs" | "11labs" => Ok(LLMBackend::ElevenLabs),
        "cohere" => Ok(LLMBackend::Cohere),
        "mistral" => Ok(LLMBackend::Mistral),
        "openrouter" => Ok(LLMBackend::OpenRouter),
        _ => Err(LlmError::Config(format!(
            "Unknown LLM provider: '{}'. Supported: ollama, openai, anthropic, google, deepseek, xai, groq, mistral, cohere, openrouter",
            provider
        ))),
    }
}

/// Universal LLM Client wrapping graniet/llm.
///
/// Provides a unified interface for multiple LLM backends using the `provider::model` format.
pub struct UniversalLLMClient {
    config: UniversalConfig,
    llm: Arc<Box<dyn LLMProvider>>,
}

impl UniversalLLMClient {
    /// Create a new client from "provider::model" format.
    ///
    /// API keys are auto-detected from environment variables.
    ///
    /// # Examples
    /// ```rust,no_run
    /// use enki_llm::UniversalLLMClient;
    ///
    /// let client = UniversalLLMClient::new("ollama::gemma3:latest")?;
    /// let client = UniversalLLMClient::new("openai::gpt-4o")?;
    /// # Ok::<(), enki_llm::LlmError>(())
    /// ```
    pub fn new(provider_model: &str) -> Result<Self> {
        Self::with_config(UniversalConfig::new(provider_model))
    }

    /// Create a new client with explicit API key.
    pub fn with_api_key(provider_model: &str, api_key: impl Into<String>) -> Result<Self> {
        Self::with_config(UniversalConfig::new(provider_model).with_api_key(api_key))
    }

    /// Create a new client with full configuration.
    pub fn with_config(config: UniversalConfig) -> Result<Self> {
        let parts: Vec<&str> = config.model.split("::").collect();

        if parts.len() < 2 {
            return Err(LlmError::Config(format!(
                "Invalid model format. Use 'provider::model-name'. Got: {}",
                config.model
            )));
        }

        let provider = parts[0];
        let model = parts[1..].join(":"); // Handle models like gemma3:latest

        // Determine API key
        let api_key = match &config.api_key {
            Some(key) => Some(key.clone()),
            None => get_api_key_from_env(provider)?,
        };

        // Parse backend
        let backend = parse_backend(provider)?;

        // Get base URL for Ollama
        let base_url = config.base_url.clone().or_else(|| {
            if provider.to_lowercase() == "ollama" {
                Some(
                    std::env::var("OLLAMA_URL")
                        .unwrap_or_else(|_| "http://127.0.0.1:11434".to_string()),
                )
            } else {
                None
            }
        });

        // Build LLM using graniet/llm builder
        let mut builder = LLMBuilder::new().backend(backend).model(&model);

        if let Some(ref key) = api_key {
            builder = builder.api_key(key);
        }

        if let Some(ref url) = base_url {
            builder = builder.base_url(url);
        }

        if let Some(max_tokens) = config.max_tokens {
            builder = builder.max_tokens(max_tokens);
        }

        if let Some(temp) = config.temperature {
            builder = builder.temperature(temp);
        }

        if let Some(top_p) = config.top_p {
            builder = builder.top_p(top_p);
        }

        if let Some(top_k) = config.top_k {
            builder = builder.top_k(top_k);
        }

        if let Some(ref system) = config.system {
            builder = builder.system(system);
        }

        if let Some(timeout) = config.timeout_seconds {
            builder = builder.timeout_seconds(timeout);
        }

        if config.resilient == Some(true) {
            builder = builder.resilient(true);
            if let Some(attempts) = config.resilient_attempts {
                builder = builder.resilient_attempts(attempts);
            }
        }

        let llm = builder
            .build()
            .map_err(|e| LlmError::Config(format!("Failed to build LLM: {}", e)))?;

        debug!(
            provider = provider,
            model = model,
            "Created UniversalLLMClient"
        );

        Ok(Self {
            config,
            llm: Arc::new(llm),
        })
    }

    /// Get the configuration.
    pub fn config(&self) -> &UniversalConfig {
        &self.config
    }

    /// Get the provider name.
    pub fn provider(&self) -> Option<&str> {
        self.config.provider()
    }

    /// Get the model name.
    pub fn model_name(&self) -> &str {
        self.config.model_name()
    }

    /// Convert internal ChatMessage to llm crate's ChatMessage.
    fn convert_messages(messages: &[ChatMessage]) -> Vec<LlmChatMessage> {
        messages
            .iter()
            .map(|msg| match msg.role {
                MessageRole::System => {
                    // System messages are often handled as user messages with context
                    LlmChatMessage::user().content(&msg.content).build()
                }
                MessageRole::User => LlmChatMessage::user().content(&msg.content).build(),
                MessageRole::Assistant => LlmChatMessage::assistant().content(&msg.content).build(),
                MessageRole::Tool => {
                    // Tool results - format as user message with context.
                    // Keep tool_call_id in-band since llm::chat::ChatMessage doesn't support it.
                    let content = match &msg.tool_call_id {
                        Some(tool_call_id) => {
                            format!(
                                "Tool result (tool_call_id={}): {}",
                                tool_call_id, msg.content
                            )
                        }
                        None => format!("Tool result: {}", msg.content),
                    };
                    LlmChatMessage::user()
                        .content(content)
                        .build()
                }
            })
            .collect()
    }
}

// Implement the existing LlmProvider trait for backwards compatibility
#[async_trait]
impl LlmProvider for UniversalLLMClient {
    async fn complete(&self, messages: &[ChatMessage], _config: &LlmConfig) -> Result<LlmResponse> {
        let chat_messages = Self::convert_messages(messages);

        let response = self
            .llm
            .chat(&chat_messages)
            .await
            .map_err(|e| LlmError::Provider(format!("LLM error: {}", e)))?;

        let content = response.text().unwrap_or_default();

        Ok(LlmResponse {
            content,
            usage: None, // graniet/llm doesn't expose usage directly in basic chat
            tool_calls: Vec::new(),
            model: self.config.model.clone(),
            finish_reason: Some("stop".to_string()),
        })
    }

    async fn complete_stream(
        &self,
        _messages: &[ChatMessage],
        _config: &LlmConfig,
    ) -> Result<ResponseStream> {
        // TODO: Implement streaming using graniet/llm's streaming support
        Err(LlmError::Provider(
            "Streaming not yet implemented for UniversalLLMClient".to_string(),
        ))
    }

    async fn complete_with_tools(
        &self,
        messages: &[ChatMessage],
        _tools: &[ToolDefinition],
        config: &LlmConfig,
    ) -> Result<LlmResponse> {
        // For now, fall back to regular completion
        // TODO: Implement tool calling using graniet/llm's function calling
        self.complete(messages, config).await
    }

    fn name(&self) -> &'static str {
        // Return a static name based on provider
        match self.provider() {
            Some("ollama") => "ollama",
            Some("openai") => "openai",
            Some("anthropic") => "anthropic",
            Some("google") => "google",
            Some("deepseek") => "deepseek",
            Some("xai") => "xai",
            Some("groq") => "groq",
            Some("mistral") => "mistral",
            _ => "universal",
        }
    }

    fn available_models(&self) -> Vec<&'static str> {
        match self.provider() {
            Some("ollama") => vec![
                "llama3",
                "llama3.2",
                "gemma3",
                "mistral",
                "codellama",
                "phi3",
            ],
            Some("openai") => vec!["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
            Some("anthropic") => vec![
                "claude-3-opus-20240229",
                "claude-3-sonnet-20240229",
                "claude-3-haiku-20240307",
            ],
            Some("google") => vec!["gemini-pro", "gemini-1.5-pro"],
            _ => vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_universal_config_parsing() {
        let config = UniversalConfig::new("ollama::gemma3:latest");
        assert_eq!(config.provider(), Some("ollama"));
        assert_eq!(config.model_name(), "gemma3:latest");
    }

    #[test]
    fn test_universal_config_simple_model() {
        let config = UniversalConfig::new("openai::gpt-4o");
        assert_eq!(config.provider(), Some("openai"));
        assert_eq!(config.model_name(), "gpt-4o");
    }

    #[test]
    fn test_universal_config_builder() {
        let config = UniversalConfig::new("anthropic::claude-3-sonnet-20240229")
            .with_api_key("test-key")
            .with_temperature(0.7)
            .with_max_tokens(2048)
            .with_resilience(3);

        assert_eq!(config.api_key, Some("test-key".to_string()));
        assert_eq!(config.temperature, Some(0.7));
        assert_eq!(config.max_tokens, Some(2048));
        assert_eq!(config.resilient, Some(true));
        assert_eq!(config.resilient_attempts, Some(3));
    }

    #[test]
    fn test_parse_backend_valid() {
        assert!(matches!(parse_backend("ollama"), Ok(LLMBackend::Ollama)));
        assert!(matches!(parse_backend("openai"), Ok(LLMBackend::OpenAI)));
        assert!(matches!(
            parse_backend("anthropic"),
            Ok(LLMBackend::Anthropic)
        ));
        assert!(matches!(parse_backend("google"), Ok(LLMBackend::Google)));
    }

    #[test]
    fn test_parse_backend_invalid() {
        assert!(parse_backend("unknown").is_err());
    }

    #[test]
    fn test_api_key_env_vars() {
        assert_eq!(get_api_key_env_var("ollama"), None);
        assert_eq!(get_api_key_env_var("openai"), Some("OPENAI_API_KEY"));
        assert_eq!(get_api_key_env_var("anthropic"), Some("ANTHROPIC_API_KEY"));
        assert_eq!(get_api_key_env_var("google"), Some("GOOGLE_API_KEY"));
    }
}
