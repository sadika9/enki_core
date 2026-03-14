#[macro_use]
mod macros;
pub mod agent;
pub mod message;
pub mod runtime;
pub mod tooling;
pub mod llm;

use crate::runtime::{CliChannel, RuntimeBuilder};
use crate::tooling::builtin_tools::{ExecTool, ReadFileTool, WriteFileTool};
use crate::tooling::types::*;
use std::env;

fn build_tools() -> ToolRegistry {
    register_tools![ReadFileTool, WriteFileTool, ExecTool]
}

#[tokio::main]
async fn main() {
    let runtime = match RuntimeBuilder::for_default_agent()
        .with_model("ollama::qwen3.5")
        .with_workspace_home("./")
        .build()
        .await
    {
        Ok(runtime) => runtime,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };

    let mut channel = match CliChannel::from_args(env::args().collect()) {
        Ok(channel) => channel,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };

    if let Err(e) = runtime.serve_channel(&mut channel).await {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
