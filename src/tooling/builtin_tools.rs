use crate::tooling::types::*;
use serde_json::{Value, json};
use std::path::Path;
use tokio::fs;
use tokio::process::Command;

pub struct ReadFileTool;
pub struct WriteFileTool;
pub struct ExecTool;

define_tool!(
    ReadFileTool,
    name: "read_file",
    description: "Read file content.",
    parameters: json!({
        "type": "object",
        "properties": {
            "path": { "type": "string" }
        },
        "required": ["path"]
    }),
    |args, _ctx| {
        let path = args
            .get("path")
            .and_then(Value::as_str)
            .unwrap_or_default();

        match fs::read_to_string(Path::new(path)).await {
            Ok(content) => content,
            Err(_) => "File not found.".to_string(),
        }
    }
);

define_tool!(
    WriteFileTool,
    name: "write_file",
    description: "Write content to file.",
    parameters: json!({
        "type": "object",
        "properties": {
            "path": { "type": "string" },
            "content": { "type": "string" }
        },
        "required": ["path", "content"]
    }),
    |args, _ctx| {
        let path = args
            .get("path")
            .and_then(Value::as_str)
            .unwrap_or_default();

        let content = args
            .get("content")
            .and_then(Value::as_str)
            .unwrap_or_default();

        match fs::write(Path::new(path), content).await {
            Ok(_) => "File written.".to_string(),
            Err(e) => format!("Error: {e}"),
        }
    }
);

define_tool!(
    ExecTool,
    name: "exec",
    description: "Execute shell command safely.",
    parameters: json!({
        "type": "object",
        "properties": {
            "cmd": { "type": "string" }
        },
        "required": ["cmd"]
    }),
    |args, ctx| {
        let cmd = args
            .get("cmd")
            .and_then(Value::as_str)
            .unwrap_or_default();

        match Command::new("sh")
            .arg("-c")
            .arg(cmd)
            .current_dir(&ctx.workspace_dir)
            .output()
            .await
        {
            Ok(result) => {
                let stdout = String::from_utf8_lossy(&result.stdout);
                let stderr = String::from_utf8_lossy(&result.stderr);
                let rc = result.status.code().unwrap_or(-1);

                format!("stdout: {stdout}\nstderr: {stderr}\nrc: {rc}")
            }
            Err(e) => format!("Exec error: {e}"),
        }
    }
);
