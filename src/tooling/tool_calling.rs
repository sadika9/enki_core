use crate::tooling::types::{Tool, ToolContext, ToolRegistry};
use serde_json::{Value, json};

pub struct ToolCallRegistry {
    tools: ToolRegistry,
}

impl ToolCallRegistry {
    pub fn new(tools: ToolRegistry) -> Self {
        Self { tools }
    }

    pub fn get(&self, tool_name: &str) -> Option<&dyn Tool> {
        self.tools.get(tool_name).map(Box::as_ref)
    }

    pub fn catalog_json(&self) -> Value {
        let mut map = serde_json::Map::new();

        for tool in self.tools.values() {
            map.insert(tool.name().to_string(), tool.as_catalog_entry());
        }

        Value::Object(map)
    }

    pub fn tools_payload(&self) -> Vec<Value> {
        self.tools.values().map(|tool| tool.as_tool_payload()).collect()
    }
}

pub trait ToolExecutor {
    fn execute(
        &self,
        registry: &ToolCallRegistry,
        tool_name: &str,
        args: &Value,
        ctx: &ToolContext,
    ) -> String;

    fn build_tool_message(
        &self,
        registry: &ToolCallRegistry,
        tool_name: &str,
        args: &Value,
        ctx: &ToolContext,
    ) -> Value {
        let content = self.execute(registry, tool_name, args, ctx);

        json!({
            "role": "tool",
            "tool_name": tool_name,
            "content": content
        })
    }
}

pub struct RegistryToolExecutor;

impl ToolExecutor for RegistryToolExecutor {
    fn execute(
        &self,
        registry: &ToolCallRegistry,
        tool_name: &str,
        args: &Value,
        ctx: &ToolContext,
    ) -> String {
        match registry.get(tool_name) {
            Some(tool) => tool.execute(args, ctx),
            None => format!("Unknown tool: {tool_name}"),
        }
    }
}
