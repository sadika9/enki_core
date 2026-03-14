use crate::tooling::types::*;
macro_rules! define_tool {
    (
        $tool_type:ty,
        name: $name:literal,
        description: $description:literal,
        parameters: $parameters:expr,
        |$args:ident, $ctx:ident| $body:block
    ) => {
        impl Tool for $tool_type {
            fn name(&self) -> &'static str {
                $name
            }

            fn description(&self) -> &'static str {
                $description
            }

            fn parameters(&self) -> Value {
                $parameters
            }

            fn execute(&self, $args: &Value, $ctx: &ToolContext) -> String $body
        }
    };
}

macro_rules! register_tools {
    ($($tool_type:ident),+ $(,)?) => {{
        let mut registry: ToolRegistry = BTreeMap::new();
        $(
            let tool: Box<dyn Tool> = Box::new($tool_type);
            registry.insert(tool.name().to_string(), tool);
        )+
        registry
    }};
}
