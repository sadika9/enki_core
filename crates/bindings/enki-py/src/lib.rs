use core_next::agent::{Agent, AgentDefinition};
use core_next::tooling::tool_calling::RegistryToolExecutor;
use std::path::PathBuf;
use std::sync::{mpsc, Mutex};
use std::thread;

struct RunRequest {
    session_id: String,
    user_message: String,
    reply_tx: tokio::sync::oneshot::Sender<String>,
}

pub struct EnkiAgent {
    request_tx: Mutex<mpsc::Sender<RunRequest>>,
}

impl EnkiAgent {
    pub fn new(
        name: String,
        system_prompt_preamble: String,
        model: String,
        max_iterations: u32,
        workspace_home: Option<String>,
    ) -> Self {
        let definition = AgentDefinition {
            name,
            system_prompt_preamble,
            model,
            max_iterations: max_iterations as usize,
        };
        let workspace_home = workspace_home.map(PathBuf::from);
        let (request_tx, request_rx) = mpsc::channel::<RunRequest>();

        thread::spawn(move || {
            let runtime = match tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            {
                Ok(runtime) => runtime,
                Err(error) => {
                    let message = format!("Initialization error: failed to create tokio runtime: {error}");
                    for request in request_rx {
                        let _ = request.reply_tx.send(message.clone());
                    }
                    return;
                }
            };

            let agent = match runtime.block_on(Agent::with_definition_executor_and_workspace(
                definition,
                Box::new(RegistryToolExecutor),
                workspace_home,
            )) {
                Ok(agent) => agent,
                Err(error) => {
                    let message = format!("Initialization error: {error}");
                    for request in request_rx {
                        let _ = request.reply_tx.send(message.clone());
                    }
                    return;
                }
            };

            for request in request_rx {
                let response = runtime.block_on(agent.run(&request.session_id, &request.user_message));
                let _ = request.reply_tx.send(response);
            }
        });

        Self {
            request_tx: Mutex::new(request_tx),
        }
    }

    pub async fn run(&self, session_id: String, user_message: String) -> String {
        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
        let request = RunRequest {
            session_id,
            user_message,
            reply_tx,
        };

        let send_result = self
            .request_tx
            .lock()
            .map_err(|_| "Worker error: request mutex poisoned".to_string())
            .and_then(|sender| {
                sender
                    .send(request)
                    .map_err(|_| "Worker error: agent worker has stopped".to_string())
            });

        if let Err(message) = send_result {
            return message;
        }

        reply_rx
            .await
            .unwrap_or_else(|_| "Worker error: agent worker dropped reply channel".to_string())
    }
}

uniffi::include_scaffolding!("enki");
