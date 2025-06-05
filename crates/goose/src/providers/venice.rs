use anyhow::Result;
use async_trait::async_trait;
use chrono::Utc;
use reqwest::{Certificate, Client, Identity, Response};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::fs::File;
use std::io::Read;
use std::time::Duration;

use super::base::{ConfigKey, Provider, ProviderMetadata, ProviderUsage, Usage};
use super::errors::ProviderError;
use crate::message::{Message, MessageContent};
use crate::model::ModelConfig;
use mcp_core::{tool::Tool, Role, ToolCall, ToolResult};

// ---------- Capability Flags ----------
#[derive(Debug)]
struct CapabilityFlags(String);

impl CapabilityFlags {
    fn from_json(value: &serde_json::Value) -> Self {
        let caps = &value["model_spec"]["capabilities"];
        let mut s = String::with_capacity(6);
        macro_rules! flag {
            ($json_key:literal, $letter:literal) => {
                if caps
                    .get($json_key)
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false)
                {
                    s.push($letter);
                }
            };
        }
        flag!("optimizedForCode", 'c'); // code
        flag!("supportsVision", 'v'); // vision
        flag!("supportsFunctionCalling", 'f');
        flag!("supportsResponseSchema", 's');
        flag!("supportsWebSearch", 'w');
        flag!("supportsReasoning", 'r');
        CapabilityFlags(s)
    }
}

impl std::fmt::Display for CapabilityFlags {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}]", self.0) // e.g. "[cvfsw]"
    }
}
// ---------- END Capability Flags ----------

// ---------- Helpers ----------
/// Return the raw model id (everything before the first space).
fn strip_flags(model: &str) -> &str {
    model.split_whitespace().next().unwrap_or(model)
}
// ---------- END Helpers ----------

pub const VENICE_DOC_URL: &str = "https://docs.venice.ai/";
pub const VENICE_DEFAULT_MODEL: &str = "llama-3.3-70b";
pub const VENICE_DEFAULT_HOST: &str = "https://api.venice.ai";
pub const VENICE_DEFAULT_BASE_PATH: &str = "api/v1/chat/completions";
pub const VENICE_DEFAULT_MODELS_PATH: &str = "api/v1/models";

// Fallback models to use when API is unavailable
const FALLBACK_MODELS: [&str; 3] = [
    "llama-3.2-3b",   // Small model with function calling
    "llama-3.3-70b",  // Default model with function calling
    "mistral-31-24b", // Another model with function calling
];

#[derive(Debug, Serialize, Deserialize)]
pub struct VeniceProvider {
    #[serde(skip)]
    client: Client,
    host: String,
    base_path: String,
    models_path: String,
    api_key: String,
    model: ModelConfig,
}

impl Default for VeniceProvider {
    fn default() -> Self {
        let model = ModelConfig::new(VENICE_DEFAULT_MODEL.to_string());
        VeniceProvider::from_env(model).expect("Failed to initialize Venice provider")
    }
}

impl VeniceProvider {
    pub fn from_env(mut model: ModelConfig) -> Result<Self> {
        let config = crate::config::Config::global();
        let api_key: String = config.get_secret("VENICE_API_KEY")?;
        let host: String = config
            .get_param("VENICE_HOST")
            .unwrap_or_else(|_| VENICE_DEFAULT_HOST.to_string());
        let base_path: String = config
            .get_param("VENICE_BASE_PATH")
            .unwrap_or_else(|_| VENICE_DEFAULT_BASE_PATH.to_string());
        let models_path: String = config
            .get_param("VENICE_MODELS_PATH")
            .unwrap_or_else(|_| VENICE_DEFAULT_MODELS_PATH.to_string());

        // Ensure we only keep the bare model id internally
        model.model_name = strip_flags(&model.model_name).to_string();

        let mut client_builder = Client::builder().timeout(Duration::from_secs(600));

        // Load client certificate and key
        if let (Ok(cert_path), Ok(key_path)) = (
            config.get_param::<String>("VENICE_CLIENT_CERTIFICATE_PATH"),
            config.get_param::<String>("VENICE_CLIENT_KEY_PATH"),
        ) {
            if !cert_path.is_empty() && !key_path.is_empty() {
                let mut cert_buf = Vec::new();
                File::open(&cert_path)
                    .map_err(|e| {
                        anyhow::anyhow!("Failed to open certificate file {}: {}", cert_path, e)
                    })?
                    .read_to_end(&mut cert_buf)
                    .map_err(|e| {
                        anyhow::anyhow!("Failed to read certificate file {}: {}", cert_path, e)
                    })?;
                let mut key_buf = Vec::new();
                File::open(&key_path)
                    .map_err(|e| anyhow::anyhow!("Failed to open key file {}: {}", key_path, e))?
                    .read_to_end(&mut key_buf)
                    .map_err(|e| anyhow::anyhow!("Failed to read key file {}: {}", key_path, e))?;
                let identity = Identity::from_pem(&[&cert_buf, &key_buf].concat())
                    .map_err(|e| anyhow::anyhow!("Failed to create identity from PEM: {}", e))?;
                client_builder = client_builder.identity(identity);
            }
        }

        // Load CA certificate
        if let Ok(ca_path) = config.get_param::<String>("VENICE_CERTIFICATE_AUTHORITY_PATH") {
            if !ca_path.is_empty() {
                let mut ca_buf = Vec::new();
                File::open(&ca_path)
                    .map_err(|e| anyhow::anyhow!("Failed to open CA file {}: {}", ca_path, e))?
                    .read_to_end(&mut ca_buf)
                    .map_err(|e| anyhow::anyhow!("Failed to read CA file {}: {}", ca_path, e))?;
                let ca_cert = Certificate::from_pem(&ca_buf)
                    .map_err(|e| anyhow::anyhow!("Failed to create CA certificate from PEM: {}", e))?;
                client_builder = client_builder.add_root_certificate(ca_cert);
            }
        }

        let client = client_builder
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to build HTTP client: {}", e))?;

        let instance = Self {
            client,
            host,
            base_path,
            models_path,
            api_key,
            model,
        };

        Ok(instance)
    }

    async fn post(&self, path: &str, body: &str) -> Result<Response, ProviderError> {
        let base_url = url::Url::parse(&self.host)
            .map_err(|e| ProviderError::RequestFailed(format!("Invalid base URL: {e}")))?;
        let url = base_url
            .join(path)
            .map_err(|e| ProviderError::RequestFailed(format!("Failed to construct URL: {e}")))?;
        // Choose GET for models endpoint, POST otherwise
        let method = if path.contains("models") {
            tracing::debug!("Using GET method for models endpoint");
            self.client.get(url.clone())
        } else {
            tracing::debug!("Using POST method for completions endpoint");
            self.client.post(url.clone())
        };

        // Log the request details
        tracing::debug!("Venice request URL: {}", url);
        tracing::debug!("Venice request body: {}", body);

        let response = method
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .body(body.to_string())
            .send()
            .await?;

        let status = response.status();
        tracing::debug!("Venice response status: {}", status);

        if !status.is_success() {
            // Read response body for more details on error
            let error_body = response.text().await.unwrap_or_default();

            // Log full error response for debugging
            tracing::debug!("Full Venice error response: {}", error_body);

            // Try to parse the error response
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&error_body) {
                // Print the full JSON error for better debugging
                println!(
                    "Venice API error response: {}",
                    serde_json::to_string_pretty(&json).unwrap_or_else(|_| json.to_string())
                );

                // Check for tool support errors
                if let Some(details) = json.get("details") {
                    // Specifically look for tool support issues
                    if let Some(tools) = details.get("tools") {
                        if let Some(errors) = tools.get("_errors") {
                            if errors.to_string().contains("not supported by this model") {
                                let model_name = self.model.model_name.clone();
                                return Err(ProviderError::RequestFailed(
                                    format!("The selected model '{}' does not support tool calls. Please select a model that supports tools, such as 'llama-3.3-70b' or 'mistral-31-24b'.", model_name)
                                ));
                            }
                        }
                    }
                }

                // Check for specific error message in context.issues
                if let Some(context) = json.get("context") {
                    if let Some(issues) = context.get("issues") {
                        if let Some(issues_array) = issues.as_array() {
                            for issue in issues_array {
                                if let Some(message) = issue.get("message").and_then(|m| m.as_str())
                                {
                                    if message.contains("tools is not supported by this model") {
                                        let model_name = self.model.model_name.clone();
                                        return Err(ProviderError::RequestFailed(
                                            format!("The selected model '{}' does not support tool calls. Please select a model that supports tools, such as 'llama-3.3-70b' or 'mistral-31-24b'.", model_name)
                                        ));
                                    }
                                }
                            }
                        }
                    }
                }

                // General error extraction
                if let Some(error_msg) = json.get("error").and_then(|e| e.as_str()) {
                    return Err(ProviderError::RequestFailed(format!(
                        "Venice API error: {}",
                        error_msg
                    )));
                }
            }

            // Fallback for unparseable errors
            return Err(ProviderError::RequestFailed(format!(
                "Venice API request failed with status code {}",
                status
            )));
        }

        Ok(response)
    }
}

#[async_trait]
impl Provider for VeniceProvider {
    fn metadata() -> ProviderMetadata {
        ProviderMetadata::new(
            "venice",
            "Venice.ai",
            "Venice.ai models (Llama, DeepSeek, Mistral) with function calling",
            VENICE_DEFAULT_MODEL,
            FALLBACK_MODELS.to_vec(),
            VENICE_DOC_URL,
            vec![
                ConfigKey::new("VENICE_API_KEY", true, true, None),
                ConfigKey::new("VENICE_HOST", true, false, Some(VENICE_DEFAULT_HOST)),
                ConfigKey::new(
                    "VENICE_BASE_PATH",
                    true,
                    false,
                    Some(VENICE_DEFAULT_BASE_PATH),
                ),
                ConfigKey::new(
                    "VENICE_MODELS_PATH",
                    true,
                    false,
                    Some(VENICE_DEFAULT_MODELS_PATH),
                ),
            ],
        )
    }

    fn get_model_config(&self) -> ModelConfig {
        self.model.clone()
    }

    async fn fetch_supported_models_async(&self) -> Result<Option<Vec<String>>, ProviderError> {
        // Fetch supported models via Venice API
        let base_url = url::Url::parse(&self.host)
            .map_err(|e| ProviderError::RequestFailed(format!("Invalid base URL: {}", e)))?;
        let models_url = base_url.join(&self.models_path).map_err(|e| {
            ProviderError::RequestFailed(format!("Failed to construct models URL: {}", e))
        })?;
        let response = self
            .client
            .get(models_url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await?;
        if !response.status().is_success() {
            return Err(ProviderError::RequestFailed(format!(
                "Venice API request failed with status {}",
                response.status()
            )));
        }
        let body = response.text().await?;
        let json: serde_json::Value = serde_json::from_str(&body)
            .map_err(|e| ProviderError::RequestFailed(format!("Failed to parse JSON: {}", e)))?;

        // Print legend once so users know what flags mean
        println!(
            "Capabilities:\n  c=code\n  f=function calls (goose supported models)\n  s=schema\n  v=vision\n  w=web search\n  r=reasoning"
        );

        let mut models = json["data"]
            .as_array()
            .ok_or_else(|| ProviderError::RequestFailed("No data field in JSON".to_string()))?
            .iter()
            .filter_map(|model| {
                let id = model["id"].as_str()?.to_owned();
                // Build flags from capabilities
                let flags = CapabilityFlags::from_json(model);
                // Only include models that support function calling (have 'f' flag)
                if flags.0.contains('f') {
                    Some(format!("{id} {flags}"))
                } else {
                    None
                }
            })
            .collect::<Vec<String>>();
        models.sort();
        Ok(Some(models))
    }

    #[tracing::instrument(
        skip(_system, messages, tools),
        fields(model_config, input, output, input_tokens, output_tokens, total_tokens)
    )]
    async fn complete(
        &self,
        _system: &str,
        messages: &[Message],
        tools: &[Tool],
    ) -> Result<(Message, ProviderUsage), ProviderError> {
        // Create properly formatted messages for Venice API
        let mut formatted_messages = Vec::new();

        // Add the system message if present
        if !_system.is_empty() {
            formatted_messages.push(json!({
                "role": "system",
                "content": _system
            }));
        }

        // Format regular messages according to Venice API requirements
        for msg in messages {
            // Venice API expects 'content' to be a string, not an array of MessageContent
            let content = match msg.role {
                Role::User => {
                    // For user messages, concatenate all text content
                    let text_content: String = msg
                        .content
                        .iter()
                        .filter_map(|c| c.as_text())
                        .collect::<Vec<_>>()
                        .join("\n");

                    // If we have text content, use it directly
                    if !text_content.is_empty() {
                        text_content
                    } else {
                        // Otherwise, try to get a reasonable string representation
                        msg.as_concat_text()
                    }
                }
                _ => {
                    // For assistant messages, handle possible tool calls
                    let has_tool_calls = msg
                        .content
                        .iter()
                        .any(|c| matches!(c, MessageContent::ToolRequest(_)));

                    if has_tool_calls {
                        // If there are tool calls, we'll handle them separately
                        // Just use an empty string for content
                        "".to_string()
                    } else {
                        // Otherwise use text content
                        msg.as_concat_text()
                    }
                }
            };

            // Create basic message with content as string
            let mut venice_msg = json!({
                "role": match msg.role {
                    Role::User => "user",
                    Role::Assistant => "assistant",
                },
                "content": content
            });

            // Add debug information to tracing
            tracing::debug!(
                "Venice message format: role={:?}, content_len={}, has_tool_calls={}",
                msg.role,
                content.len(),
                msg.content
                    .iter()
                    .any(|c| matches!(c, MessageContent::ToolRequest(_)))
            );

            // For assistant messages with tool calls, add them in Venice format
            if msg.role == Role::Assistant {
                let tool_calls: Vec<_> = msg
                    .content
                    .iter()
                    .filter_map(|c| c.as_tool_request())
                    .collect();

                if !tool_calls.is_empty() {
                    // Transform our tool calls to Venice format
                    let venice_tool_calls: Vec<Value> = tool_calls
                        .iter()
                        .filter_map(|tr| {
                            if let ToolResult::Ok(tool_call) = &tr.tool_call {
                                // Log tool call details for debugging
                                tracing::debug!(
                                    "Tool call conversion: id={}, name={}, args_len={}",
                                    tr.id,
                                    tool_call.name,
                                    tool_call.arguments.to_string().len()
                                );

                                // Convert to Venice format
                                Some(json!({
                                    "id": tr.id,
                                    "type": "function",
                                    "function": {
                                        "name": tool_call.name,
                                        "arguments": tool_call.arguments.to_string()
                                    }
                                }))
                            } else {
                                tracing::warn!("Skipping tool call with error: id={}", tr.id);
                                None
                            }
                        })
                        .collect();

                    if !venice_tool_calls.is_empty() {
                        tracing::debug!("Adding {} tool calls to message", venice_tool_calls.len());
                        venice_msg["tool_calls"] = json!(venice_tool_calls);
                    }
                }
            }

            // For tool messages with tool responses, add required tool_call_id
            // Check for tool responses regardless of role - they should have an ID
            // that corresponds to the tool call they're responding to
            {
                let tool_responses: Vec<_> = msg
                    .content
                    .iter()
                    .filter_map(|c| c.as_tool_response())
                    .collect();

                if !tool_responses.is_empty() && !tool_responses[0].id.is_empty() {
                    venice_msg["tool_call_id"] = json!(tool_responses[0].id);
                    // Venice expects tool messages to have 'role' = 'tool'
                    venice_msg["role"] = json!("tool");
                }
            }

            formatted_messages.push(venice_msg);
        }

        // Build Venice-specific payload
        let mut payload = json!({
            "model": strip_flags(&self.model.model_name),
            "messages": formatted_messages,
            "stream": false,
            "temperature": 0.7,
            "max_tokens": 2048,
        });

        if !tools.is_empty() {
            // Format tools specifically for Venice API
            let formatted_tools: Vec<serde_json::Value> = tools
                .iter()
                .map(|tool| {
                    // Format each tool in the expected Venice format
                    json!({
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.input_schema
                        }
                    })
                })
                .collect();

            payload["tools"] = json!(formatted_tools);
        }

        tracing::debug!("Sending request to Venice API");
        tracing::debug!("Venice request payload: {}", payload.to_string());

        // Send request
        let response = self.post(&self.base_path, &payload.to_string()).await?;

        // Parse the response
        let response_text = response.text().await?;
        let response_json: Value = serde_json::from_str(&response_text).map_err(|e| {
            ProviderError::RequestFailed(format!(
                "Failed to parse JSON: {}\nResponse: {}",
                e, response_text
            ))
        })?;

        // Handle tool calls from the response if present
        let tool_calls = response_json["choices"]
            .get(0)
            .and_then(|choice| choice["message"]["tool_calls"].as_array());

        if let Some(tool_calls) = tool_calls {
            if !tool_calls.is_empty() {
                // Extract tool calls and format for our internal model
                let mut content = Vec::new();

                for tool_call in tool_calls {
                    let id = tool_call["id"].as_str().unwrap_or("unknown").to_string();
                    let function = tool_call["function"].clone();
                    let name = function["name"].as_str().unwrap_or("unknown").to_string();

                    // Parse arguments string to Value if it's a string
                    let arguments = if let Some(args_str) = function["arguments"].as_str() {
                        serde_json::from_str::<Value>(args_str)
                            .unwrap_or(function["arguments"].clone())
                    } else {
                        function["arguments"].clone()
                    };

                    // Create a ToolCall using the function name and arguments
                    let tool_call = ToolCall { name, arguments };

                    // Create a ToolRequest MessageContent
                    let tool_request = MessageContent::tool_request(id, ToolResult::Ok(tool_call));

                    content.push(tool_request);
                }

                // Create message and add each content item
                let mut message = Message::assistant();
                for item in content {
                    message = message.with_content(item);
                }

                return Ok((
                    message,
                    ProviderUsage::new(
                        strip_flags(&self.model.model_name).to_string(),
                        Usage::default(),
                    ),
                ));
            }
        }

        // If we get here, it's a regular text response
        // Extract content
        let content = response_json["choices"]
            .get(0)
            .and_then(|choice| choice["message"]["content"].as_str())
            .ok_or_else(|| {
                tracing::error!("Invalid response format: {:?}", response_json);
                ProviderError::RequestFailed("Invalid response format: missing content".to_string())
            })?
            .to_string();

        // Create a vector with a single text content item
        let content = vec![MessageContent::text(content)];

        // Extract usage
        let usage_data = &response_json["usage"];
        let usage = Usage {
            input_tokens: usage_data["prompt_tokens"].as_i64().map(|v| v as i32),
            output_tokens: usage_data["completion_tokens"].as_i64().map(|v| v as i32),
            total_tokens: usage_data["total_tokens"].as_i64().map(|v| v as i32),
        };

        Ok((
            Message {
                role: Role::Assistant,
                created: Utc::now().timestamp(),
                content,
            },
            ProviderUsage::new(strip_flags(&self.model.model_name).to_string(), usage),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metadata_structure() {
        let metadata = VeniceProvider::metadata();

        assert_eq!(metadata.default_model, "llama-3.3-70b");
        assert!(!metadata.known_models.is_empty());

        assert_eq!(metadata.config_keys.len(), 4);
        assert_eq!(metadata.config_keys[0].name, "VENICE_API_KEY");
        assert_eq!(metadata.config_keys[1].name, "VENICE_HOST");
        assert_eq!(metadata.config_keys[2].name, "VENICE_BASE_PATH");
        assert_eq!(metadata.config_keys[3].name, "VENICE_MODELS_PATH");
    }
}
