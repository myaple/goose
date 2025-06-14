use anyhow::Result;
use async_trait::async_trait;
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::time::Duration;

use super::base::{ConfigKey, Provider, ProviderMetadata, ProviderUsage};
use super::errors::ProviderError;
use super::formats::snowflake::{create_request, get_usage, response_to_message};
use super::utils::{get_model, ImageFormat};
use crate::config::ConfigError;
use crate::message::Message;
use crate::model::ModelConfig;
use mcp_core::tool::Tool;
use url::Url;

pub const SNOWFLAKE_DEFAULT_MODEL: &str = "claude-3-7-sonnet";
pub const SNOWFLAKE_KNOWN_MODELS: &[&str] = &["claude-3-7-sonnet", "claude-3-5-sonnet"];

pub const SNOWFLAKE_DOC_URL: &str =
    "https://docs.snowflake.com/user-guide/snowflake-cortex/aisql#choosing-a-model";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SnowflakeAuth {
    Token(String),
}

impl SnowflakeAuth {
    pub fn token(token: String) -> Self {
        Self::Token(token)
    }
}

#[derive(Debug, serde::Serialize)]
pub struct SnowflakeProvider {
    #[serde(skip)]
    client: Client,
    host: String,
    auth: SnowflakeAuth,
    model: ModelConfig,
    image_format: ImageFormat,
}

impl Default for SnowflakeProvider {
    fn default() -> Self {
        let model = ModelConfig::new(SnowflakeProvider::metadata().default_model);
        SnowflakeProvider::from_env(model).expect("Failed to initialize Snowflake provider")
    }
}

impl SnowflakeProvider {
    pub fn from_env(model: ModelConfig) -> Result<Self> {
        let config = crate::config::Config::global();
        let mut host: Result<String, ConfigError> = config.get_param("SNOWFLAKE_HOST");
        if host.is_err() {
            host = config.get_secret("SNOWFLAKE_HOST")
        }
        if host.is_err() {
            return Err(ConfigError::NotFound(
                "Did not find SNOWFLAKE_HOST in either config file or keyring".to_string(),
            )
            .into());
        }

        let mut host = host?;

        // Convert host to lowercase
        host = host.to_lowercase();

        // Ensure host ends with snowflakecomputing.com
        if !host.ends_with("snowflakecomputing.com") {
            host = format!("{}.snowflakecomputing.com", host);
        }

        let mut token: Result<String, ConfigError> = config.get_param("SNOWFLAKE_TOKEN");

        if token.is_err() {
            token = config.get_secret("SNOWFLAKE_TOKEN")
        }

        if token.is_err() {
            return Err(ConfigError::NotFound(
                "Did not find SNOWFLAKE_TOKEN in either config file or keyring".to_string(),
            )
            .into());
        }

        let client = Client::builder()
            .timeout(Duration::from_secs(600))
            .build()?;

        // Use token-based authentication
        let api_key = token?;
        Ok(Self {
            client,
            host,
            auth: SnowflakeAuth::token(api_key),
            model,
            image_format: ImageFormat::OpenAi,
        })
    }

    async fn ensure_auth_header(&self) -> Result<String> {
        match &self.auth {
            // https://docs.snowflake.com/en/developer-guide/snowflake-rest-api/authentication#using-a-programmatic-access-token-pat
            SnowflakeAuth::Token(token) => Ok(format!("Bearer {}", token)),
        }
    }

    async fn post(&self, payload: Value) -> Result<Value, ProviderError> {
        let base_url_str =
            if !self.host.starts_with("https://") && !self.host.starts_with("http://") {
                format!("https://{}", self.host)
            } else {
                self.host.clone()
            };
        let base_url = Url::parse(&base_url_str)
            .map_err(|e| ProviderError::RequestFailed(format!("Invalid base URL: {e}")))?;
        let path = "api/v2/cortex/inference:complete";
        let url = base_url.join(path).map_err(|e| {
            ProviderError::RequestFailed(format!("Failed to construct endpoint URL: {e}"))
        })?;

        let auth_header = self.ensure_auth_header().await?;
        let response = self
            .client
            .post(url)
            .header("Authorization", auth_header)
            .header("User-Agent", "Goose")
            .json(&payload)
            .send()
            .await?;

        let status = response.status();

        let payload_text: String = response.text().await.ok().unwrap_or_default();

        if status == StatusCode::OK {
            if let Ok(payload) = serde_json::from_str::<Value>(&payload_text) {
                if payload.get("code").is_some() {
                    let code = payload
                        .get("code")
                        .and_then(|c| c.as_str())
                        .unwrap_or("Unknown code");
                    let message = payload
                        .get("message")
                        .and_then(|m| m.as_str())
                        .unwrap_or("Unknown message");
                    return Err(ProviderError::RequestFailed(format!(
                        "{} - {}",
                        code, message
                    )));
                }
            }
        }

        let lines = payload_text.lines().collect::<Vec<_>>();

        let mut text = String::new();
        let mut tool_name = String::new();
        let mut tool_input = String::new();
        let mut tool_use_id = String::new();
        for line in lines.iter() {
            if line.is_empty() {
                continue;
            }

            let json_str = match line.strip_prefix("data: ") {
                Some(s) => s,
                None => continue,
            };

            if let Ok(json_line) = serde_json::from_str::<Value>(json_str) {
                let choices = match json_line.get("choices").and_then(|c| c.as_array()) {
                    Some(choices) => choices,
                    None => {
                        continue;
                    }
                };

                let choice = match choices.first() {
                    Some(choice) => choice,
                    None => {
                        continue;
                    }
                };

                let delta = match choice.get("delta") {
                    Some(delta) => delta,
                    None => {
                        continue;
                    }
                };

                // Track if we found text in content_list to avoid duplication
                let mut found_text_in_content_list = false;

                // Handle content_list array first
                if let Some(content_list) = delta.get("content_list").and_then(|cl| cl.as_array()) {
                    for content_item in content_list {
                        match content_item.get("type").and_then(|t| t.as_str()) {
                            Some("text") => {
                                if let Some(text_content) =
                                    content_item.get("text").and_then(|t| t.as_str())
                                {
                                    text.push_str(text_content);
                                    found_text_in_content_list = true;
                                }
                            }
                            Some("tool_use") => {
                                if let Some(tool_id) =
                                    content_item.get("tool_use_id").and_then(|id| id.as_str())
                                {
                                    tool_use_id.push_str(tool_id);
                                }
                                if let Some(name) =
                                    content_item.get("name").and_then(|n| n.as_str())
                                {
                                    tool_name.push_str(name);
                                }
                                if let Some(input) =
                                    content_item.get("input").and_then(|i| i.as_str())
                                {
                                    tool_input.push_str(input);
                                }
                            }
                            _ => {
                                // Handle content items without explicit type but with tool information
                                if let Some(name) =
                                    content_item.get("name").and_then(|n| n.as_str())
                                {
                                    tool_name.push_str(name);
                                }
                                if let Some(tool_id) =
                                    content_item.get("tool_use_id").and_then(|id| id.as_str())
                                {
                                    tool_use_id.push_str(tool_id);
                                }
                                if let Some(input) =
                                    content_item.get("input").and_then(|i| i.as_str())
                                {
                                    tool_input.push_str(input);
                                }
                            }
                        }
                    }
                }

                // Handle direct content field (for text) only if we didn't find text in content_list
                if !found_text_in_content_list {
                    if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
                        text.push_str(content);
                    }
                }
            }
        }

        // Build the appropriate response structure
        let mut content_list = Vec::new();

        // Add text content if available
        if !text.is_empty() {
            content_list.push(json!({
                "type": "text",
                "text": text
            }));
        }

        // Add tool use content only if we have complete tool information
        if !tool_use_id.is_empty() && !tool_name.is_empty() {
            // Parse tool input as JSON if it's not empty
            let parsed_input = if tool_input.is_empty() {
                json!({})
            } else {
                serde_json::from_str::<Value>(&tool_input)
                    .unwrap_or_else(|_| json!({"raw_input": tool_input}))
            };

            content_list.push(json!({
                "type": "tool_use",
                "tool_use_id": tool_use_id,
                "name": tool_name,
                "input": parsed_input
            }));
        }

        // Ensure we always have at least some content
        if content_list.is_empty() {
            content_list.push(json!({
                "type": "text",
                "text": ""
            }));
        }

        let answer_payload = json!({
            "role": "assistant",
            "content": text,
            "content_list": content_list
        });

        match status {
            StatusCode::OK => Ok(answer_payload),
            StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => {
                // Extract a clean error message from the response if available
                let error_msg = payload_text
                    .lines()
                    .find(|line| line.contains("\"message\""))
                    .and_then(|line| {
                        let json_str = line.strip_prefix("data: ").unwrap_or(line);
                        serde_json::from_str::<Value>(json_str).ok()
                    })
                    .and_then(|json| {
                        json.get("message")
                            .and_then(|m| m.as_str())
                            .map(|s| s.to_string())
                    })
                    .unwrap_or_else(|| "Invalid credentials".to_string());

                Err(ProviderError::Authentication(format!(
                    "Authentication failed. Please check your SNOWFLAKE_TOKEN and SNOWFLAKE_HOST configuration. Error: {}", 
                    error_msg
                )))
            }
            StatusCode::BAD_REQUEST => {
                // Snowflake provides a generic 'error' but also includes 'external_model_message' which is provider specific
                // We try to extract the error message from the payload and check for phrases that indicate context length exceeded
                let payload_str = payload_text.to_lowercase();
                let check_phrases = [
                    "too long",
                    "context length",
                    "context_length_exceeded",
                    "reduce the length",
                    "token count",
                    "exceeds",
                    "exceed context limit",
                    "input length",
                    "max_tokens",
                    "decrease input length",
                    "context limit",
                ];
                if check_phrases.iter().any(|c| payload_str.contains(c)) {
                    return Err(ProviderError::ContextLengthExceeded("Request exceeds maximum context length. Please reduce the number of messages or content size.".to_string()));
                }

                // Try to parse a clean error message from the response
                let error_msg = if let Ok(json) = serde_json::from_str::<Value>(&payload_text) {
                    json.get("message")
                        .and_then(|m| m.as_str())
                        .map(|s| s.to_string())
                        .or_else(|| {
                            json.get("external_model_message")
                                .and_then(|ext| ext.get("message"))
                                .and_then(|m| m.as_str())
                                .map(|s| s.to_string())
                        })
                        .unwrap_or_else(|| "Bad request".to_string())
                } else {
                    "Bad request".to_string()
                };

                tracing::debug!(
                    "Provider request failed with status: {}. Response: {}",
                    status,
                    payload_text
                );
                Err(ProviderError::RequestFailed(format!(
                    "Request failed: {}",
                    error_msg
                )))
            }
            StatusCode::TOO_MANY_REQUESTS => Err(ProviderError::RateLimitExceeded(
                "Rate limit exceeded. Please try again later.".to_string(),
            )),
            StatusCode::INTERNAL_SERVER_ERROR | StatusCode::SERVICE_UNAVAILABLE => {
                Err(ProviderError::ServerError(
                    "Snowflake service is temporarily unavailable. Please try again later."
                        .to_string(),
                ))
            }
            _ => {
                tracing::debug!(
                    "Provider request failed with status: {}. Response: {}",
                    status,
                    payload_text
                );
                Err(ProviderError::RequestFailed(format!(
                    "Request failed with status: {}",
                    status
                )))
            }
        }
    }
}

#[async_trait]
impl Provider for SnowflakeProvider {
    fn metadata() -> ProviderMetadata {
        ProviderMetadata::new(
            "snowflake",
            "Snowflake",
            "Access the latest models using Snowflake Cortex services.",
            SNOWFLAKE_DEFAULT_MODEL,
            SNOWFLAKE_KNOWN_MODELS.to_vec(),
            SNOWFLAKE_DOC_URL,
            vec![
                ConfigKey::new("SNOWFLAKE_HOST", true, false, None),
                ConfigKey::new("SNOWFLAKE_TOKEN", true, true, None),
            ],
        )
    }

    fn get_model_config(&self) -> ModelConfig {
        self.model.clone()
    }

    #[tracing::instrument(
        skip(self, system, messages, tools),
        fields(model_config, input, output, input_tokens, output_tokens, total_tokens)
    )]
    async fn complete(
        &self,
        system: &str,
        messages: &[Message],
        tools: &[Tool],
    ) -> Result<(Message, ProviderUsage), ProviderError> {
        let payload = create_request(&self.model, system, messages, tools)?;

        let response = self.post(payload.clone()).await?;

        // Parse response
        let message = response_to_message(response.clone())?;
        let usage = get_usage(&response)?;
        let model = get_model(&response);
        super::utils::emit_debug_trace(&self.model, &payload, &response, &usage);

        Ok((message, ProviderUsage::new(model, usage)))
    }
}
