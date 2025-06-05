use super::errors::ProviderError;
use crate::message::Message;
use crate::model::ModelConfig;
use crate::providers::base::{ConfigKey, Provider, ProviderMetadata, ProviderUsage, Usage};
use crate::providers::formats::openai::{create_request, get_usage, response_to_message};
use crate::providers::utils::get_model;
use anyhow::Result;
use async_trait::async_trait;
use mcp_core::Tool;
use reqwest::{Certificate, Client, Identity, StatusCode};
use serde_json::Value;
use std::fs::File;
use std::io::Read;
use std::time::Duration;
use url::Url;

pub const GROQ_API_HOST: &str = "https://api.groq.com";
pub const GROQ_DEFAULT_MODEL: &str = "llama-3.3-70b-versatile";
pub const GROQ_KNOWN_MODELS: &[&str] = &["gemma2-9b-it", "llama-3.3-70b-versatile"];

pub const GROQ_DOC_URL: &str = "https://console.groq.com/docs/models";

#[derive(serde::Serialize)]
pub struct GroqProvider {
    #[serde(skip)]
    client: Client,
    host: String,
    api_key: String,
    model: ModelConfig,
}

impl Default for GroqProvider {
    fn default() -> Self {
        let model = ModelConfig::new(GroqProvider::metadata().default_model);
        GroqProvider::from_env(model).expect("Failed to initialize Groq provider")
    }
}

impl GroqProvider {
    pub fn from_env(model: ModelConfig) -> Result<Self> {
        let config = crate::config::Config::global();
        let api_key: String = config.get_secret("GROQ_API_KEY")?;
        let host: String = config
            .get_param("GROQ_HOST")
            .unwrap_or_else(|_| GROQ_API_HOST.to_string());

        let mut client_builder = Client::builder().timeout(Duration::from_secs(600));

        // Load client certificate and key
        if let (Ok(cert_path), Ok(key_path)) = (
            config.get_param::<String>("GROQ_CLIENT_CERTIFICATE_PATH"),
            config.get_param::<String>("GROQ_CLIENT_KEY_PATH"),
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
        if let Ok(ca_path) = config.get_param::<String>("GROQ_CERTIFICATE_AUTHORITY_PATH") {
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

        Ok(Self {
            client,
            host,
            api_key,
            model,
        })
    }

    async fn post(&self, payload: Value) -> anyhow::Result<Value, ProviderError> {
        let base_url = Url::parse(&self.host)
            .map_err(|e| ProviderError::RequestFailed(format!("Invalid base URL: {e}")))?;
        let url = base_url.join("openai/v1/chat/completions").map_err(|e| {
            ProviderError::RequestFailed(format!("Failed to construct endpoint URL: {e}"))
        })?;

        let response = self
            .client
            .post(url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&payload)
            .send()
            .await?;

        let status = response.status();
        let payload: Option<Value> = response.json().await.ok();

        match status {
            StatusCode::OK => payload.ok_or_else( || ProviderError::RequestFailed("Response body is not valid JSON".to_string()) ),
            StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => {
                Err(ProviderError::Authentication(format!("Authentication failed. Please ensure your API keys are valid and have the required permissions. \
                    Status: {}. Response: {:?}", status, payload)))
            }
            StatusCode::PAYLOAD_TOO_LARGE => {
                Err(ProviderError::ContextLengthExceeded(format!("{:?}", payload)))
            }
            StatusCode::TOO_MANY_REQUESTS => {
                Err(ProviderError::RateLimitExceeded(format!("{:?}", payload)))
            }
            StatusCode::INTERNAL_SERVER_ERROR | StatusCode::SERVICE_UNAVAILABLE => {
                Err(ProviderError::ServerError(format!("{:?}", payload)))
            }
            _ => {
                tracing::debug!(
                    "{}", format!("Provider request failed with status: {}. Payload: {:?}", status, payload)
                );
                Err(ProviderError::RequestFailed(format!("Request failed with status: {}", status)))
            }
        }
    }
}

#[async_trait]
impl Provider for GroqProvider {
    fn metadata() -> ProviderMetadata {
        ProviderMetadata::new(
            "groq",
            "Groq",
            "Fast inference with Groq hardware",
            GROQ_DEFAULT_MODEL,
            GROQ_KNOWN_MODELS.to_vec(),
            GROQ_DOC_URL,
            vec![
                ConfigKey::new("GROQ_API_KEY", true, true, None),
                ConfigKey::new("GROQ_HOST", false, false, Some(GROQ_API_HOST)),
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
    ) -> anyhow::Result<(Message, ProviderUsage), ProviderError> {
        let payload = create_request(
            &self.model,
            system,
            messages,
            tools,
            &super::utils::ImageFormat::OpenAi,
        )?;

        let response = self.post(payload.clone()).await?;

        let message = response_to_message(response.clone())?;
        let usage = match get_usage(&response) {
            Ok(usage) => usage,
            Err(ProviderError::UsageError(e)) => {
                tracing::debug!("Failed to get usage data: {}", e);
                Usage::default()
            }
            Err(e) => return Err(e),
        };
        let model = get_model(&response);
        super::utils::emit_debug_trace(&self.model, &payload, &response, &usage);
        Ok((message, ProviderUsage::new(model, usage)))
    }
}
