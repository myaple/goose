use super::base::{ConfigKey, Provider, ProviderMetadata, ProviderUsage, Usage};
use super::errors::ProviderError;
use super::utils::{get_model, handle_response_openai_compat};
use crate::message::Message;
use crate::model::ModelConfig;
use crate::providers::formats::openai::{create_request, get_usage, response_to_message};
use anyhow::Result;
use async_trait::async_trait;
use mcp_core::tool::Tool;
use reqwest::{Certificate, Client, Identity};
use serde_json::Value;
use std::fs::File;
use std::io::Read;
use std::time::Duration;
use url::Url;

pub const OLLAMA_HOST: &str = "localhost";
pub const OLLAMA_DEFAULT_PORT: u16 = 11434;
pub const OLLAMA_DEFAULT_MODEL: &str = "qwen2.5";
// Ollama can run many models, we only provide the default
pub const OLLAMA_KNOWN_MODELS: &[&str] = &[OLLAMA_DEFAULT_MODEL];
pub const OLLAMA_DOC_URL: &str = "https://ollama.com/library";

#[derive(serde::Serialize)]
pub struct OllamaProvider {
    #[serde(skip)]
    client: Client,
    host: String,
    model: ModelConfig,
}

impl Default for OllamaProvider {
    fn default() -> Self {
        let model = ModelConfig::new(OllamaProvider::metadata().default_model);
        OllamaProvider::from_env(model).expect("Failed to initialize Ollama provider")
    }
}

impl OllamaProvider {
    pub fn from_env(model: ModelConfig) -> Result<Self> {
        let config = crate::config::Config::global();
        let host: String = config
            .get_param("OLLAMA_HOST")
            .unwrap_or_else(|_| OLLAMA_HOST.to_string());

        let mut client_builder = Client::builder().timeout(Duration::from_secs(600));

        // Load client certificate and key
        if let (Ok(cert_path), Ok(key_path)) = (
            config.get_param::<String>("OLLAMA_CLIENT_CERTIFICATE_PATH"),
            config.get_param::<String>("OLLAMA_CLIENT_KEY_PATH"),
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
        if let Ok(ca_path) = config.get_param::<String>("OLLAMA_CERTIFICATE_AUTHORITY_PATH") {
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
            model,
        })
    }

    /// Get the base URL for Ollama API calls
    fn get_base_url(&self) -> Result<Url, ProviderError> {
        // OLLAMA_HOST is sometimes just the 'host' or 'host:port' without a scheme
        let base = if self.host.starts_with("http://") || self.host.starts_with("https://") {
            self.host.clone()
        } else {
            format!("http://{}", self.host)
        };

        let mut base_url = Url::parse(&base)
            .map_err(|e| ProviderError::RequestFailed(format!("Invalid base URL: {e}")))?;

        // Set the default port if missing
        // Don't add default port if:
        // 1. URL explicitly ends with standard ports (:80 or :443)
        // 2. URL uses HTTPS (which implicitly uses port 443)
        let explicit_default_port = self.host.ends_with(":80") || self.host.ends_with(":443");
        let is_https = base_url.scheme() == "https";

        if base_url.port().is_none() && !explicit_default_port && !is_https {
            base_url.set_port(Some(OLLAMA_DEFAULT_PORT)).map_err(|_| {
                ProviderError::RequestFailed("Failed to set default port".to_string())
            })?;
        }

        Ok(base_url)
    }

    async fn post(&self, payload: Value) -> Result<Value, ProviderError> {
        // TODO: remove this later when the UI handles provider config refresh
        let base_url = self.get_base_url()?;

        let url = base_url.join("v1/chat/completions").map_err(|e| {
            ProviderError::RequestFailed(format!("Failed to construct endpoint URL: {e}"))
        })?;

        let response = self.client.post(url).json(&payload).send().await?;

        handle_response_openai_compat(response).await
    }
}

#[async_trait]
impl Provider for OllamaProvider {
    fn metadata() -> ProviderMetadata {
        ProviderMetadata::new(
            "ollama",
            "Ollama",
            "Local open source models",
            OLLAMA_DEFAULT_MODEL,
            OLLAMA_KNOWN_MODELS.to_vec(),
            OLLAMA_DOC_URL,
            vec![ConfigKey::new(
                "OLLAMA_HOST",
                true,
                false,
                Some(OLLAMA_HOST),
            ),
                ConfigKey::new("CLIENT_CERTIFICATE_PATH", false, false, None),
                ConfigKey::new("CLIENT_KEY_PATH", false, false, None),
                ConfigKey::new("CERTIFICATE_AUTHORITY_PATH", false, false, None),
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
