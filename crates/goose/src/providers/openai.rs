use anyhow::Result;
use async_trait::async_trait;
use reqwest::{Certificate, Client, Identity};
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::time::Duration;

use super::base::{ConfigKey, Provider, ProviderMetadata, ProviderUsage, Usage};
use super::embedding::{EmbeddingCapable, EmbeddingRequest, EmbeddingResponse};
use super::errors::ProviderError;
use super::formats::openai::{create_request, get_usage, response_to_message};
use super::utils::{emit_debug_trace, get_model, handle_response_openai_compat, ImageFormat};
use crate::message::Message;
use crate::model::ModelConfig;
use mcp_core::tool::Tool;

pub const OPEN_AI_DEFAULT_MODEL: &str = "gpt-4o";
pub const OPEN_AI_KNOWN_MODELS: &[&str] = &[
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-3.5-turbo",
    "o1",
    "o3",
    "o4-mini",
];

pub const OPEN_AI_DOC_URL: &str = "https://platform.openai.com/docs/models";

#[derive(Debug, serde::Serialize)]
pub struct OpenAiProvider {
    #[serde(skip)]
    client: Client,
    host: String,
    base_path: String,
    api_key: String,
    organization: Option<String>,
    project: Option<String>,
    model: ModelConfig,
    custom_headers: Option<HashMap<String, String>>,
}

impl Default for OpenAiProvider {
    fn default() -> Self {
        let model = ModelConfig::new(OpenAiProvider::metadata().default_model);
        OpenAiProvider::from_env(model).expect("Failed to initialize OpenAI provider")
    }
}

impl OpenAiProvider {
    pub fn from_env(model: ModelConfig) -> Result<Self> {
        let config = crate::config::Config::global();
        let api_key: String = config.get_secret("OPENAI_API_KEY")?;
        let host: String = config
            .get_param("OPENAI_HOST")
            .unwrap_or_else(|_| "https://api.openai.com".to_string());
        let base_path: String = config
            .get_param("OPENAI_BASE_PATH")
            .unwrap_or_else(|_| "v1/chat/completions".to_string());
        let organization: Option<String> = config.get_param("OPENAI_ORGANIZATION").ok();
        let project: Option<String> = config.get_param("OPENAI_PROJECT").ok();
        let custom_headers: Option<HashMap<String, String>> = config
            .get_secret("OPENAI_CUSTOM_HEADERS")
            .or_else(|_| config.get_param("OPENAI_CUSTOM_HEADERS"))
            .ok()
            .map(parse_custom_headers);
        let timeout_secs: u64 = config.get_param("OPENAI_TIMEOUT").unwrap_or(600);

        let mut client_builder = Client::builder().timeout(Duration::from_secs(timeout_secs));

        // Load client certificate and key
        if let (Ok(cert_path), Ok(key_path)) = (
            config.get_param::<String>("OPENAI_CLIENT_CERTIFICATE_PATH"),
            config.get_param::<String>("OPENAI_CLIENT_KEY_PATH"),
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
        if let Ok(ca_path) = config.get_param::<String>("OPENAI_CERTIFICATE_AUTHORITY_PATH") {
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
            base_path,
            api_key,
            organization,
            project,
            model,
            custom_headers,
        })
    }

    /// Helper function to add OpenAI-specific headers to a request
    fn add_headers(&self, mut request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        // Add organization header if present
        if let Some(org) = &self.organization {
            request = request.header("OpenAI-Organization", org);
        }

        // Add project header if present
        if let Some(project) = &self.project {
            request = request.header("OpenAI-Project", project);
        }

        // Add custom headers if present
        if let Some(custom_headers) = &self.custom_headers {
            for (key, value) in custom_headers {
                request = request.header(key, value);
            }
        }

        request
    }

    async fn post(&self, payload: Value) -> Result<Value, ProviderError> {
        let base_url = url::Url::parse(&self.host)
            .map_err(|e| ProviderError::RequestFailed(format!("Invalid base URL: {e}")))?;
        let url = base_url.join(&self.base_path).map_err(|e| {
            ProviderError::RequestFailed(format!("Failed to construct endpoint URL: {e}"))
        })?;

        let request = self
            .client
            .post(url)
            .header("Authorization", format!("Bearer {}", self.api_key));

        let request = self.add_headers(request);

        let response = request.json(&payload).send().await?;

        handle_response_openai_compat(response).await
    }
}

#[async_trait]
impl Provider for OpenAiProvider {
    fn metadata() -> ProviderMetadata {
        ProviderMetadata::new(
            "openai",
            "OpenAI",
            "GPT-4 and other OpenAI models, including OpenAI compatible ones",
            OPEN_AI_DEFAULT_MODEL,
            OPEN_AI_KNOWN_MODELS.to_vec(),
            OPEN_AI_DOC_URL,
            vec![
                ConfigKey::new("OPENAI_API_KEY", true, true, None),
                ConfigKey::new("OPENAI_HOST", true, false, Some("https://api.openai.com")),
                ConfigKey::new("OPENAI_BASE_PATH", true, false, Some("v1/chat/completions")),
                ConfigKey::new("OPENAI_ORGANIZATION", false, false, None),
                ConfigKey::new("OPENAI_PROJECT", false, false, None),
                ConfigKey::new("OPENAI_CUSTOM_HEADERS", false, true, None),
                ConfigKey::new("OPENAI_TIMEOUT", false, false, Some("600")),
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
        let payload = create_request(&self.model, system, messages, tools, &ImageFormat::OpenAi)?;

        // Make request
        let response = self.post(payload.clone()).await?;

        // Parse response
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
        emit_debug_trace(&self.model, &payload, &response, &usage);
        Ok((message, ProviderUsage::new(model, usage)))
    }

    /// Fetch supported models from OpenAI; returns Err on any failure, Ok(None) if no data
    async fn fetch_supported_models_async(&self) -> Result<Option<Vec<String>>, ProviderError> {
        // List available models via OpenAI API
        let base_url =
            url::Url::parse(&self.host).map_err(|e| ProviderError::RequestFailed(e.to_string()))?;
        let url = base_url
            .join("v1/models")
            .map_err(|e| ProviderError::RequestFailed(e.to_string()))?;
        let mut request = self.client.get(url).bearer_auth(&self.api_key);
        if let Some(org) = &self.organization {
            request = request.header("OpenAI-Organization", org);
        }
        if let Some(project) = &self.project {
            request = request.header("OpenAI-Project", project);
        }
        if let Some(headers) = &self.custom_headers {
            for (key, value) in headers {
                request = request.header(key, value);
            }
        }
        let response = request.send().await?;
        let json: serde_json::Value = response.json().await?;
        if let Some(err_obj) = json.get("error") {
            let msg = err_obj
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown error");
            return Err(ProviderError::Authentication(msg.to_string()));
        }
        let data = json.get("data").and_then(|v| v.as_array()).ok_or_else(|| {
            ProviderError::UsageError("Missing data field in JSON response".into())
        })?;
        let mut models: Vec<String> = data
            .iter()
            .filter_map(|m| m.get("id").and_then(|v| v.as_str()).map(str::to_string))
            .collect();
        models.sort();
        Ok(Some(models))
    }

    fn supports_embeddings(&self) -> bool {
        true
    }

    async fn create_embeddings(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, ProviderError> {
        EmbeddingCapable::create_embeddings(self, texts)
            .await
            .map_err(|e| ProviderError::ExecutionError(e.to_string()))
    }
}

fn parse_custom_headers(s: String) -> HashMap<String, String> {
    s.split(',')
        .filter_map(|header| {
            let mut parts = header.splitn(2, '=');
            let key = parts.next().map(|s| s.trim().to_string())?;
            let value = parts.next().map(|s| s.trim().to_string())?;
            Some((key, value))
        })
        .collect()
}

#[async_trait]
impl EmbeddingCapable for OpenAiProvider {
    async fn create_embeddings(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        // Get embedding model from env var or use default
        let embedding_model = std::env::var("GOOSE_EMBEDDING_MODEL")
            .unwrap_or_else(|_| "text-embedding-3-small".to_string());

        let request = EmbeddingRequest {
            input: texts,
            model: embedding_model,
        };

        // Construct embeddings endpoint URL
        let base_url =
            url::Url::parse(&self.host).map_err(|e| anyhow::anyhow!("Invalid base URL: {e}"))?;
        let url = base_url
            .join("v1/embeddings")
            .map_err(|e| anyhow::anyhow!("Failed to construct embeddings URL: {e}"))?;

        let req = self
            .client
            .post(url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&request);

        let req = self.add_headers(req);

        let response = req
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to send embedding request: {e}"))?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!("Embedding API error: {}", error_text));
        }

        let embedding_response: EmbeddingResponse = response
            .json()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to parse embedding response: {e}"))?;

        Ok(embedding_response
            .data
            .into_iter()
            .map(|d| d.embedding)
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::fs::File;
    use std::io::Write;
    use tempfile::tempdir;

    const DUMMY_CERT_PEM: &str = "-----BEGIN CERTIFICATE-----\nMIIDddCCArKgAwIBAgIJAJBc4gBiuUuSMA0GCSqGSIb3DQEBCwUAMGQxCzAJBgNV\nBAYTAlVTMQswCQYDVQQIDAJDQTEVMBMGA1UEBwwMTW91bnRhaW4gVmlldzESMBAG\nA1UECgwJR29vZ2xlIEluYzESMBAGA1UEAwwJZXhhbXBsZS5jb20wHhcNMjQwNzAx\nMDAwMDAwWhcNMjUwNzAxMDAwMDAwWjBkMQswCQYDVQQGEwJVUzELMAkGA1UECAwC\nQ0ExFTATBgNVBAcMDEN1cGVydGlubyBGQjESMBAGA1UECgwJR29vZ2xlIEluYzES\nMBAGA1UEAwwJZXhhbXBsZS5jb20wggEiMA0GCSqGSIb3DQEBAQUAA4IBDwAwggEK\nAoIBAQC0fPz7T6QlZtZ0b2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0\nf2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0\nf2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0\nf2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0\nCAwEAAaNTMFEwHQYDVR0OBAYEFECD4jFxD6vjY5tC+2o7q3B+A4gxMB8GA1UdIwQY\nMBaAFECD4jFxD6vjY5tC+2o7q3B+A4gxMA8GA1UdEwEB/wQFMAMBAf8wDQYJKoZI\nhvcNAQELBQADggEBAIZU8oJyX0g8rYtC7iWjH8rYtC7iWjH8rYtC7iWjH8rYtC7i\nWjH8rYtC7iWjH8rYtC7iWjH8rYtC7iWjH8rYtC7iWjH8rYtC7iWjH8rYtC7iWjH8\nrYtC7iWjH8rYtC7iWjH8rYtC7iWjH8rYtC7iWjH8rYtC7iWjH8rYtC7iWjH8rYtC\n7iWjH8rYtC7iWjH8rYtC7iWjH8rYtC7iWjH8rYtC7iWjH8rYtC7iWjH8rYtC7iWj\nH8rYtC7iWjH8rYtC7iWjH8rYtC7iWjH8rYtC7g=\n-----END CERTIFICATE-----";
    const DUMMY_KEY_PEM: &str = "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC0fPz7T6QlZtZ0\nb2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0\nf2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0\nf2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0\nf2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0CAwEAAQKBgQC0fPz7T6Ql\nZtZ0b2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0\nf2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0\nf2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0\nf2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0AoGBANf8N4gL9sDG\n7aZ1c2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0\nf2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0\nAoGBANs/9j1z+iYlZtZ0b2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0\nf2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0\nf2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0AoGAOUb/h3N+lYlZ\ntZ0b2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0\nf2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0\nAoGAcr7/8t8+kYlZtZ0b2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0\nf2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0\nf2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0AoGAJqL8z9X+kYlZ\ntZ0b2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0\nf2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0\n-----END PRIVATE KEY-----";
    const DUMMY_CA_PEM: &str = "-----BEGIN CERTIFICATE-----\nMIIDdzCCAl+gAwIBAgIJAKeVqL+P6Jd+MA0GCSqGSIb3DQEBCwUAMGQxCzAJBgNV\nBAYTAlVTMQswCQYDVQQIDAJDQTEVMBMGA1UEBwwMTW91bnRhaW4gVmlldzESMBAG\nA1UECgwJR29vZ2xlIEluYzESMBAGA1UEAwwJZXhhbXBsZS5jb20wHhcNMjQwNzAx\nMDAwMDAwWhcNMjUwNzAxMDAwMDAwWjBkMQswCQYDVQQGEwJVUzELMAkGA1UECAwC\nQ0ExFTATBgNVBAcMDEN1cGVydGlubyBGQjESMBAGA1UECgwJR29vZ2xlIEluYzES\nMBAGA1UEAwwJZXhhbXBsZS5jb20wggEiMA0GCSqGSIb3DQEBAQUAA4IBDwAwggEK\nAoIBAQC0fPz7T6QlZtZ0b2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0\nf2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0\nf2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0\nf2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0f2Z0\nCAwEAAaNQME4wHQYDVR0OBAYEFECD4jFxD6vjY5tC+2o7q3B+A4gxMB8GA1UdIwQY\nMBaAFECD4jFxD6vjY5tC+2o7q3B+A4gxMAwGA1UdEwEB/wQCMAAwDQYJKoZIhvcN\nAQELBQADggEBAIZU8oJyX0g8rYtC7iWjH8rYtC7iWjH8rYtC7iWjH8rYtC7iWjH8\nrYtC7iWjH8rYtC7iWjH8rYtC7iWjH8rYtC7iWjH8rYtC7iWjH8rYtC7iWjH8rYtC\n7iWjH8rYtC7iWjH8rYtC7iWjH8rYtC7iWjH8rYtC7iWjH8rYtC7iWjH8rYtC7iWj\nH8rYtC7iWjH8rYtC7iWjH8rYtC7iWjH8rYtC7iWjH8rYtC7iWjH8rYtC7iWjH8rYt\nC7iWjH8rYtC7iWjH8rYtC7iWjH8rYtC7iWjH8=\n-----END CERTIFICATE-----";


    struct TestEnv {
        _dir: tempfile::TempDir, // Keep TempDir in scope for automatic cleanup
        cert_path: Option<String>,
        key_path: Option<String>,
        ca_path: Option<String>,
    }

    fn setup_test_env(use_client_cert: bool, use_client_key: bool, use_ca_cert: bool) -> TestEnv {
        let dir = tempdir().unwrap();
        let mut test_env = TestEnv {
            _dir: dir,
            cert_path: None,
            key_path: None,
            ca_path: None,
        };

        env::set_var("OPENAI_API_KEY", "dummy_api_key");

        if use_client_cert {
            let cert_file_path = test_env._dir.path().join("client.crt");
            let mut cert_file = File::create(&cert_file_path).unwrap();
            cert_file.write_all(DUMMY_CERT_PEM.as_bytes()).unwrap();
            let path_str = cert_file_path.to_str().unwrap().to_string();
            env::set_var("OPENAI_CLIENT_CERTIFICATE_PATH", &path_str);
            test_env.cert_path = Some(path_str);
        } else {
            env::remove_var("OPENAI_CLIENT_CERTIFICATE_PATH");
        }

        if use_client_key {
            let key_file_path = test_env._dir.path().join("client.key");
            let mut key_file = File::create(&key_file_path).unwrap();
            key_file.write_all(DUMMY_KEY_PEM.as_bytes()).unwrap();
            let path_str = key_file_path.to_str().unwrap().to_string();
            env::set_var("OPENAI_CLIENT_KEY_PATH", &path_str);
            test_env.key_path = Some(path_str);
        } else {
            env::remove_var("OPENAI_CLIENT_KEY_PATH");
        }

        if use_ca_cert {
            let ca_file_path = test_env._dir.path().join("ca.crt");
            let mut ca_file = File::create(&ca_file_path).unwrap();
            ca_file.write_all(DUMMY_CA_PEM.as_bytes()).unwrap();
            let path_str = ca_file_path.to_str().unwrap().to_string();
            env::set_var("OPENAI_CERTIFICATE_AUTHORITY_PATH", &path_str);
            test_env.ca_path = Some(path_str);
        } else {
            env::remove_var("OPENAI_CERTIFICATE_AUTHORITY_PATH");
        }
        test_env
    }

    fn default_model_config() -> ModelConfig {
        ModelConfig::new("gpt-4o".to_string())
    }

    #[test]
    fn test_mtls_client_cert_and_key_valid_paths() {
        let _env = setup_test_env(true, true, false);
        let provider = OpenAiProvider::from_env(default_model_config());
        assert!(provider.is_ok());
    }

    #[test]
    fn test_mtls_ca_cert_valid_path() {
        let _env = setup_test_env(false, false, true);
        let provider = OpenAiProvider::from_env(default_model_config());
        assert!(provider.is_ok());
    }

    #[test]
    fn test_mtls_all_params_valid_paths() {
        let _env = setup_test_env(true, true, true);
        let provider = OpenAiProvider::from_env(default_model_config());
        assert!(provider.is_ok());
    }

    #[test]
    fn test_mtls_client_cert_invalid_path() {
        let _env = setup_test_env(false, true, false); // No cert file created
        env::set_var("OPENAI_CLIENT_CERTIFICATE_PATH", "/non/existent/cert.crt");
        // Key path is set by setup_test_env but cert is not, then explicitly set to invalid
        let provider = OpenAiProvider::from_env(default_model_config());
        assert!(provider.is_err());
        if let Err(e) = provider {
            assert!(e.to_string().contains("Failed to open certificate file"));
        }
    }

    #[test]
    fn test_mtls_client_key_invalid_path() {
        let _env = setup_test_env(true, false, false); // No key file created
        env::set_var("OPENAI_CLIENT_KEY_PATH", "/non/existent/key.key");
        let provider = OpenAiProvider::from_env(default_model_config());
        assert!(provider.is_err());
        if let Err(e) = provider {
            assert!(e.to_string().contains("Failed to open key file"));
        }
    }

    #[test]
    fn test_mtls_ca_cert_invalid_path() {
        let _env = setup_test_env(false, false, false); // No CA file created
        env::set_var("OPENAI_CERTIFICATE_AUTHORITY_PATH", "/non/existent/ca.crt");
        let provider = OpenAiProvider::from_env(default_model_config());
        assert!(provider.is_err());
        if let Err(e) = provider {
            assert!(e.to_string().contains("Failed to open CA file"));
        }
    }

    #[test]
    fn test_mtls_client_cert_empty_path_key_present() {
        // If cert path is empty but key path is present, it should not attempt to load identity.
        let _env = setup_test_env(false, true, false);
        env::set_var("OPENAI_CLIENT_CERTIFICATE_PATH", "");
        let provider = OpenAiProvider::from_env(default_model_config());
        assert!(provider.is_ok()); // Should be Ok, as empty path skips loading identity
    }

    #[test]
    fn test_mtls_client_key_empty_path_cert_present() {
        // If key path is empty but cert path is present, it should not attempt to load identity.
        let _env = setup_test_env(true, false, false);
        env::set_var("OPENAI_CLIENT_KEY_PATH", "");
        let provider = OpenAiProvider::from_env(default_model_config());
        assert!(provider.is_ok()); // Should be Ok, as empty path skips loading identity
    }

    #[test]
    fn test_mtls_client_cert_and_key_both_empty_paths() {
        let _env = setup_test_env(false, false, false);
        env::set_var("OPENAI_CLIENT_CERTIFICATE_PATH", "");
        env::set_var("OPENAI_CLIENT_KEY_PATH", "");
        let provider = OpenAiProvider::from_env(default_model_config());
        assert!(provider.is_ok());
    }

    #[test]
    fn test_mtls_ca_cert_empty_path() {
        let _env = setup_test_env(false, false, false);
        env::set_var("OPENAI_CERTIFICATE_AUTHORITY_PATH", "");
        let provider = OpenAiProvider::from_env(default_model_config());
        assert!(provider.is_ok()); // Should be Ok, as empty path skips loading CA
    }

    // Teardown: env vars are typically isolated between tests by the test runner.
    // If not, manual env::remove_var would be needed in each test or a custom test harness.
}
