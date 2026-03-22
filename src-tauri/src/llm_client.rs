use crate::settings::{PostProcessProvider, COHERE_PROVIDER_ID};
use log::debug;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE, REFERER, USER_AGENT};
use reqwest::Url;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use specta::Type;

#[derive(Debug, Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct JsonSchema {
    name: String,
    strict: bool,
    schema: Value,
}

#[derive(Debug, Serialize)]
struct ResponseFormat {
    #[serde(rename = "type")]
    format_type: String,
    json_schema: JsonSchema,
}

#[derive(Debug, Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<ResponseFormat>,
}

#[derive(Debug, Clone, Copy)]
pub struct CohereThinkingOptions {
    pub enabled: bool,
    pub token_budget: u32,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, Type)]
pub struct PostProcessModelCapabilities {
    pub supports_thinking: bool,
    pub supports_token_budget: bool,
}

#[derive(Debug, Serialize)]
struct CohereThinkingRequest {
    #[serde(rename = "type")]
    thinking_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    token_budget: Option<u32>,
}

#[derive(Debug, Serialize)]
struct CohereChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<CohereThinkingRequest>,
}

/// OpenAI-style response: choices[].message.content
#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatMessageResponse,
}

#[derive(Debug, Deserialize)]
struct ChatMessageResponse {
    content: Option<String>,
}

/// Cohere v2 Chat API response: message.content is array of { type, text }
#[derive(Debug, Deserialize)]
struct CohereChatResponse {
    message: CohereMessage,
}

#[derive(Debug, Deserialize)]
struct CohereMessage {
    content: CohereContent,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum CohereContent {
    Array(Vec<CohereContentBlock>),
    String(String),
}

#[derive(Debug, Deserialize)]
struct CohereContentBlock {
    #[serde(rename = "type")]
    block_type: Option<String>,
    text: Option<String>,
}

fn is_cohere_host(url: &Url) -> bool {
    matches!(
        url.host_str(),
        Some(host) if host.ends_with("cohere.com") || host.ends_with("cohere.ai")
    )
}

fn is_cohere_provider(provider: &PostProcessProvider) -> bool {
    provider.id == COHERE_PROVIDER_ID || crate::settings::is_cohere_v2_chat_url(&provider.base_url)
}

fn cohere_models_url(base_url: &str) -> Option<String> {
    let mut url = Url::parse(base_url).ok()?;
    if !is_cohere_host(&url) {
        return None;
    }

    url.set_path("/v1/models");
    {
        let mut query = url.query_pairs_mut();
        query.clear();
        query.append_pair("endpoint", "chat");
        query.append_pair("page_size", "1000");
    }
    url.set_fragment(None);
    Some(url.to_string())
}

fn cohere_model_capabilities(model: &str) -> PostProcessModelCapabilities {
    match model {
        "command-a-reasoning-08-2025" => PostProcessModelCapabilities {
            supports_thinking: true,
            supports_token_budget: true,
        },
        "command-a-03-2025" => PostProcessModelCapabilities::default(),
        _ => PostProcessModelCapabilities::default(),
    }
}

pub fn get_post_process_model_capabilities(
    provider: &PostProcessProvider,
    model: &str,
) -> PostProcessModelCapabilities {
    if is_cohere_provider(provider) {
        cohere_model_capabilities(model)
    } else {
        PostProcessModelCapabilities::default()
    }
}

/// Build headers for API requests based on provider type
fn build_headers(provider: &PostProcessProvider, api_key: &str) -> Result<HeaderMap, String> {
    let mut headers = HeaderMap::new();

    // Common headers
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    headers.insert(
        REFERER,
        HeaderValue::from_static("https://github.com/cjpais/Handy"),
    );
    headers.insert(
        USER_AGENT,
        HeaderValue::from_static("Handy/1.0 (+https://github.com/cjpais/Handy)"),
    );
    headers.insert("X-Title", HeaderValue::from_static("Handy"));

    // Provider-specific auth headers
    if !api_key.is_empty() {
        if provider.id == "anthropic" {
            headers.insert(
                "x-api-key",
                HeaderValue::from_str(api_key)
                    .map_err(|e| format!("Invalid API key header value: {}", e))?,
            );
            headers.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));
        } else {
            headers.insert(
                AUTHORIZATION,
                HeaderValue::from_str(&format!("Bearer {}", api_key))
                    .map_err(|e| format!("Invalid authorization header value: {}", e))?,
            );
        }
    }

    Ok(headers)
}

/// Create an HTTP client with provider-specific headers
fn create_client(provider: &PostProcessProvider, api_key: &str) -> Result<reqwest::Client, String> {
    let headers = build_headers(provider, api_key)?;
    reqwest::Client::builder()
        .default_headers(headers)
        .build()
        .map_err(|e| format!("Failed to build HTTP client: {}", e))
}

/// Send a chat completion request to an OpenAI-compatible API
/// Returns Ok(Some(content)) on success, Ok(None) if response has no content,
/// or Err on actual errors (HTTP, parsing, etc.)
pub async fn send_chat_completion(
    provider: &PostProcessProvider,
    api_key: String,
    model: &str,
    prompt: String,
    cohere_thinking: Option<CohereThinkingOptions>,
) -> Result<Option<String>, String> {
    send_chat_completion_with_schema(
        provider,
        api_key,
        model,
        prompt,
        None,
        None,
        cohere_thinking,
    )
    .await
}

/// Send a chat completion request with structured output support
/// When json_schema is provided, uses structured outputs mode
/// system_prompt is used as the system message when provided
pub async fn send_chat_completion_with_schema(
    provider: &PostProcessProvider,
    api_key: String,
    model: &str,
    user_content: String,
    system_prompt: Option<String>,
    json_schema: Option<Value>,
    cohere_thinking: Option<CohereThinkingOptions>,
) -> Result<Option<String>, String> {
    let base_url = provider.base_url.trim_end_matches('/');
    let is_cohere = is_cohere_provider(provider);
    let url = if is_cohere {
        base_url.to_string()
    } else {
        format!("{}/chat/completions", base_url)
    };

    debug!("Sending chat completion request to: {}", url);

    let client = create_client(provider, &api_key)?;

    // Build messages vector
    let mut messages = Vec::new();

    // Add system prompt if provided
    if let Some(system) = system_prompt {
        messages.push(ChatMessage {
            role: "system".to_string(),
            content: system,
        });
    }

    // Add user message
    messages.push(ChatMessage {
        role: "user".to_string(),
        content: user_content,
    });

    let has_json_schema = json_schema.is_some();

    // Build response_format if schema is provided
    let response_format = json_schema.map(|schema| ResponseFormat {
        format_type: "json_schema".to_string(),
        json_schema: JsonSchema {
            name: "transcription_output".to_string(),
            strict: true,
            schema,
        },
    });

    let request_body = ChatCompletionRequest {
        model: model.to_string(),
        messages,
        response_format,
    };
    let response = if is_cohere {
        let capabilities = get_post_process_model_capabilities(provider, model);
        if has_json_schema {
            debug!("Ignoring OpenAI structured output schema for Cohere request");
        }

        let request_body = CohereChatRequest {
            model: model.to_string(),
            messages: request_body.messages,
            thinking: if capabilities.supports_thinking {
                cohere_thinking.map(|options| CohereThinkingRequest {
                    thinking_type: if options.enabled {
                        "enabled".to_string()
                    } else {
                        "disabled".to_string()
                    },
                    token_budget: if options.enabled
                        && capabilities.supports_token_budget
                        && options.token_budget > 0
                    {
                        Some(options.token_budget)
                    } else {
                        None
                    },
                })
            } else {
                None
            },
        };

        client
            .post(&url)
            .json(&request_body)
            .send()
            .await
            .map_err(|e| format!("HTTP request failed: {}", e))?
    } else {
        client
            .post(&url)
            .json(&request_body)
            .send()
            .await
            .map_err(|e| format!("HTTP request failed: {}", e))?
    };

    let status = response.status();
    let response_text = response
        .text()
        .await
        .unwrap_or_else(|_| "Failed to read response".to_string());
    if !status.is_success() {
        return Err(format!(
            "API request failed with status {}: {}",
            status, response_text
        ));
    }

    if is_cohere {
        let completion: CohereChatResponse = serde_json::from_str(&response_text)
            .map_err(|e| format!("Failed to parse Cohere API response: {}", e))?;
        let text = match completion.message.content {
            CohereContent::String(s) => Some(s),
            CohereContent::Array(blocks) => {
                let parts: Vec<String> = blocks.into_iter().filter_map(|b| b.text).collect();
                if parts.is_empty() {
                    None
                } else {
                    Some(parts.join("\n"))
                }
            }
        };
        Ok(text)
    } else {
        let completion: ChatCompletionResponse = serde_json::from_str(&response_text)
            .map_err(|e| format!("Failed to parse API response: {}", e))?;
        Ok(completion
            .choices
            .first()
            .and_then(|choice| choice.message.content.clone()))
    }
}

/// Fetch available models from an OpenAI-compatible API
/// Returns a list of model IDs
pub async fn fetch_models(
    provider: &PostProcessProvider,
    api_key: String,
) -> Result<Vec<String>, String> {
    let base_url = provider.base_url.trim_end_matches('/');
    let url = if is_cohere_provider(provider) {
        cohere_models_url(base_url).unwrap_or_else(|| format!("{}/models", base_url))
    } else {
        format!("{}/models", base_url)
    };

    debug!("Fetching models from: {}", url);

    let client = create_client(provider, &api_key)?;

    let response = client
        .get(&url)
        .send()
        .await
        .map_err(|e| format!("Failed to fetch models: {}", e))?;

    let status = response.status();
    if !status.is_success() {
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        return Err(format!(
            "Model list request failed ({}): {}",
            status, error_text
        ));
    }

    let parsed: serde_json::Value = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse response: {}", e))?;

    let mut models = Vec::new();

    // Handle OpenAI format: { data: [ { id: "..." }, ... ] }
    if let Some(data) = parsed.get("data").and_then(|d| d.as_array()) {
        for entry in data {
            if let Some(id) = entry.get("id").and_then(|i| i.as_str()) {
                models.push(id.to_string());
            } else if let Some(name) = entry.get("name").and_then(|n| n.as_str()) {
                models.push(name.to_string());
            }
        }
    }
    // Handle Cohere format: { models: [ { name: "..." }, ... ] }
    else if let Some(data) = parsed.get("models").and_then(|d| d.as_array()) {
        for entry in data {
            if let Some(name) = entry.get("name").and_then(|n| n.as_str()) {
                models.push(name.to_string());
            } else if let Some(id) = entry.get("id").and_then(|i| i.as_str()) {
                models.push(id.to_string());
            }
        }
    }
    // Handle array format: [ "model1", "model2", ... ]
    else if let Some(array) = parsed.as_array() {
        for entry in array {
            if let Some(model) = entry.as_str() {
                models.push(model.to_string());
            }
        }
    }

    Ok(models)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cohere_provider() -> PostProcessProvider {
        PostProcessProvider {
            id: COHERE_PROVIDER_ID.to_string(),
            label: "Cohere".to_string(),
            base_url: "https://api.cohere.com/v2/chat".to_string(),
            allow_base_url_edit: false,
            models_endpoint: Some("/v1/models".to_string()),
            supports_structured_output: false,
        }
    }

    #[test]
    fn returns_safe_default_capabilities_for_unknown_cohere_models() {
        let provider = cohere_provider();

        let non_reasoning = get_post_process_model_capabilities(&provider, "command-a-03-2025");
        assert!(!non_reasoning.supports_thinking);
        assert!(!non_reasoning.supports_token_budget);

        let reasoning =
            get_post_process_model_capabilities(&provider, "command-a-reasoning-08-2025");
        assert!(reasoning.supports_thinking);
        assert!(reasoning.supports_token_budget);

        let unknown = get_post_process_model_capabilities(&provider, "future-cohere-model");
        assert!(!unknown.supports_thinking);
        assert!(!unknown.supports_token_budget);
    }

    #[test]
    fn builds_cohere_chat_model_list_url() {
        let url = cohere_models_url("https://api.cohere.com/v2/chat").unwrap();
        assert_eq!(
            url,
            "https://api.cohere.com/v1/models?endpoint=chat&page_size=1000"
        );
    }
}
