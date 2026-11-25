use agentkit_http::HttpError;
use thiserror::Error;

/// Errors produced by the Gemini adapter.
#[derive(Debug, Error)]
pub enum GeminiError {
    /// A required environment variable is not set or could not be parsed.
    #[error("missing or invalid environment variable {0}")]
    MissingEnv(&'static str),

    /// Neither an API key nor an auth token was supplied.
    #[error(
        "no Gemini credentials configured: set GEMINI_API_KEY, GOOGLE_API_KEY, or GEMINI_AUTH_TOKEN"
    )]
    MissingCredentials,

    /// The configured model identifier is empty.
    #[error("Gemini model identifier must not be empty")]
    InvalidModel,

    /// Failure constructing the underlying HTTP client.
    #[error(transparent)]
    HttpClient(#[from] HttpError),
}
