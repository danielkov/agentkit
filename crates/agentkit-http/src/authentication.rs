use std::any::Any;
use std::borrow::Cow;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use http::{HeaderMap, HeaderName, HeaderValue, header};
use zeroize::Zeroizing;

use crate::HttpError;

/// The headers and provider-private state produced by one authentication attempt.
///
/// The state is deliberately opaque to callers. An authentication provider can
/// recover it with [`AuthenticationAttempt::state`] when a 401 response asks it
/// to refresh credentials. Providers can additionally attach a stable, non-secret
/// credential identity or generation with [`AuthenticationAttempt::with_binding`].
#[derive(Clone)]
pub struct AuthenticationAttempt {
    headers: HeaderMap,
    state: Arc<dyn Any + Send + Sync>,
    binding: Option<Arc<str>>,
}

impl AuthenticationAttempt {
    /// Creates an attempt with provider-private state.
    pub fn new<T>(mut headers: HeaderMap, state: T) -> Self
    where
        T: Any + Send + Sync,
    {
        mark_sensitive(&mut headers);
        Self {
            headers,
            state: Arc::new(state),
            binding: None,
        }
    }

    /// Creates an attempt which does not need provider-private state.
    pub fn stateless(headers: HeaderMap) -> Self {
        Self::new(headers, ())
    }

    /// Attaches a stable, non-secret credential identity or generation.
    ///
    /// The binding is intended for replay/continuation checks. It must not be a
    /// raw bearer token, API key, header value, or derivative of secret material.
    pub fn with_binding(mut self, binding: impl Into<Arc<str>>) -> Self {
        self.binding = Some(binding.into());
        self
    }

    /// Returns the stable, non-secret credential binding, when supplied.
    pub fn binding(&self) -> Option<&str> {
        self.binding.as_deref()
    }

    /// Returns the authentication headers. Header values are marked sensitive.
    pub fn headers(&self) -> &HeaderMap {
        &self.headers
    }

    /// Downcasts the opaque state saved by the provider on the prior attempt.
    pub fn state<T: Any>(&self) -> Option<&T> {
        self.state.downcast_ref()
    }
}

impl fmt::Debug for AuthenticationAttempt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AuthenticationAttempt")
            .field("header_names", &self.headers.keys().collect::<Vec<_>>())
            .field("state", &"<opaque>")
            .field("binding_present", &self.binding.is_some())
            .finish()
    }
}

/// Asynchronously supplies authentication headers and optional refresh state.
#[async_trait]
pub trait AuthenticationProvider: Send + Sync + 'static {
    /// Authenticates a request. `previous` is `None` initially and contains the
    /// exact opaque prior attempt when called reactively after a 401 response.
    async fn authenticate(
        &self,
        previous: Option<&AuthenticationAttempt>,
    ) -> Result<AuthenticationAttempt, HttpError>;
}

/// Clone-cheap, type-erased authentication provider handle.
///
/// Converting a bare `String`, `Box<str>`, `Arc<str>`, `Cow<str>`, or `&str`
/// into `Authentication` creates bearer authentication. Borrowed strings are
/// copied into owned storage whose bytes are zeroized on drop.
#[derive(Clone)]
pub struct Authentication {
    inner: Arc<dyn AuthenticationProvider>,
}

impl Authentication {
    pub fn new<P: AuthenticationProvider>(provider: P) -> Self {
        Self {
            inner: Arc::new(provider),
        }
    }

    pub fn from_arc(provider: Arc<dyn AuthenticationProvider>) -> Self {
        Self { inner: provider }
    }

    pub async fn authenticate(
        &self,
        previous: Option<&AuthenticationAttempt>,
    ) -> Result<AuthenticationAttempt, HttpError> {
        self.inner.authenticate(previous).await
    }

    /// Creates bearer authentication from an owned token. The retained token
    /// bytes are overwritten when the last handle is dropped.
    pub fn bearer(token: impl Into<String>) -> Self {
        Self::new(StaticAuthentication::bearer(token.into().into_bytes()))
    }

    /// Creates bearer authentication from a static token.
    ///
    /// This compatibility helper copies the token into owned storage whose bytes
    /// are overwritten when the last handle is dropped.
    pub fn bearer_static(token: &'static str) -> Self {
        Self::bearer(token.to_owned())
    }

    /// Creates authentication using one arbitrary secret header.
    pub fn header(name: HeaderName, value: impl Into<String>) -> Self {
        Self::new(StaticAuthentication::header(
            name,
            value.into().into_bytes(),
        ))
    }

    /// Creates authentication using one arbitrary static secret header.
    ///
    /// This compatibility helper copies the value into owned zeroizing storage.
    pub fn header_static(name: HeaderName, value: &'static str) -> Self {
        Self::header(name, value.to_owned())
    }
}

impl fmt::Debug for Authentication {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Authentication").finish_non_exhaustive()
    }
}

impl From<String> for Authentication {
    fn from(token: String) -> Self {
        Self::bearer(token)
    }
}

impl From<Box<str>> for Authentication {
    fn from(token: Box<str>) -> Self {
        Self::bearer(token.into_string())
    }
}

impl From<Arc<str>> for Authentication {
    fn from(token: Arc<str>) -> Self {
        Self::bearer(token.as_ref().to_owned())
    }
}

impl<'a> From<Cow<'a, str>> for Authentication {
    fn from(token: Cow<'a, str>) -> Self {
        match token {
            Cow::Owned(token) => Self::bearer(token),
            Cow::Borrowed(token) => Self::bearer(token.to_owned()),
        }
    }
}

impl From<&str> for Authentication {
    fn from(token: &str) -> Self {
        Self::bearer(token.to_owned())
    }
}

impl From<&String> for Authentication {
    fn from(token: &String) -> Self {
        Self::bearer(token.to_owned())
    }
}

impl From<(HeaderName, HeaderValue)> for Authentication {
    fn from((name, value): (HeaderName, HeaderValue)) -> Self {
        Self::new(FixedHeaders::one(name, value))
    }
}

impl From<HeaderMap> for Authentication {
    fn from(headers: HeaderMap) -> Self {
        Self::new(FixedHeaders(headers))
    }
}

struct StaticAuthentication {
    name: HeaderName,
    value: Zeroizing<Vec<u8>>,
    bearer: bool,
    binding: Arc<str>,
}

fn static_authentication_binding() -> Arc<str> {
    static NEXT_ID: AtomicU64 = AtomicU64::new(1);
    let created_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!(
        "static-auth-v1-{}-{created_at:x}-{:x}",
        std::process::id(),
        NEXT_ID.fetch_add(1, Ordering::Relaxed)
    )
    .into()
}

impl StaticAuthentication {
    fn bearer(value: Vec<u8>) -> Self {
        Self {
            name: header::AUTHORIZATION,
            value: Zeroizing::new(value),
            bearer: true,
            binding: static_authentication_binding(),
        }
    }
    fn header(name: HeaderName, value: Vec<u8>) -> Self {
        Self {
            name,
            value: Zeroizing::new(value),
            bearer: false,
            binding: static_authentication_binding(),
        }
    }
    fn value(&self) -> Result<HeaderValue, HttpError> {
        let bytes = self.value.as_slice();
        let rendered = if self.bearer {
            let mut rendered = Zeroizing::new(Vec::with_capacity(7 + bytes.len()));
            rendered.extend_from_slice(b"Bearer ");
            rendered.extend_from_slice(bytes);
            rendered
        } else {
            Zeroizing::new(bytes.to_vec())
        };
        let result = HeaderValue::from_bytes(&rendered)
            .map_err(|error| HttpError::InvalidHeader(format!("authentication value: {error}")));
        let mut value = result?;
        value.set_sensitive(true);
        Ok(value)
    }
}

#[async_trait]
impl AuthenticationProvider for StaticAuthentication {
    async fn authenticate(
        &self,
        _previous: Option<&AuthenticationAttempt>,
    ) -> Result<AuthenticationAttempt, HttpError> {
        let mut headers = HeaderMap::new();
        headers.insert(self.name.clone(), self.value()?);
        Ok(AuthenticationAttempt::stateless(headers).with_binding(self.binding.clone()))
    }
}

struct FixedHeaders(HeaderMap);
impl FixedHeaders {
    fn one(name: HeaderName, mut value: HeaderValue) -> Self {
        value.set_sensitive(true);
        let mut headers = HeaderMap::new();
        headers.insert(name, value);
        Self(headers)
    }
}

#[async_trait]
impl AuthenticationProvider for FixedHeaders {
    async fn authenticate(
        &self,
        _previous: Option<&AuthenticationAttempt>,
    ) -> Result<AuthenticationAttempt, HttpError> {
        Ok(AuthenticationAttempt::stateless(self.0.clone()))
    }
}

fn mark_sensitive(headers: &mut HeaderMap) {
    for value in headers.values_mut() {
        value.set_sensitive(true);
    }
}
