//! HTTP transport trait and request/response types.
//!
//! Implementors satisfy [`HttpClient::execute`]; [`HttpRequestBuilder`] and
//! [`HttpResponse`] carry the ergonomics (body encoding, header helpers,
//! streaming). Enable the `reqwest-client` feature (default) for an
//! `impl HttpClient for reqwest::Client`; disable it to compile trait-only.

mod authentication;
mod client;
mod error;
mod request;
mod resilience;
mod response;

#[cfg(feature = "reqwest-client")]
mod reqwest_impl;

#[cfg(feature = "reqwest-middleware-client")]
mod reqwest_middleware_impl;

pub use authentication::{Authentication, AuthenticationAttempt, AuthenticationProvider};
pub use client::{Http, HttpClient};
pub use error::{BoxError, HttpError};
pub use request::{HttpRequest, HttpRequestBuilder};
pub use resilience::{
    LogicalDeadline, ResilienceConfig, TruncatedStreamDetector, is_retryable_body_read,
    is_retryable_status, next_body_chunk, next_body_chunk_bounded, retry_hint, run_bounded, sleep,
};
pub use response::{BodyStream, HttpResponse};

pub use bytes::Bytes;
pub use http::{HeaderMap, HeaderName, HeaderValue, Method, StatusCode, Uri, header};

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use bytes::Bytes;
    use futures_util::stream;
    use std::borrow::Cow;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct StubClient {
        calls: AtomicUsize,
        status: StatusCode,
        body: Bytes,
        expected_body: Option<Bytes>,
    }

    #[async_trait]
    impl HttpClient for StubClient {
        async fn execute(&self, request: HttpRequest) -> Result<HttpResponse, HttpError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if let Some(expected) = &self.expected_body {
                assert_eq!(request.body.as_deref(), Some(expected.as_ref()));
            }
            let body = self.body.clone();
            let stream = stream::once(async move { Ok::<_, HttpError>(body) });
            Ok(HttpResponse::new(
                self.status,
                request.headers.clone(),
                request.url.clone(),
                Box::pin(stream),
            ))
        }
    }

    #[tokio::test]
    async fn builder_sends_json_body_and_decodes_response() {
        #[derive(serde::Serialize)]
        struct Req {
            name: &'static str,
        }
        #[derive(serde::Deserialize, Debug, PartialEq)]
        struct Resp {
            ok: bool,
        }

        let stub = StubClient {
            calls: AtomicUsize::new(0),
            status: StatusCode::OK,
            body: Bytes::from_static(br#"{"ok":true}"#),
            expected_body: Some(Bytes::from_static(br#"{"name":"agentkit"}"#)),
        };
        let http = Http::from_arc(Arc::new(stub));

        let resp = http
            .post("https://example.test/echo")
            .bearer_auth("tok")
            .json(&Req { name: "agentkit" })
            .send()
            .await
            .expect("send");

        assert_eq!(resp.status(), StatusCode::OK);
        let auth = resp.headers().get(http::header::AUTHORIZATION).unwrap();
        assert_eq!(auth, "Bearer tok");
        let ct = resp.headers().get(http::header::CONTENT_TYPE).unwrap();
        assert_eq!(ct, "application/json");

        let decoded: Resp = resp.json().await.expect("json");
        assert_eq!(decoded, Resp { ok: true });
    }

    #[tokio::test]
    async fn type_erased_authentication_preserves_opaque_prior_state_and_redacts() {
        struct Refreshing(AtomicUsize);

        #[async_trait]
        impl AuthenticationProvider for Refreshing {
            async fn authenticate(
                &self,
                previous: Option<&AuthenticationAttempt>,
            ) -> Result<AuthenticationAttempt, HttpError> {
                let generation = self.0.fetch_add(1, Ordering::SeqCst);
                assert_eq!(
                    previous
                        .and_then(|attempt| attempt.state::<usize>())
                        .copied(),
                    generation.checked_sub(1)
                );
                let mut headers = HeaderMap::new();
                headers.insert(
                    header::AUTHORIZATION,
                    HeaderValue::from_str(&format!("Bearer secret-{generation}")).unwrap(),
                );
                Ok(AuthenticationAttempt::new(headers, generation)
                    .with_binding(format!("credential-generation-{generation}")))
            }
        }

        let authentication = Authentication::new(Refreshing(AtomicUsize::new(0)));
        let first = authentication.authenticate(None).await.unwrap();
        let second = authentication.authenticate(Some(&first)).await.unwrap();
        assert_eq!(second.state::<usize>(), Some(&1));
        assert_eq!(first.binding(), Some("credential-generation-0"));
        assert_eq!(second.binding(), Some("credential-generation-1"));
        assert_eq!(second.clone().binding(), second.binding());
        let debug = format!("{authentication:?} {second:?}");
        assert!(!debug.contains("secret"));
        assert!(!debug.contains("credential-generation-1"));
        assert!(debug.contains("binding_present: true"));
        assert!(second.headers()[header::AUTHORIZATION].is_sensitive());

        let borrowed_source = String::from("borrowed-secret");
        let borrowed: Authentication = borrowed_source.as_str().into();
        drop(borrowed_source);

        let borrowed_string_source = String::from("borrowed-string-secret");
        let borrowed_string: Authentication = (&borrowed_string_source).into();
        drop(borrowed_string_source);

        let cow_borrowed_source = String::from("cow-borrowed-secret");
        let cow_borrowed: Authentication = Cow::Borrowed(cow_borrowed_source.as_str()).into();
        drop(cow_borrowed_source);

        let bearers = [
            (String::from("string-secret").into(), "string-secret"),
            (
                String::from("box-secret").into_boxed_str().into(),
                "box-secret",
            ),
            (Arc::<str>::from("arc-secret").into(), "arc-secret"),
            (
                Cow::<'static, str>::Owned(String::from("cow-owned-secret")).into(),
                "cow-owned-secret",
            ),
            (borrowed, "borrowed-secret"),
            (borrowed_string, "borrowed-string-secret"),
            (cow_borrowed, "cow-borrowed-secret"),
            ("static-secret".into(), "static-secret"),
        ];
        for (bearer, secret) in bearers {
            let attempt = bearer.authenticate(None).await.unwrap();
            assert_eq!(
                attempt.headers()[header::AUTHORIZATION].to_str().unwrap(),
                format!("Bearer {secret}")
            );
            assert!(attempt.headers()[header::AUTHORIZATION].is_sensitive());
            assert!(attempt.binding().is_some());
            assert_eq!(attempt.clone().binding(), attempt.binding());
            assert!(!attempt.binding().unwrap().contains(secret));
            let debug = format!("{bearer:?} {attempt:?}");
            assert!(!debug.contains(secret));
            assert!(debug.contains("binding_present: true"));
        }

        let header = Authentication::header(
            HeaderName::from_static("x-api-key"),
            String::from("owned-header-secret"),
        );
        let header_attempt = header.authenticate(None).await.unwrap();
        assert!(header_attempt.binding().is_some());
        assert!(
            !header_attempt
                .binding()
                .unwrap()
                .contains("owned-header-secret")
        );
        assert!(header_attempt.headers()["x-api-key"].is_sensitive());
        assert!(!format!("{header_attempt:?}").contains("owned-header-secret"));
    }

    #[test]
    fn resilience_classifies_hints_and_truncation() {
        assert!(is_retryable_status(StatusCode::TOO_MANY_REQUESTS));
        assert!(is_retryable_status(StatusCode::BAD_GATEWAY));
        assert!(!is_retryable_status(StatusCode::BAD_REQUEST));
        let body_error = HttpError::body(std::io::Error::other("disconnected"));
        assert!(is_retryable_body_read(StatusCode::OK, &body_error));
        assert!(!is_retryable_body_read(
            StatusCode::BAD_REQUEST,
            &body_error
        ));
        assert!(!is_retryable_body_read(
            StatusCode::OK,
            &HttpError::Other("protocol error".into())
        ));

        let mut headers = HeaderMap::new();
        headers.insert("retry-after", HeaderValue::from_static("3"));
        assert_eq!(
            retry_hint(&headers),
            Some(std::time::Duration::from_secs(3))
        );
        headers.insert(
            "retry-after",
            HeaderValue::from_static("Wed, 21 Oct 2099 07:28:00 GMT"),
        );
        assert!(retry_hint(&headers).is_some_and(|delay| !delay.is_zero()));
        headers.insert(
            "retry-after",
            HeaderValue::from_static("Tue, 31 Feb 2099 07:28:00 GMT"),
        );
        assert_eq!(retry_hint(&headers), None);

        headers.clear();
        headers.insert(header::CONTENT_LENGTH, HeaderValue::from_static("5"));
        let mut detector = TruncatedStreamDetector::from_headers(&headers);
        detector.observe(&Bytes::from_static(b"123"));
        assert!(matches!(
            detector.finish(),
            Err(HttpError::TruncatedBody {
                expected: 5,
                received: 3
            })
        ));
    }

    #[test]
    fn retry_hints_handle_non_finite_and_overflow_values() {
        let hint = |name: &'static str, value: &'static str| {
            let mut headers = HeaderMap::new();
            headers.insert(name, HeaderValue::from_static(value));
            retry_hint(&headers)
        };

        assert_eq!(hint("retry-after", "NaN"), None);
        assert_eq!(hint("retry-after", "inf"), None);
        assert_eq!(hint("retry-after", "1e999"), None);
        assert_eq!(hint("retry-after", "-inf"), None);
        assert_eq!(hint("retry-after", "-1"), None);
        assert_eq!(hint("x-ratelimit-reset", "NaN"), None);
        assert_eq!(hint("x-ratelimit-reset", "inf"), None);
        assert_eq!(
            hint("x-ratelimit-reset-requests-day", "42s"),
            Some(std::time::Duration::from_secs(42))
        );
        assert_eq!(
            hint("x-ratelimit-reset-tokens-minute", "1.5s"),
            Some(std::time::Duration::from_millis(1500))
        );

        let config = ResilienceConfig {
            initial_backoff: std::time::Duration::from_millis(5),
            max_backoff: std::time::Duration::from_millis(5),
            ..ResilienceConfig::default()
        };
        let mut headers = HeaderMap::new();
        headers.insert("retry-after", HeaderValue::from_static("inf"));
        assert!(config.retry_delay(0, Some(&headers)) <= std::time::Duration::from_millis(5));
        assert!(format!("{config:?}").contains("retry_budget"));
    }

    #[tokio::test]
    async fn error_for_status_flags_4xx() {
        let stub = StubClient {
            calls: AtomicUsize::new(0),
            status: StatusCode::BAD_REQUEST,
            body: Bytes::from_static(b"nope"),
            expected_body: None,
        };
        let http = Http::from_arc(Arc::new(stub));

        let resp = http.get("https://example.test").send().await.unwrap();
        assert!(resp.error_for_status().is_err());
    }
}
