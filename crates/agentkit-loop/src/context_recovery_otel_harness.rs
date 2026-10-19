// Appended to TRUE_OTEL_HARNESS, using its real tracing-opentelemetry layer
// and capturing OpenTelemetry Tracer. No SDK dependency or production hook.
struct RecoveryOtelAdapter {
    requests: Arc<Mutex<Vec<TurnRequest>>>,
    reject_first: bool,
    early_stream: bool,
}
struct RecoveryOtelSession {
    requests: Arc<Mutex<Vec<TurnRequest>>>,
    reject_first: bool,
    early_stream: bool,
}
struct RecoveryOtelTurn {
    reject: bool,
}
struct RecoveryOtelMutator;

fn context_rejection() -> LoopError {
    LoopError::ProviderFailure(Box::new(agentkit_loop::ProviderFailure {
        route: agentkit_loop::ProviderRoute::Unknown,
        reason: agentkit_loop::ProviderFailureReason::InvalidRequest,
        last_attempt_reason: None,
        upstream: agentkit_loop::ProviderClassification {
            code: agentkit_loop::UpstreamErrorKind::ContextLengthExceeded,
            ..Default::default()
        },
        accounting: Default::default(),
    }))
}
#[async_trait]
impl ModelAdapter for RecoveryOtelAdapter {
    type Session = RecoveryOtelSession;
    async fn start_session(&self, _: SessionConfig) -> Result<Self::Session, LoopError> {
        Ok(RecoveryOtelSession {
            requests: self.requests.clone(),
            reject_first: self.reject_first,
            early_stream: self.early_stream,
        })
    }
}
#[async_trait]
impl ModelSession for RecoveryOtelSession {
    type Turn = RecoveryOtelTurn;
    fn model_name(&self) -> Option<&str> {
        Some("recovery-model")
    }
    fn provider_name(&self) -> Option<&str> {
        Some("recovery-provider")
    }
    async fn begin_turn(
        &mut self,
        request: TurnRequest,
        _: Option<TurnCancellation>,
    ) -> Result<Self::Turn, LoopError> {
        let reject = {
            let mut requests = self.requests.lock().unwrap();
            requests.push(request);
            self.reject_first && requests.len() == 1
        };
        if reject && !self.early_stream {
            return Err(context_rejection());
        }
        Ok(RecoveryOtelTurn { reject })
    }
}
#[async_trait]
impl ModelTurn for RecoveryOtelTurn {
    async fn next_event(
        &mut self,
        _: Option<TurnCancellation>,
    ) -> Result<Option<ModelTurnEvent>, LoopError> {
        if self.reject {
            return Err(context_rejection());
        }
        Ok(Some(ModelTurnEvent::Finished(ModelTurnResult {
            finish_reason: FinishReason::Completed,
            output_items: vec![Item::text(ItemKind::Assistant, "replacement result")],
            usage: Some(Usage {
                tokens: Some(TokenUsage::new(7, 3)),
                cost: None,
                metadata: MetadataMap::new(),
            }),
            metadata: MetadataMap::new(),
            model: Some("response-model".into()),
            response_id: Some("response-1".into()),
        })))
    }
}
#[async_trait]
impl agentkit_loop::LoopMutator for RecoveryOtelMutator {
    async fn mutate(
        &self,
        cursor: &mut agentkit_loop::TranscriptCursor<'_>,
        ctx: agentkit_loop::LoopCtx<'_>,
    ) -> Result<(), LoopError> {
        if ctx.point == agentkit_loop::MutationPoint::ContextLengthRejected {
            cursor.insert(0, Item::text(ItemKind::System, "recovery summary"));
        }
        Ok(())
    }
}
fn recovery_attempts_have_distinct_otel_spans() {
    for (reject_first, early_stream) in [(true, false), (true, true), (false, false)] {
        let tracer = MemoryTracer::default();
        let subscriber = tracing_subscriber::registry()
            .with(tracing_opentelemetry::layer().with_tracer(tracer.clone()));
        let requests = Arc::new(Mutex::new(Vec::new()));
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        let result = tracing::subscriber::with_default(subscriber, || {
            runtime.block_on(async {
                let agent = Agent::builder()
                    .model(RecoveryOtelAdapter {
                        requests: requests.clone(),
                        reject_first,
                        early_stream,
                    })
                    .input(vec![Item::text(ItemKind::User, "original request")])
                    .context_recovery(agentkit_loop::ContextRecoveryPolicy {
                        max_attempts: 1,
                        budget: std::time::Duration::from_secs(5),
                    })
                    .mutator(RecoveryOtelMutator)
                    .telemetry(
                        TelemetryConfig::default()
                            .with_input_messages(MessageCapture::new(10, 10000).unwrap()),
                    )
                    .build()
                    .unwrap();
                let mut driver = agent
                    .start(
                        SessionConfig::new("recovery-otel")
                            .with_cache(agentkit_loop::PromptCacheRequest::automatic()),
                    )
                    .await
                    .unwrap();
                driver.next().await.unwrap()
            })
        });
        let LoopStep::Finished(result) = result else {
            panic!("expected finished turn")
        };
        assert_eq!(result.usage.unwrap().tokens.unwrap().input_tokens, 7);
        let exported = tracer.exported.lock().unwrap();
        let chats: Vec<_> = exported
            .iter()
            .filter(|s| operation(s) == Some("chat"))
            .collect();
        let requests = requests.lock().unwrap();
        assert_eq!(chats.len(), if reject_first { 2 } else { 1 });
        assert_eq!(chats.len(), requests.len());
        for (index, chat) in chats.iter().enumerate() {
            // Count every occurrence, not a last-value lookup that hides duplicates.
            assert_eq!(
                chat.attributes
                    .iter()
                    .filter(|a| a.key.as_str() == "gen_ai.input.messages")
                    .count(),
                1
            );
            let input = json_array(chat, "gen_ai.input.messages");
            assert_eq!(input.len(), requests[index].transcript.len());
            assert!(
                input
                    .last()
                    .unwrap()
                    .to_string()
                    .contains("original request")
            );
            assert_eq!(
                input
                    .iter()
                    .any(|item| item.to_string().contains("recovery summary")),
                reject_first && index == 1
            );
            assert_eq!(
                attr(chat, "gen_ai.conversation.id"),
                Some(&Value::String("recovery-otel".into()))
            );
            assert_eq!(
                attr(chat, "gen_ai.provider.name"),
                Some(&Value::String("recovery-provider".into()))
            );
            assert_eq!(
                attr(chat, "gen_ai.request.model"),
                Some(&Value::String("recovery-model".into()))
            );
            if reject_first && index == 0 {
                assert_eq!(
                    attr(chat, "error.type"),
                    Some(&Value::String("context_length_exceeded".into()))
                );
                assert!(matches!(chat.status, Status::Error { .. }));
                assert!(attr(chat, "gen_ai.usage.input_tokens").is_none());
                assert!(attr(chat, "gen_ai.response.id").is_none());
            } else {
                assert!(attr(chat, "error.type").is_none());
                assert_eq!(
                    attr(chat, "gen_ai.usage.input_tokens"),
                    Some(&Value::I64(7))
                );
                assert_eq!(
                    attr(chat, "gen_ai.usage.output_tokens"),
                    Some(&Value::I64(3))
                );
                assert_eq!(
                    attr(chat, "gen_ai.response.id"),
                    Some(&Value::String("response-1".into()))
                );
            }
        }
        if reject_first {
            assert_ne!(
                chats[0].capture_id, chats[1].capture_id,
                "each attempt must build a distinct OTel span"
            );
            assert_eq!(requests[0].turn_id, requests[1].turn_id);
            assert_eq!(requests[0].cache, requests[1].cache);
            assert_eq!(requests[0].available_tools, requests[1].available_tools);
        }
    }
}
