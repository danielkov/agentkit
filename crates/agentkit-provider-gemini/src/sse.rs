//! Minimal Server-Sent Events framer with incremental UTF-8 reassembly.
//!
//! Gemini's `:streamGenerateContent?alt=sse` endpoint emits `data: <json>\n\n`
//! records (LF or CRLF). Buffering raw bytes before UTF-8 validation ensures
//! multi-byte UTF-8 sequences split across HTTP chunk boundaries are decoded
//! without spurious errors.

/// An individual parsed SSE record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SseEvent {
    /// Concatenated `data:` payload lines.
    pub(crate) data: String,
}

/// Chunked byte decoder that frames SSE records safely across arbitrary byte boundaries.
#[derive(Default)]
pub(crate) struct SseDecoder {
    /// Incomplete UTF-8 trailing bytes carried from the previous chunk.
    utf8_tail: Vec<u8>,
    /// Decoded UTF-8 text not yet framed into a complete SSE record.
    buffer: String,
}

impl SseDecoder {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Feeds raw bytes from the HTTP response stream and returns any complete SSE events.
    pub(crate) fn feed_bytes(
        &mut self,
        bytes: &[u8],
    ) -> Result<Vec<SseEvent>, std::str::Utf8Error> {
        let mut combined;
        let slice = if self.utf8_tail.is_empty() {
            bytes
        } else {
            combined = std::mem::take(&mut self.utf8_tail);
            combined.extend_from_slice(bytes);
            &combined
        };

        match std::str::from_utf8(slice) {
            Ok(valid) => {
                self.buffer.push_str(valid);
            }
            Err(error) => {
                let valid_up_to = error.valid_up_to();
                if valid_up_to > 0 {
                    let valid = std::str::from_utf8(&slice[..valid_up_to])
                        .expect("prefix up to valid_up_to is valid UTF-8");
                    self.buffer.push_str(valid);
                }
                if error.error_len().is_none() {
                    // Incomplete multi-byte sequence at end of chunk; carry forward.
                    self.utf8_tail.extend_from_slice(&slice[valid_up_to..]);
                } else {
                    return Err(error);
                }
            }
        }

        Ok(self.drain_records())
    }

    /// Flushes any trailing record at EOF (if the server omitted the final blank line).
    pub(crate) fn finish(&mut self) -> Result<Vec<SseEvent>, std::str::Utf8Error> {
        if !self.utf8_tail.is_empty() {
            let tail = std::mem::take(&mut self.utf8_tail);
            let _ = std::str::from_utf8(&tail)?;
        }
        let mut out = self.drain_records();
        let remaining = self.buffer.trim();
        if !remaining.is_empty()
            && let Some(event) = parse_record(remaining)
        {
            out.push(event);
        }
        self.buffer.clear();
        Ok(out)
    }

    fn drain_records(&mut self) -> Vec<SseEvent> {
        let mut out = Vec::new();
        while let Some(end) = find_record_boundary(&self.buffer) {
            let record: String = self.buffer.drain(..end).collect();
            let record = record.trim_end_matches(&['\r', '\n'][..]);
            if record.is_empty() {
                continue;
            }
            if let Some(event) = parse_record(record) {
                out.push(event);
            }
        }
        out
    }
}

fn find_record_boundary(buf: &str) -> Option<usize> {
    let lf = buf.find("\n\n").map(|idx| idx + 2);
    let crlf = buf.find("\r\n\r\n").map(|idx| idx + 4);
    match (lf, crlf) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

fn parse_record(record: &str) -> Option<SseEvent> {
    let mut data_lines: Vec<&str> = Vec::new();
    for raw_line in record.split('\n') {
        let line = raw_line.strip_suffix('\r').unwrap_or(raw_line);
        if line.is_empty() || line.starts_with(':') {
            continue;
        }
        if let Some(rest) = line.strip_prefix("data:") {
            data_lines.push(rest.strip_prefix(' ').unwrap_or(rest));
        }
    }
    if data_lines.is_empty() {
        return None;
    }
    Some(SseEvent {
        data: data_lines.join("\n"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_crlf_and_lf_records() {
        let mut dec = SseDecoder::new();
        let events = dec
            .feed_bytes(b"data: {\"a\":1}\r\n\r\ndata: {\"b\":2}\n\n")
            .unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].data, r#"{"a":1}"#);
        assert_eq!(events[1].data, r#"{"b":2}"#);
    }

    #[test]
    fn handles_split_multibyte_utf8_across_chunks() {
        let mut dec = SseDecoder::new();
        let full = "data: {\"text\":\"こんにちは\"}\n\n".as_bytes();
        let split = 17; // mid-character inside UTF-8 sequence
        let first = dec.feed_bytes(&full[..split]).unwrap();
        assert!(first.is_empty());
        let second = dec.feed_bytes(&full[split..]).unwrap();
        assert_eq!(second.len(), 1);
        assert_eq!(second[0].data, r#"{"text":"こんにちは"}"#);
    }

    #[test]
    fn flushes_trailing_record_without_blank_line() {
        let mut dec = SseDecoder::new();
        assert!(dec.feed_bytes(b"data: {\"done\":true}").unwrap().is_empty());
        let tail = dec.finish().unwrap();
        assert_eq!(tail.len(), 1);
        assert_eq!(tail[0].data, r#"{"done":true}"#);
    }
}
