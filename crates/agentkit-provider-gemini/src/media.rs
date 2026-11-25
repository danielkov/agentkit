use agentkit_core::{DataRef, FilePart, MediaPart};
use base64::Engine;
use serde_json::{Value, json};

use crate::request::BuildError;

/// Encodes a [`MediaPart`] into a Gemini `inlineData` or `fileData` part.
///
/// Gemini natively supports image, audio, video, and binary/document MIME types.
pub(crate) fn media_to_part(media: &MediaPart) -> Result<Value, BuildError> {
    data_ref_to_part(&media.data, &media.mime_type)
}

/// Encodes a [`FilePart`] into a Gemini `inlineData`, `fileData`, or `text` part.
pub(crate) fn file_to_part(file: &FilePart) -> Result<Value, BuildError> {
    let mime = file
        .mime_type
        .as_deref()
        .unwrap_or("application/octet-stream");
    match &file.data {
        DataRef::InlineText(text) if mime == "text/plain" || mime.starts_with("text/") => {
            if let Some((header, encoded)) = text.split_once(";base64,") {
                let detected = header.strip_prefix("data:").unwrap_or(mime);
                Ok(json!({
                    "inlineData": {
                        "mimeType": detected,
                        "data": encoded,
                    }
                }))
            } else if is_uri(text) {
                Ok(json!({
                    "fileData": {
                        "mimeType": mime,
                        "fileUri": text,
                    }
                }))
            } else {
                Ok(json!({ "text": text }))
            }
        }
        _ => data_ref_to_part(&file.data, mime),
    }
}

fn data_ref_to_part(data: &DataRef, mime_type: &str) -> Result<Value, BuildError> {
    match data {
        DataRef::Uri(uri) => Ok(json!({
            "fileData": {
                "mimeType": mime_type,
                "fileUri": uri,
            }
        })),
        DataRef::InlineBytes(bytes) => Ok(json!({
            "inlineData": {
                "mimeType": mime_type,
                "data": base64::engine::general_purpose::STANDARD.encode(bytes),
            }
        })),
        DataRef::InlineText(text) => {
            if let Some((header, encoded)) = text.split_once(";base64,") {
                let detected = header.strip_prefix("data:").unwrap_or(mime_type);
                Ok(json!({
                    "inlineData": {
                        "mimeType": detected,
                        "data": encoded,
                    }
                }))
            } else if is_uri(text) {
                Ok(json!({
                    "fileData": {
                        "mimeType": mime_type,
                        "fileUri": text,
                    }
                }))
            } else {
                Err(BuildError::UnsupportedDataRef(
                    "media/file inputs must be a URI, data URL, or inline bytes".into(),
                ))
            }
        }
        DataRef::Handle(handle) => Err(BuildError::UnsupportedDataRef(format!(
            "artifact handle {} cannot be sent directly to Gemini",
            handle.0
        ))),
    }
}

fn is_uri(s: &str) -> bool {
    s.starts_with("http://") || s.starts_with("https://") || s.starts_with("gs://")
}
