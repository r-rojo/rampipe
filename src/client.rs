//! A pure-socket client for `rampiped` (`src/bin/rampiped.rs`) — behind
//! its own `client` feature so a caller that only wants to *talk* to an
//! already-running daemon (taskpipe running with `--rampiped`, or the
//! separate AI-backed shell project) never links `llama-cpp-2` and its
//! native dependencies just to send a request over a Unix socket.
//!
//! One request per connection, matching `rampiped`'s own accept loop
//! (it reads exactly one line, replies with exactly one line, then
//! drops the connection) — `generate()` opens a fresh connection every
//! call rather than holding one open across calls.

use crate::protocol::{GenerateRequest, GenerateResponse, WireSampling};
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};

#[derive(Debug, thiserror::Error)]
pub enum RampipedError {
    #[error("connecting to rampiped socket {path} (is rampiped running?): {source}")]
    Connect { path: PathBuf, source: std::io::Error },
    #[error("encoding request to rampiped: {0}")]
    Encode(#[source] serde_json::Error),
    #[error("sending request to rampiped: {0}")]
    Send(#[source] std::io::Error),
    #[error("reading response from rampiped: {0}")]
    Read(#[source] std::io::Error),
    #[error("decoding response from rampiped: {0}")]
    Decode(#[source] serde_json::Error),
    #[error("rampiped reported an error: {0}")]
    Remote(String),
}

/// A successful generation, as reported by `rampiped` — mirrors
/// `rampipe::llama::GenerationResult`, but with `time_to_first_token`
/// already flattened to milliseconds (a `Duration` doesn't survive JSON
/// round-tripping without extra serde plumbing neither side otherwise
/// needs).
#[derive(Debug, Clone)]
pub struct GenerateOutcome {
    pub text: String,
    pub tokens_generated: usize,
    pub time_to_first_token_ms: u64,
}

pub struct RampipedClient {
    socket_path: PathBuf,
}

impl RampipedClient {
    /// Fails fast if nothing is listening at `socket_path` (a real
    /// connect attempt, immediately dropped) rather than only
    /// discovering that on the first real `generate()` call — the
    /// connection itself isn't kept, since `rampiped` serves one
    /// request per connection (see module doc comment).
    pub fn connect(socket_path: impl Into<PathBuf>) -> Result<Self, RampipedError> {
        let socket_path = socket_path.into();
        UnixStream::connect(&socket_path).map_err(|source| RampipedError::Connect { path: socket_path.clone(), source })?;
        Ok(Self { socket_path })
    }

    pub fn generate(
        &self,
        model_path: &Path,
        prompt: &str,
        max_new_tokens: i32,
        sampling: WireSampling,
    ) -> Result<GenerateOutcome, RampipedError> {
        let request = GenerateRequest { model_path: model_path.to_path_buf(), prompt: prompt.to_string(), max_new_tokens, sampling };
        let mut payload = serde_json::to_vec(&request).map_err(RampipedError::Encode)?;
        payload.push(b'\n');

        let mut stream =
            UnixStream::connect(&self.socket_path).map_err(|source| RampipedError::Connect { path: self.socket_path.clone(), source })?;
        stream.write_all(&payload).map_err(RampipedError::Send)?;
        stream.shutdown(std::net::Shutdown::Write).map_err(RampipedError::Send)?;

        let mut reader = BufReader::new(stream);
        let mut line = String::new();
        reader.read_line(&mut line).map_err(RampipedError::Read)?;
        let response: GenerateResponse = serde_json::from_str(line.trim()).map_err(RampipedError::Decode)?;

        match response {
            GenerateResponse::Ok { text, tokens_generated, time_to_first_token_ms } => {
                Ok(GenerateOutcome { text, tokens_generated, time_to_first_token_ms })
            }
            GenerateResponse::Err { message } => Err(RampipedError::Remote(message)),
        }
    }
}
