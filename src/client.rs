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

use crate::protocol::{
    ClientMessage, ConversationResponse, ConversationTurnRequest, GenerateRequest, GenerateResponse, GrammarCompletion,
    OpenConversationRequest, StatusResponse, WireOverflowPolicy, WireSampling,
};
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};

/// Re-exported so a caller only needs `rampipe::client` to both find and
/// connect to the daemon, without a separate `use rampipe::protocol` just
/// for this one function.
pub use crate::protocol::default_socket_path;

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
    /// See `rampipe::llama::GenerationResult::formatted_prompt`'s doc
    /// comment.
    pub formatted_prompt: String,
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
        grammar: Option<&str>,
        assistant_prefill: Option<&str>,
        grammar_completion: Option<GrammarCompletion>,
    ) -> Result<GenerateOutcome, RampipedError> {
        let request = ClientMessage::Generate(GenerateRequest {
            model_path: model_path.to_path_buf(),
            prompt: prompt.to_string(),
            max_new_tokens,
            sampling,
            grammar: grammar.map(str::to_string),
            assistant_prefill: assistant_prefill.map(str::to_string),
            grammar_completion,
        });
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
            GenerateResponse::Ok { text, tokens_generated, time_to_first_token_ms, formatted_prompt } => {
                Ok(GenerateOutcome { text, tokens_generated, time_to_first_token_ms, formatted_prompt })
            }
            GenerateResponse::Err { message } => Err(RampipedError::Remote(message)),
        }
    }

    /// Asks the daemon about itself -- pid, its own executable's path
    /// and startup-time mtime, and what's currently resident. See
    /// [`StatusResponse`]'s own doc comment for why `exe_modified_unix_secs`
    /// specifically is the useful part: a caller that also knows the
    /// `rampiped` binary's path can stat that file itself and compare,
    /// telling "reachable" apart from "reachable, but running code from
    /// before the last rebuild."
    pub fn status(&self) -> Result<StatusResponse, RampipedError> {
        let request = ClientMessage::Status;
        let mut payload = serde_json::to_vec(&request).map_err(RampipedError::Encode)?;
        payload.push(b'\n');

        let mut stream =
            UnixStream::connect(&self.socket_path).map_err(|source| RampipedError::Connect { path: self.socket_path.clone(), source })?;
        stream.write_all(&payload).map_err(RampipedError::Send)?;
        stream.shutdown(std::net::Shutdown::Write).map_err(RampipedError::Send)?;

        let mut reader = BufReader::new(stream);
        let mut line = String::new();
        reader.read_line(&mut line).map_err(RampipedError::Read)?;
        serde_json::from_str(line.trim()).map_err(RampipedError::Decode)
    }
}

/// A still-open, KV-cache-persistent conversation against `rampiped` —
/// unlike `RampipedClient::generate()`, which opens a fresh connection
/// and shuts down its write half every single call, this holds one
/// connection open across every `send()`, mirroring
/// `rampipe::llama::Conversation`'s own in-process shape but over the
/// wire. Construct via [`RampipedConversation::open`].
pub struct RampipedConversation {
    reader: BufReader<UnixStream>,
    writer: UnixStream,
}

impl RampipedConversation {
    /// Opens a fresh connection, sends `OpenConversationRequest`, and
    /// waits for the daemon's `Opened` ack before returning — a caller
    /// that gets `Ok` back knows the daemon has already loaded (or is
    /// already loading) `model_path` and is ready for `send()`.
    pub fn open(socket_path: impl Into<PathBuf>, model_path: &Path, n_ctx: u32, overflow: WireOverflowPolicy) -> Result<Self, RampipedError> {
        let socket_path = socket_path.into();
        let stream = UnixStream::connect(&socket_path).map_err(|source| RampipedError::Connect { path: socket_path.clone(), source })?;
        let mut writer = stream.try_clone().map_err(|source| RampipedError::Connect { path: socket_path.clone(), source })?;
        let mut reader = BufReader::new(stream);

        let request =
            ClientMessage::OpenConversation(OpenConversationRequest { model_path: model_path.to_path_buf(), n_ctx, overflow });
        let mut payload = serde_json::to_vec(&request).map_err(RampipedError::Encode)?;
        payload.push(b'\n');
        writer.write_all(&payload).map_err(RampipedError::Send)?;

        let mut line = String::new();
        reader.read_line(&mut line).map_err(RampipedError::Read)?;
        let response: ConversationResponse = serde_json::from_str(line.trim()).map_err(RampipedError::Decode)?;
        match response {
            ConversationResponse::Opened => Ok(Self { reader, writer }),
            ConversationResponse::Err { message } => Err(RampipedError::Remote(message)),
            ConversationResponse::Turn { .. } => {
                Err(RampipedError::Remote("rampiped sent a turn response before the conversation was opened".to_string()))
            }
        }
    }

    /// Sends one new turn into this still-open conversation and waits
    /// for the reply — same parameter shape as
    /// `rampipe::llama::Conversation::send`, just over the wire.
    pub fn send(
        &mut self,
        message: &str,
        max_new_tokens: i32,
        sampling: WireSampling,
        grammar: Option<&str>,
        assistant_prefill: Option<&str>,
        grammar_completion: Option<GrammarCompletion>,
    ) -> Result<GenerateOutcome, RampipedError> {
        let turn = ConversationTurnRequest {
            message: message.to_string(),
            max_new_tokens,
            sampling,
            grammar: grammar.map(str::to_string),
            assistant_prefill: assistant_prefill.map(str::to_string),
            grammar_completion,
        };
        let mut payload = serde_json::to_vec(&turn).map_err(RampipedError::Encode)?;
        payload.push(b'\n');
        self.writer.write_all(&payload).map_err(RampipedError::Send)?;

        let mut line = String::new();
        self.reader.read_line(&mut line).map_err(RampipedError::Read)?;
        let response: ConversationResponse = serde_json::from_str(line.trim()).map_err(RampipedError::Decode)?;
        match response {
            ConversationResponse::Turn { text, tokens_generated, time_to_first_token_ms, formatted_prompt } => {
                Ok(GenerateOutcome { text, tokens_generated, time_to_first_token_ms, formatted_prompt })
            }
            ConversationResponse::Err { message } => Err(RampipedError::Remote(message)),
            ConversationResponse::Opened => Err(RampipedError::Remote("rampiped re-sent Opened mid-conversation".to_string())),
        }
    }
}
