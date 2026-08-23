//! A pure-socket client for `rampiped` (`src/bin/rampiped.rs`) -- behind
//! its own `client` feature so a caller that only wants to *talk* to an
//! already-running daemon (taskpipe running with `--rampiped`, or the
//! separate AI-backed shell project) never links `llama-cpp-2` and its
//! native dependencies just to send a request over a Unix socket.
//!
//! One request per connection, matching `rampiped`'s own accept loop
//! (it reads exactly one line, replies with exactly one line, then
//! drops the connection) -- `generate()` opens a fresh connection every
//! call rather than holding one open across calls.

use crate::protocol::{
    ClientMessage, ConversationRequest, ConversationResponse, ConversationTurnRequest,
    EmbedRequest, EmbedResponse, GenerateRequest, GenerateResponse, GrammarCompletion,
    OpenConversationRequest, SnapshotRef, StatusResponse, WireOverflowPolicy, WirePooling,
    WireSampling,
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
    Connect {
        path: PathBuf,
        source: std::io::Error,
    },
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

/// A successful generation, as reported by `rampiped` -- mirrors
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
    /// See `rampipe::conversation::GenerationResult::tool_calls`. Parsed
    /// daemon-side, where the model's template lives, rather than being
    /// re-derived here: a client deliberately does not link the template
    /// machinery (that is the whole point of the `client` feature), so
    /// it could not parse these itself.
    pub tool_calls: Vec<crate::protocol::ToolCall>,
    /// See `rampipe::conversation::GenerationResult::truncated_tool_call`.
    pub truncated_tool_call: bool,
}

/// A successful [`RampipedClient::embed`] -- one vector per input text,
/// in request order.
#[derive(Debug, Clone)]
pub struct EmbedOutcome {
    pub vectors: Vec<Vec<f32>>,
    /// See `crate::protocol::EmbedResponse::Ok::n_embd`.
    pub n_embd: usize,
}

pub struct RampipedClient {
    socket_path: PathBuf,
}

impl RampipedClient {
    /// Fails fast if nothing is listening at `socket_path` (a real
    /// connect attempt, immediately dropped) rather than only
    /// discovering that on the first real `generate()` call -- the
    /// connection itself isn't kept, since `rampiped` serves one
    /// request per connection (see module doc comment).
    ///
    /// For a caller about to make a real request immediately afterward
    /// anyway (`generate()`, `status()`), that request already reports
    /// the identical connection failure on its own -- use [`Self::new`]
    /// instead and skip paying for this probe's own throwaway
    /// connection, which `rampiped`'s accept loop logs as a connection
    /// error (zero bytes sent before the probe drops it) for no reason
    /// a real caller needs. This is for a caller that wants reachability
    /// answered as its own fast, standalone question -- `brush-ai`'s own
    /// `is_rampiped_reachable` -- with no request to make yet.
    pub fn connect(socket_path: impl Into<PathBuf>) -> Result<Self, RampipedError> {
        let socket_path = socket_path.into();
        UnixStream::connect(&socket_path).map_err(|source| RampipedError::Connect {
            path: socket_path.clone(),
            source,
        })?;
        Ok(Self { socket_path })
    }

    /// A client for `socket_path`, with no upfront connectivity probe --
    /// see [`Self::connect`]'s own doc comment for when to prefer this.
    /// The first real call (`generate()`/`status()`) reports a dead
    /// socket exactly as clearly, just without a second, unused
    /// connection along the way.
    pub fn new(socket_path: impl Into<PathBuf>) -> Self {
        Self {
            socket_path: socket_path.into(),
        }
    }

    // The parameters mirror `GenerateRequest`'s fields one-for-one; a params
    // struct here would only duplicate the wire type it serializes into.
    #[allow(clippy::too_many_arguments)]
    pub fn generate(
        &self,
        model_path: &Path,
        prompt: &str,
        // `None` lets the daemon apply what this model is configured
        // for -- see `rampipe::model_settings`.
        max_new_tokens: Option<i32>,
        // `None` lets the daemon apply what this model is configured
        // for -- see `rampipe::model_settings`.
        sampling: Option<WireSampling>,
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
            UnixStream::connect(&self.socket_path).map_err(|source| RampipedError::Connect {
                path: self.socket_path.clone(),
                source,
            })?;
        stream.write_all(&payload).map_err(RampipedError::Send)?;
        stream
            .shutdown(std::net::Shutdown::Write)
            .map_err(RampipedError::Send)?;

        let mut reader = BufReader::new(stream);
        let mut line = String::new();
        reader.read_line(&mut line).map_err(RampipedError::Read)?;
        let response: GenerateResponse =
            serde_json::from_str(line.trim()).map_err(RampipedError::Decode)?;

        match response {
            GenerateResponse::Ok {
                text,
                tokens_generated,
                time_to_first_token_ms,
                formatted_prompt,
            } => Ok(GenerateOutcome {
                text,
                tokens_generated,
                time_to_first_token_ms,
                formatted_prompt,
                // One-shot `generate` offers no tools -- see
                // `rampipe::llama::LlamaSession::generate`.
                tool_calls: Vec::new(),
                truncated_tool_call: false,
            }),
            GenerateResponse::Err { message } => Err(RampipedError::Remote(message)),
        }
    }

    /// Embeds a batch of texts, returning one vector each in the same
    /// order. Same one-shot connection shape as
    /// [`RampipedClient::generate`].
    pub fn embed(
        &self,
        model_path: &Path,
        texts: &[String],
        pooling: WirePooling,
        normalize: bool,
    ) -> Result<EmbedOutcome, RampipedError> {
        let request = ClientMessage::Embed(EmbedRequest {
            model_path: model_path.to_path_buf(),
            texts: texts.to_vec(),
            pooling,
            normalize,
        });
        let mut payload = serde_json::to_vec(&request).map_err(RampipedError::Encode)?;
        payload.push(b'\n');

        let mut stream =
            UnixStream::connect(&self.socket_path).map_err(|source| RampipedError::Connect {
                path: self.socket_path.clone(),
                source,
            })?;
        stream.write_all(&payload).map_err(RampipedError::Send)?;
        stream
            .shutdown(std::net::Shutdown::Write)
            .map_err(RampipedError::Send)?;

        let mut reader = BufReader::new(stream);
        let mut line = String::new();
        reader.read_line(&mut line).map_err(RampipedError::Read)?;
        let response: EmbedResponse =
            serde_json::from_str(line.trim()).map_err(RampipedError::Decode)?;
        match response {
            EmbedResponse::Ok { vectors, n_embd } => Ok(EmbedOutcome { vectors, n_embd }),
            EmbedResponse::Err { message } => Err(RampipedError::Remote(message)),
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
            UnixStream::connect(&self.socket_path).map_err(|source| RampipedError::Connect {
                path: self.socket_path.clone(),
                source,
            })?;
        stream.write_all(&payload).map_err(RampipedError::Send)?;
        stream
            .shutdown(std::net::Shutdown::Write)
            .map_err(RampipedError::Send)?;

        let mut reader = BufReader::new(stream);
        let mut line = String::new();
        reader.read_line(&mut line).map_err(RampipedError::Read)?;
        serde_json::from_str(line.trim()).map_err(RampipedError::Decode)
    }
}

/// A still-open, KV-cache-persistent conversation against `rampiped` --
/// unlike `RampipedClient::generate()`, which opens a fresh connection
/// and shuts down its write half every single call, this holds one
/// connection open across every `send()`, mirroring
/// `rampipe::llama::Conversation`'s own in-process shape but over the
/// wire. Construct via [`RampipedConversation::open`].
pub struct RampipedConversation {
    reader: BufReader<UnixStream>,
    writer: UnixStream,
    /// What the daemon answered about tool calling when this
    /// conversation was opened -- see
    /// `protocol::ConversationResponse::OpenedWithTools`. `false` both
    /// when no tools were offered and when the daemon is an older build
    /// that answered a bare `Opened`.
    supports_tool_calls: bool,
    /// Completed user/assistant exchanges on this conversation, counted
    /// client-side. The daemon owns the real KV cache and its own turn
    /// boundaries; nothing in the wire protocol reports them back, so
    /// this is a local tally incremented per successful `send` purely to
    /// satisfy `conversation::ConversationHandle::turn_count`. Matches
    /// `llama::Conversation::turn_count`'s own unit (exchanges, not
    /// individual messages).
    turns: usize,
}

impl RampipedConversation {
    /// Opens a fresh connection, sends `OpenConversationRequest`, and
    /// waits for the daemon's `Opened` ack before returning -- a caller
    /// that gets `Ok` back knows the daemon has already loaded (or is
    /// already loading) `model_path` and is ready for `send()`.
    ///
    /// `restore_from`, when `Some`, reopens a conversation a prior
    /// [`RampipedConversation::snapshot`] saved instead of starting
    /// fresh -- `n_ctx`/`overflow` are then ignored in favor of whatever
    /// the snapshot itself recorded (see
    /// `OpenConversationRequest::restore_from`'s own doc comment).
    /// `system` and `tools` are forwarded to
    /// `rampipe::llama::ConversationOptions`' own fields of the same
    /// names -- see those for what each does and when each is ignored.
    /// Both are inert when `restore_from` is set: a restored
    /// conversation already holds its own in its reloaded KV cache.
    #[expect(
        clippy::too_many_arguments,
        reason = "each maps one-to-one onto an OpenConversationRequest field; grouping them would just be that struct again"
    )]
    pub fn open(
        socket_path: impl Into<PathBuf>,
        model_path: &Path,
        n_ctx: u32,
        overflow: WireOverflowPolicy,
        restore_from: Option<SnapshotRef>,
        system: Option<String>,
        tools: Vec<crate::protocol::ToolSpec>,
        tool_format: Option<crate::protocol::ToolFormat>,
    ) -> Result<Self, RampipedError> {
        let socket_path = socket_path.into();
        let stream =
            UnixStream::connect(&socket_path).map_err(|source| RampipedError::Connect {
                path: socket_path.clone(),
                source,
            })?;
        let mut writer = stream
            .try_clone()
            .map_err(|source| RampipedError::Connect {
                path: socket_path.clone(),
                source,
            })?;
        let mut reader = BufReader::new(stream);

        let request = ClientMessage::OpenConversation(OpenConversationRequest {
            model_path: model_path.to_path_buf(),
            n_ctx,
            overflow,
            restore_from,
            system,
            tools,
            tool_format,
        });
        let mut payload = serde_json::to_vec(&request).map_err(RampipedError::Encode)?;
        payload.push(b'\n');
        writer.write_all(&payload).map_err(RampipedError::Send)?;

        let mut line = String::new();
        reader.read_line(&mut line).map_err(RampipedError::Read)?;
        let response: ConversationResponse =
            serde_json::from_str(line.trim()).map_err(RampipedError::Decode)?;
        match response {
            // A bare `Opened` in reply to a request that *did* carry
            // tools means an older daemon -- see `OpenedWithTools`'s own
            // doc comment. Reporting `false` here is what makes a caller
            // fall back instead of waiting for calls that cannot come.
            ConversationResponse::Opened => Ok(Self {
                reader,
                writer,
                turns: 0,
                supports_tool_calls: false,
            }),
            ConversationResponse::OpenedWithTools {
                supports_tool_calls,
            } => Ok(Self {
                reader,
                writer,
                turns: 0,
                supports_tool_calls,
            }),
            ConversationResponse::Err { message } => Err(RampipedError::Remote(message)),
            ConversationResponse::Turn { .. } | ConversationResponse::Snapshotted => {
                Err(RampipedError::Remote(
                    "rampiped sent an unexpected response before the conversation was opened"
                        .to_string(),
                ))
            }
        }
    }

    /// Sends one new turn into this still-open conversation and waits
    /// for the reply -- same parameter shape as
    /// `rampipe::llama::Conversation::send`, just over the wire.
    pub fn send(
        &mut self,
        message: &str,
        // `None` lets the daemon apply what this model is configured
        // for -- see `rampipe::model_settings`.
        max_new_tokens: Option<i32>,
        sampling: Option<WireSampling>,
        grammar: Option<&str>,
        assistant_prefill: Option<&str>,
        grammar_completion: Option<GrammarCompletion>,
    ) -> Result<GenerateOutcome, RampipedError> {
        self.exchange(ConversationTurnRequest {
            tool_results: None,
            message: message.to_string(),
            max_new_tokens,
            sampling,
            grammar: grammar.map(str::to_string),
            assistant_prefill: assistant_prefill.map(str::to_string),
            grammar_completion,
        })
    }

    /// Whether the daemon said this conversation's tool calls are
    /// parseable -- see
    /// `protocol::ConversationResponse::OpenedWithTools`.
    #[must_use]
    pub fn supports_tool_calls(&self) -> bool {
        self.supports_tool_calls
    }

    /// Feeds executed tool results back over the wire -- the daemon
    /// routes these to `rampipe::llama::Conversation::send_tool_results`
    /// rather than to `send`, so they reach the model as results.
    pub fn send_tool_results(
        &mut self,
        results: &[String],
        // `None` lets the daemon apply what this model is configured
        // for -- see `rampipe::model_settings`.
        max_new_tokens: Option<i32>,
        sampling: Option<WireSampling>,
        grammar: Option<&str>,
        grammar_completion: Option<GrammarCompletion>,
    ) -> Result<GenerateOutcome, RampipedError> {
        self.exchange(ConversationTurnRequest {
            tool_results: Some(results.to_vec()),
            // Ignored daemon-side whenever `tool_results` is set (see
            // that field's own doc comment); empty rather than
            // duplicating the results into it, so a mistaken read of
            // this field on either side yields nothing rather than
            // something plausible.
            message: String::new(),
            max_new_tokens,
            sampling,
            grammar: grammar.map(str::to_string),
            assistant_prefill: None,
            grammar_completion,
        })
    }

    /// One request/reply round trip on this conversation's own socket --
    /// shared by `send` and `send_tool_results`, which differ only in
    /// the request they build.
    fn exchange(&mut self, turn: ConversationTurnRequest) -> Result<GenerateOutcome, RampipedError> {
        let turn = ConversationRequest::Turn(turn);
        let mut payload = serde_json::to_vec(&turn).map_err(RampipedError::Encode)?;
        payload.push(b'\n');
        self.writer
            .write_all(&payload)
            .map_err(RampipedError::Send)?;

        let mut line = String::new();
        self.reader
            .read_line(&mut line)
            .map_err(RampipedError::Read)?;
        let response: ConversationResponse =
            serde_json::from_str(line.trim()).map_err(RampipedError::Decode)?;
        match response {
            ConversationResponse::Turn {
                text,
                tokens_generated,
                time_to_first_token_ms,
                formatted_prompt,
                tool_calls,
                truncated_tool_call,
            } => {
                self.turns += 1;
                Ok(GenerateOutcome {
                    text,
                    tokens_generated,
                    time_to_first_token_ms,
                    formatted_prompt,
                    tool_calls,
                    truncated_tool_call,
                })
            }
            ConversationResponse::Err { message } => Err(RampipedError::Remote(message)),
            ConversationResponse::Opened
            | ConversationResponse::OpenedWithTools { .. }
            | ConversationResponse::Snapshotted => {
                Err(RampipedError::Remote(
                    "rampiped sent an unexpected response to a conversation turn".to_string(),
                ))
            }
        }
    }

    /// Persists this conversation's KV cache to `state_path`/`meta_path`
    /// and ends it -- the daemon closes the connection right after
    /// replying, so any further call on `self` (a `send()`, another
    /// `snapshot()`) fails rather than doing anything meaningful; this
    /// takes `&mut self` rather than consuming it only so it can satisfy
    /// [`crate::conversation::ConversationHandle::snapshot`]'s own
    /// by-reference shape (see that trait method's doc comment for why).
    /// A later [`RampipedConversation::open`] with a matching
    /// `restore_from` picks the conversation back up without replaying
    /// it turn by turn.
    pub fn snapshot(
        &mut self,
        state_path: impl Into<PathBuf>,
        meta_path: impl Into<PathBuf>,
    ) -> Result<(), RampipedError> {
        let request = ConversationRequest::Snapshot(SnapshotRef {
            state_path: state_path.into(),
            meta_path: meta_path.into(),
        });
        let mut payload = serde_json::to_vec(&request).map_err(RampipedError::Encode)?;
        payload.push(b'\n');
        self.writer
            .write_all(&payload)
            .map_err(RampipedError::Send)?;

        let mut line = String::new();
        self.reader
            .read_line(&mut line)
            .map_err(RampipedError::Read)?;
        let response: ConversationResponse =
            serde_json::from_str(line.trim()).map_err(RampipedError::Decode)?;
        match response {
            ConversationResponse::Snapshotted => Ok(()),
            ConversationResponse::Err { message } => Err(RampipedError::Remote(message)),
            ConversationResponse::Opened
            | ConversationResponse::OpenedWithTools { .. }
            | ConversationResponse::Turn { .. } => {
                Err(RampipedError::Remote(
                    "rampiped sent an unexpected response to a snapshot request".to_string(),
                ))
            }
        }
    }
}

/// Makes a daemon-backed conversation usable through the same
/// [`crate::conversation::ConversationHandle`] seam an in-process
/// `llama::Conversation` already implements, so a caller holding
/// `Box<dyn ConversationHandle>` no longer needs to know which backend
/// it got -- the case `llama::LocalModel`'s own doc comment already
/// named ("a `rampiped`-socket-backed... implementation") but nothing
/// provided.
///
/// Deliberately *not* gated on `llama`. It used to be, because the trait
/// itself lived behind that feature and there was no way to name it
/// otherwise -- which meant a client-only build (the whole point of the
/// `client` feature: talking to an already-running daemon without
/// linking `llama-cpp-2`) got `RampipedConversation`'s inherent `send`
/// and nothing generic. Now that the seam lives in
/// [`crate::conversation`], this is exactly the build that needs it.
/// `conversation::Sampling` -> `protocol::WireSampling`. A free
/// function rather than a closure inside one method, since both trait
/// methods below need it and duplicating it is exactly how the two
/// would drift.
fn sampling_to_wire(sampling: crate::conversation::Sampling) -> WireSampling {
    let penalties = |p: crate::conversation::Penalties| crate::protocol::WirePenalties {
        last_n: p.last_n,
        repeat: p.repeat,
        freq: p.freq,
        present: p.present,
    };
    match sampling {
        crate::conversation::Sampling::Greedy { penalties: p } => WireSampling::Greedy {
            penalties: penalties(p),
        },
        crate::conversation::Sampling::Temperature {
            temperature,
            top_k,
            top_p,
            min_p,
            seed,
            penalties: p,
        } => WireSampling::Temperature {
            temperature,
            top_k,
            top_p,
            min_p,
            seed,
            penalties: penalties(p),
        },
    }
}

impl crate::conversation::ConversationHandle for RampipedConversation {
    fn send(
        &mut self,
        message: &str,
        max_new_tokens: i32,
        sampling: crate::conversation::Sampling,
        grammar: Option<&str>,
        assistant_prefill: Option<&str>,
        grammar_completion: Option<GrammarCompletion>,
    ) -> Result<crate::conversation::GenerationResult, crate::conversation::ConversationError> {
        let outcome = RampipedConversation::send(
            self,
            message,
            Some(max_new_tokens),
            Some(sampling_to_wire(sampling)),
            grammar,
            assistant_prefill,
            grammar_completion,
        )
        .map_err(|e| crate::conversation::ConversationError::Backend(e.to_string()))?;
        Ok(crate::conversation::GenerationResult {
            text: outcome.text,
            // Back to a `Duration` from the milliseconds the wire format
            // flattens it to (see `GenerateOutcome`'s own doc comment) --
            // sub-millisecond precision is already gone by this point, so
            // this restores the shape, not the lost resolution.
            time_to_first_token: std::time::Duration::from_millis(outcome.time_to_first_token_ms),
            tokens_generated: outcome.tokens_generated,
            formatted_prompt: outcome.formatted_prompt,
            tool_calls: outcome.tool_calls,
            truncated_tool_call: outcome.truncated_tool_call,
        })
    }

    fn supports_tool_calls(&self) -> bool {
        RampipedConversation::supports_tool_calls(self)
    }

    fn send_tool_results(
        &mut self,
        results: &[String],
        max_new_tokens: i32,
        sampling: crate::conversation::Sampling,
        grammar: Option<&str>,
        grammar_completion: Option<GrammarCompletion>,
    ) -> Result<crate::conversation::GenerationResult, crate::conversation::ConversationError> {
        let outcome = RampipedConversation::send_tool_results(
            self,
            results,
            Some(max_new_tokens),
            Some(sampling_to_wire(sampling)),
            grammar,
            grammar_completion,
        )
        .map_err(|e| crate::conversation::ConversationError::Backend(e.to_string()))?;
        Ok(crate::conversation::GenerationResult {
            text: outcome.text,
            time_to_first_token: std::time::Duration::from_millis(outcome.time_to_first_token_ms),
            tokens_generated: outcome.tokens_generated,
            formatted_prompt: outcome.formatted_prompt,
            tool_calls: outcome.tool_calls,
            truncated_tool_call: outcome.truncated_tool_call,
        })
    }

    fn turn_count(&self) -> usize {
        self.turns
    }

    fn snapshot(
        &mut self,
        state_path: &std::path::Path,
        meta_path: &std::path::Path,
    ) -> Result<(), crate::conversation::ConversationError> {
        RampipedConversation::snapshot(self, state_path, meta_path)
            .map_err(|error| crate::conversation::ConversationError::Backend(error.to_string()))
    }
}
