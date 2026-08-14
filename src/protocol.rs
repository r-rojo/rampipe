//! The `rampiped` wire protocol: one JSON value + `\n` per message over a
//! Unix socket, matching `taskpipe::daemon`'s own established convention
//! (see that module's doc comment) rather than introducing a new shared
//! IPC crate for ~100 lines of socket boilerplate.
//!
//! Deliberately its own module, compiled unconditionally (no `llama`
//! feature gate): both the daemon (`src/bin/rampiped.rs`, needs `llama`
//! to actually run inference) and the client (`src/client.rs`, behind
//! the `client` feature, needs none of `llama-cpp-2`'s heavy native
//! dependencies to just talk to a daemon over a socket) depend on these
//! same request/response shapes. Keeping them here, not in `llama.rs`,
//! is what lets a caller that only wants to *talk* to a already-running
//! daemon avoid linking `llama-cpp-2` at all.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Mirrors `llama::Sampling` field-for-field but stays independent of it
/// (and of the `llama` feature) so this module compiles without
/// `llama-cpp-2` in the dependency graph. `src/bin/rampiped.rs` converts
/// between the two at the one point that actually needs both.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum WireSampling {
    Greedy,
    Temperature { temperature: f32, top_k: i32, seed: u32 },
}

/// How a caller's own generation loop should decide "the grammar-
/// constrained response is complete, stop before sampling again" --
/// needed because llama.cpp's grammar sampler has a real, reproducible
/// crash (a hard process abort, not a recoverable error) if `sample()`
/// is called again on one grammar-sampler instance once it's already
/// produced a token; see `rampipe::llama::run_generation_loop`'s own doc
/// comment for the full account. Serializable (unlike a raw closure) so
/// it can ride along in [`GenerateRequest`] and be turned into a real
/// predicate on whichever side actually runs the model -- the daemon's
/// own generation loop needs this exactly as much as an in-process
/// caller's does.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GrammarCompletion {
    /// Stop as soon as the accumulated generated text exactly equals one
    /// of `options` -- e.g. a classifier constrained to `"YES" | "NO"`.
    ExactMatch(Vec<String>),
    /// Stop as soon as `prefill` (see [`GenerateRequest::assistant_prefill`])
    /// followed by the accumulated generated text is a syntactically
    /// complete JSON value.
    ValidJson { prefill: String },
}

impl GrammarCompletion {
    /// Builds the real predicate `rampipe::llama::LlamaSession::generate`'s
    /// `grammar_complete` parameter expects, from this wire-friendly
    /// description.
    pub fn into_predicate(self) -> Box<dyn Fn(&str) -> bool> {
        match self {
            GrammarCompletion::ExactMatch(options) => Box::new(move |text: &str| options.iter().any(|option| option == text)),
            GrammarCompletion::ValidJson { prefill } => {
                Box::new(move |text: &str| serde_json::from_str::<serde_json::Value>(&format!("{prefill}{text}")).is_ok())
            }
        }
    }
}

/// A request to generate against one model. `model_path` is a real,
/// already-resolved local file path — HF repo resolution/download stays
/// entirely a client-side concern (taskpipe already does this), not
/// something the daemon or this protocol need to know about.
#[derive(Debug, Serialize, Deserialize)]
pub struct GenerateRequest {
    pub model_path: PathBuf,
    pub prompt: String,
    pub max_new_tokens: i32,
    pub sampling: WireSampling,
    /// A GBNF grammar to constrain generation to, applied on the daemon's
    /// own sampler chain — see `rampipe::llama::LlamaSession::generate`'s
    /// `grammar` parameter, which this mirrors. `None` behaves exactly as
    /// before this field existed: unconstrained generation.
    #[serde(default)]
    pub grammar: Option<String>,
    /// Text to seed the assistant's turn with before generation resumes —
    /// see `LlamaSession::generate`'s `assistant_prefill` parameter.
    #[serde(default)]
    pub assistant_prefill: Option<String>,
    /// See [`GrammarCompletion`]. `None` is only valid alongside
    /// `grammar: None` -- a grammar-constrained request that omits this
    /// risks the crash `GrammarCompletion`'s own doc comment describes.
    #[serde(default)]
    pub grammar_completion: Option<GrammarCompletion>,
}

/// One line back per request — an enum (not a bare struct + separate
/// error channel) so a caller can never observe a response that's
/// simultaneously a success and a failure; matches `taskpipe::daemon::
/// QueryResponse`'s own shape.
#[derive(Debug, Serialize, Deserialize)]
pub enum GenerateResponse {
    Ok {
        text: String,
        tokens_generated: usize,
        time_to_first_token_ms: u64,
        /// See `rampipe::llama::GenerationResult::formatted_prompt`'s doc
        /// comment — the same value, carried across the socket so a
        /// daemon-backed caller has the same visibility into what the
        /// model actually saw as an in-process caller already does.
        formatted_prompt: String,
    },
    Err {
        message: String,
    },
}

/// The first message a client sends on every fresh connection —
/// distinguishes today's one-shot `Generate` (connection closes after
/// one reply, unchanged from before this enum existed) from
/// `OpenConversation` (the connection stays open for a whole multi-turn
/// session — see `ConversationTurnRequest`/`ConversationResponse` for
/// what follows on that same connection once it's open). Wrapping the
/// request this way, rather than changing `GenerateRequest`'s own
/// shape, means the one-shot path's fields and behavior don't change at
/// all, only their wire envelope.
#[derive(Debug, Serialize, Deserialize)]
pub enum ClientMessage {
    Generate(GenerateRequest),
    OpenConversation(OpenConversationRequest),
    /// Asks for [`StatusResponse`] — a one-shot request/reply, connection
    /// closes after, same shape as `Generate`.
    Status,
}

/// What `rampiped` reports about itself for a `ClientMessage::Status`
/// request — the daemon-side half of a stale-binary check: a client that
/// also knows its own configured `rampiped_path` can stat that file
/// itself and compare against `exe_modified_unix_secs` here to tell
/// "reachable" apart from "reachable, but running code built before the
/// binary on disk was last rebuilt" (a real, live-reproduced failure
/// mode — a `rampiped` process that outlived a rebuild silently kept
/// serving the *old* wire protocol, closing every connection using the
/// new one with no response at all).
#[derive(Debug, Serialize, Deserialize)]
pub struct StatusResponse {
    pub pid: u32,
    pub exe_path: Option<PathBuf>,
    /// Unix seconds since epoch of the running binary's own mtime,
    /// captured once at daemon startup — deliberately *not* a live
    /// re-stat of `exe_path` on every status request, which would just
    /// reflect whatever happens to be on disk right now regardless of
    /// what code this already-running process actually has loaded (the
    /// exact trap this field exists to let a client detect).
    pub exe_modified_unix_secs: Option<u64>,
    pub resident_model_paths: Vec<PathBuf>,
}

/// Mirrors `llama::OverflowPolicy` field-for-field, same reason
/// [`WireSampling`] mirrors `Sampling` — keeps this module independent
/// of the `llama` feature.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum WireOverflowPolicy {
    Fail,
    DropOldestTurns,
}

/// Opens a persistent, KV-cache-backed conversation against
/// `model_path` on the connection this arrives on. The daemon replies
/// with one [`ConversationResponse::Opened`] (or `Err`), then the
/// connection stays open for any number of subsequent
/// [`ConversationTurnRequest`]/[`ConversationResponse`] round trips,
/// one per line each way, until the client drops the connection — no
/// explicit close message needed.
#[derive(Debug, Serialize, Deserialize)]
pub struct OpenConversationRequest {
    pub model_path: PathBuf,
    pub n_ctx: u32,
    pub overflow: WireOverflowPolicy,
}

/// One turn sent into an already-open conversation (see
/// [`OpenConversationRequest`]) — same fields as [`GenerateRequest`]
/// minus `model_path`/`prompt` (the conversation itself already fixes
/// the model; `message` stands in for `prompt`, scoped to just this
/// turn's new text).
#[derive(Debug, Serialize, Deserialize)]
pub struct ConversationTurnRequest {
    pub message: String,
    pub max_new_tokens: i32,
    pub sampling: WireSampling,
    #[serde(default)]
    pub grammar: Option<String>,
    #[serde(default)]
    pub assistant_prefill: Option<String>,
    #[serde(default)]
    pub grammar_completion: Option<GrammarCompletion>,
}

/// One line back per conversation-mode message — `Opened` acknowledges
/// [`OpenConversationRequest`], `Turn` answers a
/// [`ConversationTurnRequest`], `Err` can follow either.
#[derive(Debug, Serialize, Deserialize)]
pub enum ConversationResponse {
    Opened,
    Turn {
        text: String,
        tokens_generated: usize,
        time_to_first_token_ms: u64,
        /// See `GenerateResponse::Ok::formatted_prompt` — here, just
        /// this turn's own new text (mirrors
        /// `rampipe::llama::Conversation::send`'s `GenerationResult::
        /// formatted_prompt`, which is scoped to the turn, not the whole
        /// conversation).
        formatted_prompt: String,
    },
    Err {
        message: String,
    },
}

/// `~/.rampipe/rampiped.sock` — the one real source of truth for
/// "where's the daemon," so `rampiped`'s own `main()` and every client
/// (`rampipe::client`, or an external caller like taskpipe) fall back to
/// the same path instead of each hardcoding it independently. Without
/// this, two independently started processes agreeing to share one
/// daemon required a human to type the identical path on both command
/// lines; with it, doing nothing on either side already agrees. `--socket`
/// (`rampiped`) and an explicit client-supplied path still override this,
/// same as before — this is only ever the fallback when neither says
/// otherwise.
///
/// `None` only when `$HOME` isn't set — matches `system_free_bytes`'s own
/// "can't determine, never guess" convention (`crate::lib`) rather than
/// falling back to something like `/tmp`, which would silently put the
/// socket somewhere no other process's own unset-`$HOME` fallback would
/// agree on either.
pub fn default_socket_path() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    Some(PathBuf::from(home).join(".rampipe").join("rampiped.sock"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_socket_path_ends_with_the_expected_relative_layout() {
        // Not asserting the exact absolute path (that's whatever $HOME
        // happens to be in the test environment) -- just the stable part
        // this function's own contract promises.
        let path = default_socket_path().expect("HOME should be set in a real test environment");
        assert!(path.ends_with(".rampipe/rampiped.sock"));
    }
}
