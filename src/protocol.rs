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
    Greedy {
        penalties: WirePenalties,
    },
    Temperature {
        temperature: f32,
        top_k: i32,
        seed: u32,
        penalties: WirePenalties,
    },
}

/// Mirrors `conversation::Penalties` field-for-field, same reasoning as
/// `WireSampling` itself.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct WirePenalties {
    pub last_n: i32,
    pub repeat: f32,
    pub freq: f32,
    pub present: f32,
}

impl Default for WirePenalties {
    fn default() -> Self {
        Self {
            last_n: 0,
            repeat: 1.0,
            freq: 0.0,
            present: 0.0,
        }
    }
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
            GrammarCompletion::ExactMatch(options) => {
                Box::new(move |text: &str| options.iter().any(|option| option == text))
            }
            GrammarCompletion::ValidJson { prefill } => Box::new(move |text: &str| {
                serde_json::from_str::<serde_json::Value>(&format!("{prefill}{text}")).is_ok()
            }),
        }
    }
}

/// One tool offered to a model, in the OpenAI-style shape every chat
/// template targeted here expects to iterate (`{type: "function",
/// function: {name, description, parameters}}`).
///
/// `parameters` is a JSON Schema object carried as an opaque
/// `serde_json::Value` rather than a typed schema struct: templates
/// walk it generically (Qwen3-Coder's own renders every key it doesn't
/// recognize as its own XML element), so any structure this crate
/// imposed would be one more thing to keep in sync with a spec it does
/// not own.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolSpec {
    #[serde(rename = "type")]
    pub kind: String,
    pub function: ToolFunction,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolFunction {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

impl ToolSpec {
    /// `kind` is always `"function"` today -- every template this
    /// targets branches on `tool.function is defined` and nothing else,
    /// so a constructor is friendlier than making every call site repeat
    /// the one legal value.
    #[must_use]
    pub fn new(name: impl Into<String>, description: impl Into<String>, parameters: serde_json::Value) -> Self {
        Self {
            kind: "function".to_string(),
            function: ToolFunction {
                name: name.into(),
                description: description.into(),
                parameters,
            },
        }
    }
}

/// One tool call a model actually emitted, already decoded out of
/// whatever textual format its template uses (see
/// `crate::tool_format`). `arguments` is a JSON object; the
/// one-block-per-argument family carries every value as a string, since
/// that format has no types of its own -- a caller wanting a number or
/// bool parses it from the string, exactly as it would have had to from
/// that format's raw text.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub name: String,
    pub arguments: serde_json::Value,
}

/// How a model writes a tool call, either derived from its own chat
/// template (`crate::tool_format::derive_tool_call_format`) or, when
/// that declines, supplied by the host alongside its other per-model
/// settings.
///
/// Serializable because the derivation happens wherever the template
/// is -- in the daemon, which owns the model -- while a client may want
/// to know what it is getting back. Two variants, not an open-ended
/// grammar, because these are the two families real templates actually
/// use; anything else returns `None` from derivation and needs a config
/// entry rather than a silent wrong guess.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ToolFormat {
    /// A single JSON object per call, wrapped in fixed delimiters --
    /// Hermes and Qwen2.5-style. `name_key`/`arguments_key` are
    /// recovered by looking for the probe's *values*, never by assuming
    /// the conventional spellings.
    Json {
        call_open: String,
        call_close: String,
        name_key: String,
        arguments_key: String,
    },
    /// One delimited block per argument, values unquoted and unescaped
    /// -- Qwen3-Coder-style:
    /// `<tool_call><function=NAME><parameter=ARG>value</parameter>...`
    Delimited {
        call_open: String,
        name_close: String,
        arg_open: String,
        arg_name_close: String,
        arg_close: String,
        call_close: String,
    },
}

/// A request to generate against one model. `model_path` is a real,
/// already-resolved local file path -- HF repo resolution/download stays
/// entirely a client-side concern (taskpipe already does this), not
/// something the daemon or this protocol need to know about.
#[derive(Debug, Serialize, Deserialize)]
pub struct GenerateRequest {
    pub model_path: PathBuf,
    pub prompt: String,
    /// `None` means "as much as this model is configured for".
    #[serde(default)]
    pub max_new_tokens: Option<i32>,
    /// `None` means "use whatever this model is configured for" -- the
    /// same rule as [`ConversationTurnRequest::sampling`], and for the
    /// same reason: a one-shot generation against a model should get
    /// that model's own recommended settings without the caller
    /// carrying them.
    #[serde(default)]
    pub sampling: Option<WireSampling>,
    /// A GBNF grammar to constrain generation to, applied on the daemon's
    /// own sampler chain -- see `rampipe::llama::LlamaSession::generate`'s
    /// `grammar` parameter, which this mirrors. `None` behaves exactly as
    /// before this field existed: unconstrained generation.
    #[serde(default)]
    pub grammar: Option<String>,
    /// Text to seed the assistant's turn with before generation resumes --
    /// see `LlamaSession::generate`'s `assistant_prefill` parameter.
    #[serde(default)]
    pub assistant_prefill: Option<String>,
    /// See [`GrammarCompletion`]. `None` is only valid alongside
    /// `grammar: None` -- a grammar-constrained request that omits this
    /// risks the crash `GrammarCompletion`'s own doc comment describes.
    #[serde(default)]
    pub grammar_completion: Option<GrammarCompletion>,
}

/// One line back per request -- an enum (not a bare struct + separate
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
        /// comment -- the same value, carried across the socket so a
        /// daemon-backed caller has the same visibility into what the
        /// model actually saw as an in-process caller already does.
        formatted_prompt: String,
    },
    Err {
        message: String,
    },
}

/// How token embeddings get collapsed into one vector per text.
/// Mirrors `llama_cpp_2::context::params::LlamaPoolingType`, same reason
/// [`WireSampling`] mirrors `Sampling` -- keeps this module free of the
/// `llama` feature.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WirePooling {
    /// Whatever the model's own GGUF metadata declares -- CLS for a
    /// bge-family encoder, Mean for nomic-embed, and so on. The right
    /// default: each retrieval encoder was trained with one specific
    /// pooling and using a different one silently degrades the vectors
    /// rather than failing.
    ///
    /// Note this is not a no-op. llama.cpp resolves an unspecified
    /// pooling type against the model's hparams and, if *those* are also
    /// unspecified, lands on `NONE` -- which makes
    /// `llama_get_embeddings_seq` return null and the binding report
    /// `EmbeddingsError::NonePoolType`. `LlamaSession::embed` retries
    /// once with [`WirePooling::Mean`] in that case rather than
    /// surfacing an error a caller can do nothing about.
    Model,
    Mean,
    Cls,
    Last,
}

fn default_pooling() -> WirePooling {
    WirePooling::Model
}

fn default_true() -> bool {
    true
}

/// A request to embed one or more texts against `model_path`.
///
/// Batched (`texts` is a list, not one string) because the calling
/// pattern this exists for -- embedding a vocabulary of short cue
/// phrases -- is many tiny texts at once, and one round trip plus one
/// context creation for the whole batch is most of the cost saved.
#[derive(Debug, Serialize, Deserialize)]
pub struct EmbedRequest {
    pub model_path: PathBuf,
    pub texts: Vec<String>,
    #[serde(default = "default_pooling")]
    pub pooling: WirePooling,
    /// L2-normalize each vector, so a dot product is cosine similarity.
    /// Defaults to true: every consumer of this so far wants cosine, and
    /// normalizing on the daemon side keeps callers from each
    /// reimplementing it.
    #[serde(default = "default_true")]
    pub normalize: bool,
}

/// One line back per [`EmbedRequest`]. Same success-or-failure-never-both
/// shape as [`GenerateResponse`], for the same reason.
#[derive(Debug, Serialize, Deserialize)]
pub enum EmbedResponse {
    Ok {
        /// One vector per input text, in the same order.
        vectors: Vec<Vec<f32>>,
        /// Width of each vector -- the model's `n_embd_out`, which is not
        /// always `n_embd` (see `LlamaContext::embeddings_seq_ith`'s own
        /// doc comment). Carried explicitly so a caller can reject a
        /// dimension mismatch against vectors it stored earlier without
        /// having to infer the width from a possibly-empty batch.
        n_embd: usize,
    },
    Err {
        message: String,
    },
}

/// The first message a client sends on every fresh connection --
/// distinguishes today's one-shot `Generate` (connection closes after
/// one reply, unchanged from before this enum existed) from
/// `OpenConversation` (the connection stays open for a whole multi-turn
/// session -- see `ConversationTurnRequest`/`ConversationResponse` for
/// what follows on that same connection once it's open). Wrapping the
/// request this way, rather than changing `GenerateRequest`'s own
/// shape, means the one-shot path's fields and behavior don't change at
/// all, only their wire envelope.
#[derive(Debug, Serialize, Deserialize)]
pub enum ClientMessage {
    Generate(GenerateRequest),
    OpenConversation(OpenConversationRequest),
    /// Asks for [`StatusResponse`] -- a one-shot request/reply, connection
    /// closes after, same shape as `Generate`.
    Status,
    /// Embeds a batch of texts -- one-shot, same connection shape as
    /// `Generate`. Separate from `Generate` rather than a mode of it
    /// because the two share no parameters at all: there is no sampling,
    /// no grammar, no prefill and no token budget in an embedding
    /// request, and nothing but a vector comes back.
    Embed(EmbedRequest),
}

/// What `rampiped` reports about itself for a `ClientMessage::Status`
/// request -- the daemon-side half of a stale-binary check: a client that
/// also knows its own configured `rampiped_path` can stat that file
/// itself and compare against `exe_modified_unix_secs` here to tell
/// "reachable" apart from "reachable, but running code built before the
/// binary on disk was last rebuilt" (a real, live-reproduced failure
/// mode -- a `rampiped` process that outlived a rebuild silently kept
/// serving the *old* wire protocol, closing every connection using the
/// new one with no response at all).
#[derive(Debug, Serialize, Deserialize)]
pub struct StatusResponse {
    pub pid: u32,
    pub exe_path: Option<PathBuf>,
    /// Unix seconds since epoch of the running binary's own mtime,
    /// captured once at daemon startup -- deliberately *not* a live
    /// re-stat of `exe_path` on every status request, which would just
    /// reflect whatever happens to be on disk right now regardless of
    /// what code this already-running process actually has loaded (the
    /// exact trap this field exists to let a client detect).
    pub exe_modified_unix_secs: Option<u64>,
    /// Seconds since this process started -- lets a caller tell "long-
    /// running, seen real traffic" apart from "just restarted" without
    /// needing its own separate liveness tracking.
    pub uptime_secs: u64,
    /// Successful `Generate` requests and conversation turns combined,
    /// since daemon startup. Cumulative, not a rate -- a caller wanting
    /// throughput derives it from two samples' delta over their own
    /// elapsed time, same as any other counter-style metric.
    pub requests_served: u64,
    /// Requests/turns that reached `rampiped` but failed (model load
    /// error, decode error, etc.) -- counted separately from
    /// `requests_served` rather than folded in, so a caller can compute
    /// an error rate directly instead of needing to infer it.
    pub requests_failed: u64,
    /// Sum of `tokens_generated` across every successful request/turn
    /// since startup.
    pub total_tokens_generated: u64,
    /// Total bytes currently mapped across every resident model, per
    /// `SwapRegistry::mapped_bytes()` -- system RAM, not VRAM.
    pub resident_bytes: usize,
    /// Free/total VRAM in bytes, per whatever GPU backend device is
    /// actually present (CUDA or Metal) -- `None` on a CPU-only build or
    /// if no GPU device was found. Queried live at `Status`-request time,
    /// not cached, since free VRAM is exactly the number that changes
    /// moment to moment.
    pub gpu_free_bytes: Option<u64>,
    pub gpu_total_bytes: Option<u64>,
    pub models: Vec<ModelStatus>,
}

/// Per-model counters, one entry per currently-resident model. Reset if a
/// model is evicted and later reloaded -- these describe the *current*
/// residency, not a model's lifetime history across evictions.
#[derive(Debug, Serialize, Deserialize)]
pub struct ModelStatus {
    pub path: PathBuf,
    pub requests_served: u64,
    pub tokens_generated: u64,
    /// Unix seconds of this model's most recent successful request/turn.
    pub last_used_unix_secs: u64,
}

/// Mirrors `llama::OverflowPolicy` field-for-field, same reason
/// [`WireSampling`] mirrors `Sampling` -- keeps this module independent
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
/// [`ConversationRequest`]/[`ConversationResponse`] round trips, one per
/// line each way, until the client drops the connection (or a
/// `Snapshot` request ends it deliberately -- see [`ConversationRequest`]
/// and [`ConversationResponse::Snapshotted`]).
#[derive(Debug, Serialize, Deserialize)]
pub struct OpenConversationRequest {
    pub model_path: PathBuf,
    pub n_ctx: u32,
    pub overflow: WireOverflowPolicy,
    /// Opens from a prior [`ConversationRequest::Snapshot`] instead of
    /// starting fresh -- `n_ctx` above is then ignored in favor of
    /// whatever the snapshot itself recorded (see
    /// `rampipe::llama::LlamaSession::open_conversation_from_state`'s own
    /// doc comment for why the saved value wins over a caller-supplied
    /// one). Fails closed (`ConversationResponse::Err`) if the snapshot's
    /// own recorded model doesn't match `model_path`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub restore_from: Option<SnapshotRef>,
    /// Instructions for the model's own system block -- see
    /// `rampipe::llama::ConversationOptions::system`. `#[serde(default)]`
    /// throughout, so a client built before these existed keeps talking
    /// to a newer daemon unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    /// Tools to offer, rendered in this model's own tool-call format --
    /// see `rampipe::llama::ConversationOptions::tools`. Ignored when
    /// `restore_from` is set: a restored conversation's tools are
    /// already in the KV cache being reloaded, and are recorded in its
    /// own snapshot sidecar.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolSpec>,
    /// Host-supplied fallback for a model whose template can't be
    /// probed for its call format -- see
    /// `rampipe::llama::ConversationOptions::tool_format`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_format: Option<ToolFormat>,
}

/// Where a conversation's saved KV-cache state lives -- both paths are
/// caller-chosen (see `Conversation::save_state`'s own doc comment), so
/// this is just what a [`ConversationRequest::Snapshot`] and a later
/// [`OpenConversationRequest::restore_from`] need to agree on to name the
/// same saved conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotRef {
    pub state_path: PathBuf,
    pub meta_path: PathBuf,
}

/// What arrives on an already-open conversation's connection -- `Turn`
/// is today's only case, renamed from the bare
/// (now-removed) `ConversationTurnRequest` shape into one arm of this
/// enum so `Snapshot` can arrive on the same connection without a
/// separate message type the read loop would have to guess between.
#[derive(Debug, Serialize, Deserialize)]
pub enum ConversationRequest {
    Turn(ConversationTurnRequest),
    /// Persists this conversation's KV cache to `state_path`/`meta_path`
    /// (see [`SnapshotRef`]) and ends the connection -- the daemon replies
    /// [`ConversationResponse::Snapshotted`] and then closes, the same
    /// way an ordinary client disconnect already ends a conversation,
    /// freeing whatever VRAM its `LlamaContext` held. A later
    /// `OpenConversationRequest` naming the same paths via `restore_from`
    /// picks the conversation back up without replaying it turn by turn.
    Snapshot(SnapshotRef),
}

/// One turn sent into an already-open conversation (see
/// [`ConversationRequest::Turn`]) -- same fields as [`GenerateRequest`]
/// minus `model_path`/`prompt` (the conversation itself already fixes
/// the model; `message` stands in for `prompt`, scoped to just this
/// turn's new text).
#[derive(Debug, Serialize, Deserialize)]
pub struct ConversationTurnRequest {
    /// Executed tool results to feed back, in call order, instead of
    /// `message` -- see `rampipe::llama::Conversation::send_tool_results`
    /// for why a result must arrive as a *result* rather than as
    /// ordinary user text. `message` is ignored when this is `Some`.
    ///
    /// A field on the existing turn request rather than a new
    /// `ConversationRequest` variant: every other parameter here
    /// (`max_new_tokens`, `sampling`, `grammar`, ...) means exactly the
    /// same thing for a result turn as for a message turn, so a second
    /// variant would be this struct again minus one field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_results: Option<Vec<String>>,
    pub message: String,
    /// `None` means "as much as this model is configured for" -- see
    /// `crate::model_settings::Entry::max_new_tokens`.
    ///
    /// Same rule as `sampling` below, and for a sharper reason: a
    /// caller holding this number as a literal is how a model came to
    /// be cut off mid-file, with the fragment written to disk. A caller
    /// that genuinely wants a small answer -- a classifier turn wanting
    /// five tokens -- still says so and still wins.
    #[serde(default)]
    pub max_new_tokens: Option<i32>,
    /// `None` means "use whatever this model is configured for" -- see
    /// `crate::model_settings`.
    ///
    /// Optional rather than required so a caller stops having to hold a
    /// number it read off a model card. It stays overridable because a
    /// caller sometimes genuinely knows better: a classifier turn wants
    /// different sampling from a coding turn against the same model.
    ///
    /// `#[serde(default)]`, so a client built before this existed keeps
    /// working unchanged -- it always sends a value, and a value always
    /// wins.
    #[serde(default)]
    pub sampling: Option<WireSampling>,
    #[serde(default)]
    pub grammar: Option<String>,
    #[serde(default)]
    pub assistant_prefill: Option<String>,
    #[serde(default)]
    pub grammar_completion: Option<GrammarCompletion>,
}

/// One line back per conversation-mode message -- `Opened` acknowledges
/// [`OpenConversationRequest`], `Turn` answers a
/// [`ConversationRequest::Turn`], `Snapshotted` acknowledges a
/// [`ConversationRequest::Snapshot`] (the connection ends right after),
/// `Err` can follow any of the above.
#[derive(Debug, Serialize, Deserialize)]
pub enum ConversationResponse {
    Opened,
    /// Sent in place of [`ConversationResponse::Opened`] when, and only
    /// when, the open request actually carried tools.
    ///
    /// A separate variant rather than a field on `Opened` keeps the
    /// wire backward-compatible in both directions: an old client never
    /// sends tools and so never sees this, while a new client that
    /// sends tools and gets a bare `Opened` back has learned something
    /// real -- it is talking to a daemon built before tool calling
    /// existed, and must fall back rather than wait for tool calls that
    /// can never arrive.
    ///
    /// `supports_tool_calls` is `false` when the tools were rendered
    /// into the prompt but the model's template could not be probed for
    /// how it emits calls (and no `tool_format` fallback was supplied)
    /// -- the model will be told the tools exist and its calls will not
    /// be parseable, which a caller needs to know *before* spending a
    /// generation on it.
    OpenedWithTools {
        supports_tool_calls: bool,
    },
    Turn {
        text: String,
        tokens_generated: usize,
        time_to_first_token_ms: u64,
        /// See `rampipe::conversation::GenerationResult::tool_calls` --
        /// parsed daemon-side, since that is where the model's chat
        /// template is. `#[serde(default)]` so a newer client stays
        /// compatible with a daemon built before this field existed
        /// (which simply never emits tool calls).
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        tool_calls: Vec<ToolCall>,
        /// See `rampipe::conversation::GenerationResult::truncated_tool_call`.
        /// `#[serde(default)]` so an older daemon (which never sets it)
        /// simply reads as "not truncated".
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        truncated_tool_call: bool,
        /// See `GenerateResponse::Ok::formatted_prompt` -- here, just
        /// this turn's own new text (mirrors
        /// `rampipe::llama::Conversation::send`'s `GenerationResult::
        /// formatted_prompt`, which is scoped to the turn, not the whole
        /// conversation).
        formatted_prompt: String,
    },
    Snapshotted,
    Err {
        message: String,
    },
}

/// `~/.rampipe/rampiped.sock` -- the one real source of truth for
/// "where's the daemon," so `rampiped`'s own `main()` and every client
/// (`rampipe::client`, or an external caller like taskpipe) fall back to
/// the same path instead of each hardcoding it independently. Without
/// this, two independently started processes agreeing to share one
/// daemon required a human to type the identical path on both command
/// lines; with it, doing nothing on either side already agrees. `--socket`
/// (`rampiped`) and an explicit client-supplied path still override this,
/// same as before -- this is only ever the fallback when neither says
/// otherwise.
///
/// `None` only when `$HOME` isn't set -- matches `system_free_bytes`'s own
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

    /// `restore_from: None` is the only shape every caller before this
    /// field existed ever sent -- pinning that it's omitted, not
    /// null-serialized, is what keeps a pre-existing wire capture (or a
    /// hand-written request) still valid.
    #[test]
    fn a_fresh_open_request_omits_restore_from_entirely() {
        let request = OpenConversationRequest {
            model_path: PathBuf::from("/models/coder.gguf"),
            n_ctx: 8192,
            overflow: WireOverflowPolicy::DropOldestTurns,
            restore_from: None,
            system: None,
            tools: Vec::new(),
            tool_format: None,
        };
        let encoded = serde_json::to_string(&request).unwrap();
        assert!(!encoded.contains("restore_from"), "{encoded}");
        // Same contract as `restore_from` above, for the same reason:
        // a caller offering neither must produce the exact bytes it
        // did before these fields existed.
        assert!(!encoded.contains("system"), "{encoded}");
        assert!(!encoded.contains("tools"), "{encoded}");
    }

    #[test]
    fn a_restoring_open_request_carries_the_snapshot_paths() {
        let request = OpenConversationRequest {
            model_path: PathBuf::from("/models/coder.gguf"),
            n_ctx: 8192,
            overflow: WireOverflowPolicy::DropOldestTurns,
            restore_from: Some(SnapshotRef {
                state_path: PathBuf::from("/state/coder.state"),
                meta_path: PathBuf::from("/state/coder.meta.json"),
            }),
            system: None,
            tools: Vec::new(),
            tool_format: None,
        };
        let encoded = serde_json::to_string(&request).unwrap();
        assert!(
            encoded.contains(r#""state_path":"/state/coder.state""#),
            "{encoded}"
        );
        assert!(
            encoded.contains(r#""meta_path":"/state/coder.meta.json""#),
            "{encoded}"
        );
    }

    /// `ConversationRequest::Turn` and `::Snapshot` are externally tagged
    /// the same way `ClientMessage`'s own variants are -- pinned so a
    /// change here is a deliberate, visible wire-format decision, not an
    /// incidental derive drift.
    #[test]
    fn a_turn_request_is_tagged_turn() {
        let request = ConversationRequest::Turn(ConversationTurnRequest {
            tool_results: None,
            message: "hi".to_string(),
            max_new_tokens: Some(16),
            sampling: Some(WireSampling::Greedy {
                penalties: WirePenalties::default(),
            }),
            grammar: None,
            assistant_prefill: None,
            grammar_completion: None,
        });
        let encoded = serde_json::to_string(&request).unwrap();
        assert!(encoded.starts_with(r#"{"Turn":"#), "{encoded}");
    }

    #[test]
    fn a_snapshot_request_is_tagged_snapshot() {
        let request = ConversationRequest::Snapshot(SnapshotRef {
            state_path: PathBuf::from("/state/coder.state"),
            meta_path: PathBuf::from("/state/coder.meta.json"),
        });
        let encoded = serde_json::to_string(&request).unwrap();
        assert!(encoded.starts_with(r#"{"Snapshot":"#), "{encoded}");
    }

    #[test]
    fn snapshotted_is_a_bare_string() {
        assert_eq!(
            serde_json::to_string(&ConversationResponse::Snapshotted).unwrap(),
            "\"Snapshotted\""
        );
    }
}
