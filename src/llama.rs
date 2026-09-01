//! Real inference backend behind the `llama` feature -- llama.cpp via
//! `llama-cpp-2`. Kept optional so Phase 1 (`SwapRegistry` alone) stays
//! dependency-light: "v0 measures paging cost with zero backend coupling."
//!
//! llama.cpp always loads models from a file path with its own internal
//! mmap -- there's no API to hand it bytes we've already mapped ourselves.
//! `LlamaSession` bundles a `rampipe` `ModelHandle` (for accounting and
//! eviction safety) with the llama.cpp model loaded from the same path.
//! Both are separate mappings of the same file, so the OS page cache
//! shares the physical pages between them -- prefaulting through our own
//! mapping genuinely warms what llama.cpp will read, it isn't a mapping
//! llama.cpp never touches.

use crate::{ModelHandle, Residency, SwapMetrics, SwapRegistry};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::{LlamaContextParams, LlamaPoolingType};
// Re-exported: `LlamaSession::load` (and, previously, `generate`) took
// `&LlamaBackend` as a parameter, so any caller constructing one (every
// caller, since there's no other way to get one) needs to be able to name
// the type -- without this, that means every caller pinning its own direct
// `llama-cpp-2` dependency just to match whatever version this crate
// happens to use internally, a leaky-abstraction cost `rampipe` itself
// is better positioned to absorb than each of its callers repeating it.
pub use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaModel, RopeType};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use std::ffi::CString;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

// The conversation seam -- `ConversationHandle` and the value types its
// surface is written in -- moved to `crate::conversation` so a build
// without this feature can still name it; see that module for why.
// Re-exported here because every caller (and every doc link) already
// refers to these as `rampipe::llama::*`, and they remain part of this
// module's vocabulary even though their definitions no longer live in it.
pub use crate::conversation::{
    ConversationError, ConversationHandle, GenerationResult, Penalties, Sampling,
};

#[derive(Debug, thiserror::Error)]
pub enum LlamaSessionError {
    #[error("residency registry error: {0}")]
    Registry(#[from] crate::LoadError),
    #[error("llama.cpp model load error: {0}")]
    ModelLoad(#[from] llama_cpp_2::LlamaModelLoadError),
    #[error("llama.cpp context error: {0}")]
    ContextLoad(#[from] llama_cpp_2::LlamaContextLoadError),
    /// What running out of device memory actually looks like.
    ///
    /// llama.cpp has no "out of VRAM" result: a context it cannot
    /// allocate comes back as a null pointer, which `llama_cpp_2` turns
    /// into `LlamaContextLoadError::NullReturn` and this crate used to
    /// pass straight through as `llama.cpp context error: null reference
    /// from llama.cpp`. That message named neither the size asked for
    /// nor the memory available, and the same text is produced by an
    /// exhausted device, an absurd `n_ctx`, and a genuinely broken
    /// model -- three problems with three different answers.
    #[error(
        "could not allocate a {n_ctx}-token context: {source} -- {free_mib} MiB free of \
         {total_mib} MiB on the device (llama.cpp reports an exhausted device as a null \
         pointer, so this is usually what out-of-memory looks like)"
    )]
    ContextAllocation {
        n_ctx: u32,
        free_mib: u64,
        total_mib: u64,
        #[source]
        source: llama_cpp_2::LlamaContextLoadError,
    },
    #[error("llama.cpp decode error: {0}")]
    Decode(#[from] llama_cpp_2::DecodeError),
    #[error("tokenize error: {0}")]
    Tokenize(#[from] llama_cpp_2::StringToTokenError),
    #[error("detokenize error: {0}")]
    Detokenize(#[from] llama_cpp_2::TokenToStringError),
    #[error("batch error: {0}")]
    Batch(#[from] llama_cpp_2::llama_batch::BatchAddError),
    #[error("prompt ({prompt_tokens} tokens) leaves no room in context size ({n_ctx})")]
    PromptTooLong { prompt_tokens: i32, n_ctx: i32 },
    #[error("chat template error: {0}")]
    ChatTemplate(#[from] llama_cpp_2::ChatTemplateError),
    #[error("chat message construction error: {0}")]
    ChatMessage(#[from] llama_cpp_2::NewLlamaChatMessageError),
    #[error("chat template application error: {0}")]
    ApplyChatTemplate(#[from] llama_cpp_2::ApplyChatTemplateError),
    #[error("kv cache operation failed: {0}")]
    KvCache(#[from] llama_cpp_2::context::kv_cache::KvCacheConversionError),
    #[error("model path is not valid UTF-8: {0}")]
    PathNotUtf8(PathBuf),
    #[error("model path contains a NUL byte: {0}")]
    PathHasNul(#[from] std::ffi::NulError),
    #[error("fitting model params to device memory: {0}")]
    Fit(#[from] llama_cpp_2::model::params::FitError),
    #[error("embeddings error: {0}")]
    Embeddings(#[from] llama_cpp_2::EmbeddingsError),
    #[error(
        "model has no chat template this crate can safely reuse turn-by-turn (either no \
         template at all, or its rendering isn't provably steady-state per turn) -- \
         open_conversation needs one; one-shot generate() doesn't"
    )]
    ConversationTemplateUnavailable,
    #[error(
        "conversation turn ({needed} new tokens) doesn't fit even after dropping every droppable turn: committed={committed_pos}, n_ctx={n_ctx}"
    )]
    ConversationContextFull {
        committed_pos: i32,
        needed: i32,
        n_ctx: i32,
    },
    #[error("conversation overflowed its context window but has fewer than 2 turns left to drop")]
    ConversationTooLargeToTrim,
    #[error("grammar error: {0}")]
    Grammar(#[from] llama_cpp_2::GrammarError),
    #[error("saving conversation state: {0}")]
    SaveState(#[from] llama_cpp_2::context::session::SaveSessionError),
    #[error("loading conversation state: {0}")]
    LoadState(#[from] llama_cpp_2::context::session::LoadSessionError),
    #[error("conversation snapshot metadata (de)serialization: {0}")]
    SnapshotMeta(#[from] serde_json::Error),
    #[error("conversation snapshot file I/O: {0}")]
    SnapshotIo(#[from] std::io::Error),
    /// Fail-closed rejection when a saved snapshot's own recorded model
    /// path doesn't match the session it's being reloaded against -- a
    /// state file's KV cache bytes are tied to one specific model's
    /// architecture/quantization/`n_ctx` (llama.cpp itself will reject a
    /// mismatch on shape, but a *different model at the same path*, or
    /// the same model moved to a different path, wouldn't necessarily
    /// trip that check before doing something worse) -- caught here,
    /// before ever calling into llama.cpp's own loader.
    #[error("saved conversation snapshot is for model {saved}, but this session is {current}")]
    SnapshotModelMismatch {
        saved: std::path::PathBuf,
        current: std::path::PathBuf,
    },
}

/// A manual override for how a prompt is wrapped into a chat turn --
/// `prefix + prompt + suffix`, used *instead of* the GGUF's own baked-in
/// chat template. Was the primary fix for a template llama.cpp's own
/// minimal Jinja engine can't render (live case: AI21's Jamba Mini --
/// `apply_chat_template` returns `ffi error -1` on macro/namespace usage
/// llama.cpp's Jinja subset doesn't support); now that
/// `render_with_minijinja` below handles that same template correctly
/// (a real, general Jinja engine, not llama.cpp's limited one), this is
/// a narrower last-resort: a hand-captured wrap for the rare template
/// even `minijinja` can't render, tried only after that's already
/// failed. Deliberately not removed just because no current candidate
/// needs it -- a real, if hopefully rarely-used, escape hatch.
///
/// Single-shot only: this wraps one whole prompt as one atomic block
/// with no separate marker for where a reply ends and a new turn
/// begins, so `Conversation` (which needs to compose turns
/// incrementally) never falls back to it -- see
/// `LlamaSession::open_conversation`'s doc comment.
#[derive(Debug, Clone)]
pub struct ChatWrap {
    pub prefix: String,
    pub suffix: String,
}

/// Renders `template_text` (a model's own real `tokenizer.chat_template`
/// Jinja source, as returned by `chat_template()`) against `messages`,
/// each `(role, content)` in order. `add_generation_prompt` leaves
/// generation open for the assistant to continue into when `true`.
/// `None` on any parse or render failure (unsupported syntax, a template
/// that genuinely calls `raise_exception` for this input shape, etc.).
///
/// `set_trim_blocks`/`set_lstrip_blocks`: real HF chat templates
/// (Jamba's included) are authored assuming `transformers`' own
/// `jinja2.Environment(trim_blocks=True, lstrip_blocks=True)` convention
/// -- without it, the newlines/indentation between `{% %}` control tags
/// that don't carry their own `-` trim markers leak into macro return
/// values and break arithmetic/filters downstream (a real, live failure
/// hit rendering Jamba's own `get_last_user_index` macro before this was
/// set: `|int` failed on a string padded with accumulated block-tag
/// whitespace, not the "0" the macro's actual `{{- ... -}}` content
/// produced).
///
/// `raise_exception`: not a builtin in any Jinja engine -- every real
/// chat-template caller (including `transformers` itself) registers
/// this by convention, since templates call it as an ordinary function
/// for their own input-validation errors (e.g. an unsupported tool
/// type). Registering it here matches that convention rather than
/// leaving the name undefined and turning a template's own deliberate
/// validation error into an unrelated "unknown function" failure.
///
/// `pycompat::unknown_method_callback`: real, live gap this closes --
/// Qwen 3.8's own `tokenizer.chat_template` calls
/// `content.startswith('<tool_response>')`, which Python's own Jinja2
/// silently supports (it falls through to native Python `str` methods
/// for anything its own filter/method table doesn't define); minijinja
/// is a from-scratch Rust reimplementation with no such fallback, so
/// without this the call fails with "unknown method: string has no
/// method named startswith" -- confirmed live, this was the actual
/// reason `open_conversation` rejected that model's template as unusable
/// (surfaced generically as "no template at all, or not provably
/// steady-state," since a render failure and genuine non-determinism
/// both collapse to the same `None` here). `minijinja-contrib`'s own
/// `pycompat` module is exactly the fix its docs recommend for this
/// class of problem -- confirmed live (a standalone probe against this
/// exact template) that registering it alone, with no template changes,
/// makes the render succeed.
use crate::chat_template::render_messages;

/// Single-user-turn convenience wrapper over
/// [`crate::chat_template::render_messages`] -- the one shape
/// `LlamaSession::formatted_prompt` (one-shot `generate()`) needs.
fn render_with_minijinja(template_text: &str, prompt: &str) -> Option<String> {
    render_messages(template_text, &[("user", prompt)], true)
}

/// Silences llama.cpp/ggml's own stderr logging (model-loader tensor
/// dumps, Metal kernel-compile spam, etc.) and works around a real
/// upstream llama.cpp/ggml-metal bug, both discovered live wiring up a
/// caller (brush's `aish` builtin) that talks to this crate in-process:
///
/// - llama.cpp logs straight to the process's real stderr by default,
///   with no hook a caller embedding this crate can filter through its
///   own output -- this replaces ggml's log callback with a no-op.
/// - Separately, its residency-set collection asserts `count == 0` in
///   its own process-exit teardown (`ggml_metal_rsets_free`,
///   `ggml-metal-device.m:656`) -- reliably reproduced on Apple Silicon.
///   Setting `GGML_METAL_NO_RESIDENCY` disables that bookkeeping
///   entirely, avoiding the assert; it's a GPU scheduling hint, not a
///   correctness requirement, and the env var is ggml's own documented
///   escape hatch for it. A no-op on non-macOS builds (the Metal
///   backend doesn't exist there to read it).
///
/// Call this *before* [`LlamaBackend::init`], not after -- Metal device
/// registration (the source of most of the log spam, and the only
/// thing that reads `GGML_METAL_NO_RESIDENCY`) happens during `init()`
/// itself, before a `LlamaBackend` value exists for any per-instance
/// equivalent to act on.
///
/// # Safety
///
/// Mutates the process environment (`GGML_METAL_NO_RESIDENCY`) via
/// `std::env::set_var`, which is only sound if nothing else in the
/// process is concurrently reading or writing the environment. Callers
/// embedding this crate in-process should call this once, early
/// (before spawning any threads that might touch the environment
/// concurrently), the same way `LlamaBackend::init` itself is normally
/// called once at startup.
pub unsafe fn suppress_logs() {
    // Safety: see this function's own `# Safety` section above -- the
    // caller is responsible for the "nothing else touches the
    // environment concurrently" precondition `std::env::set_var` itself
    // requires.
    unsafe {
        std::env::set_var("GGML_METAL_NO_RESIDENCY", "1");
    }

    unsafe extern "C" fn void_log(
        _level: llama_cpp_sys_2::ggml_log_level,
        _text: *const std::os::raw::c_char,
        _user_data: *mut std::os::raw::c_void,
    ) {
    }
    // Safety: `void_log` matches `ggml_log_callback`'s required
    // signature exactly, and a null user-data pointer is valid since
    // `void_log` never dereferences it.
    unsafe {
        llama_cpp_sys_2::llama_log_set(Some(void_log), std::ptr::null_mut());
    }
}

/// A model resident in both `rampipe`'s accounting mmap and llama.cpp's
/// own loaded state -- see module docs for why there are two mappings.
///
/// Owns the `LlamaBackend` it was loaded against (via `Arc`, so several
/// sessions can share the one process-wide backend `LlamaBackend::init`
/// produces) rather than taking it as a parameter to every call -- a
/// caller constructs it once, at startup, the same way it only calls
/// `LlamaBackend::init()` once. `LlamaBackend` has no fields of its own
/// (see `llama_cpp_2::llama_backend::LlamaBackend` -- a bare "proof of
/// init" token), so it's `Send + Sync` automatically and `Arc`-safe to
/// share across the worker threads a real caller (`rampiped`) runs
/// generation from.
/// The in-process half of [`crate::protocol::WirePooling`]. Kept
/// separate for the same reason `Sampling` and `WireSampling` are: this
/// one names a `llama_cpp_2` type and so cannot exist in a build without
/// the `llama` feature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pooling {
    Model,
    Mean,
    Cls,
    Last,
}

impl Pooling {
    /// `None` means "leave `LlamaContextParams` alone", which is how
    /// llama.cpp is told to consult the model's own hparams.
    fn to_llama(self) -> Option<LlamaPoolingType> {
        match self {
            Pooling::Model => None,
            Pooling::Mean => Some(LlamaPoolingType::Mean),
            Pooling::Cls => Some(LlamaPoolingType::Cls),
            Pooling::Last => Some(LlamaPoolingType::Last),
        }
    }
}

/// Scales `v` to unit length in place. A no-op on an all-zero vector
/// rather than producing NaNs -- an empty or fully-truncated input is a
/// caller error worth surfacing as a zero vector (which matches nothing)
/// rather than as poison that silently contaminates every later cosine.
fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Cap on tokens decoded per embedded text, and on the batch allocated
/// to do it. Retrieval encoders are trained at 512 positions and the
/// texts this exists for are short cue phrases, so a cap here costs
/// nothing real and keeps a caller from having a 32K-context model
/// allocate a 32K batch to embed the words "north elevator".
const EMBED_MAX_TOKENS: usize = 512;

pub struct LlamaSession {
    handle: ModelHandle,
    model: LlamaModel,
    backend: Arc<LlamaBackend>,
    /// `None` (the default `load` leaves this) for every model whose own
    /// chat template renders correctly, or has no template at all -- see
    /// `ChatWrap`'s doc comment for when a caller sets this instead, via
    /// `with_chat_wrap`.
    chat_wrap: Option<ChatWrap>,
}

/// Decodes `tokens` onto `ctx`'s KV cache starting at `*n_cur`, in chunks
/// of at most `batch`'s 512-token capacity -- the prompt-prefill half of
/// both `LlamaSession::generate` (starting fresh, `*n_cur == 0`) and
/// `Conversation::send` (continuing from wherever the cache already is).
/// Only the very last token of the whole `tokens` slice requests logits
/// -- that's the one `run_generation_loop`'s first sample actually reads;
/// a caller that isn't about to sample right after (`Conversation`
/// decoding a turn-transition's trailing text with nothing to sample
/// yet) still behaves correctly, since unread logits are simply never
/// read.
fn decode_chunked(
    ctx: &mut LlamaContext,
    batch: &mut LlamaBatch,
    tokens: &[LlamaToken],
    n_cur: &mut i32,
) -> Result<(), LlamaSessionError> {
    if tokens.is_empty() {
        return Ok(());
    }
    let last_index = tokens.len() - 1;
    for chunk_start in (0..tokens.len()).step_by(512) {
        let chunk_end = (chunk_start + 512).min(tokens.len());
        batch.clear();
        for (offset, &token) in tokens[chunk_start..chunk_end].iter().enumerate() {
            let pos = *n_cur + (chunk_start + offset) as i32;
            let is_last = chunk_start + offset == last_index;
            batch.add(token, pos, &[0], is_last)?;
        }
        ctx.decode(batch)?;
    }
    *n_cur += tokens.len() as i32;
    Ok(())
}

/// The actual token-by-token sampling loop -- shared by `generate()`
/// (fresh context, `*n_cur` starting at the prompt's own length) and
/// `Conversation::send` (persistent context, `*n_cur` starting wherever
/// the cache already is after this turn's prefill). Requires `batch` to
/// be the same batch whose last `decode` call is what populated the
/// logits this loop's first `sample` reads -- see `decode_chunked`'s doc
/// comment; the two are always called back-to-back on one shared batch
/// by both callers, exactly as this one was before being split out of
/// `generate()`.
#[allow(clippy::too_many_arguments)]
fn run_generation_loop(
    ctx: &mut LlamaContext,
    model: &LlamaModel,
    batch: &mut LlamaBatch,
    n_cur: &mut i32,
    n_ctx: i32,
    max_new_tokens: i32,
    sampling: Sampling,
    grammar: Option<&str>,
    grammar_complete: Option<&dyn Fn(&str) -> bool>,
    // Ends the turn when the model stops calling tools and starts
    // writing the harness's half of the conversation. See
    // `crate::tool_format::TurnEnd` for the run this cost.
    turn_end: Option<&crate::tool_format::TurnEnd>,
    start: Instant,
) -> Result<(String, Vec<LlamaToken>, Duration), LlamaSessionError> {
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    // `Greedy` chains straight to `greedy()` -- no `dist()` in front of
    // it, since a sampler chain's *last* stage picks the token, and
    // `greedy()` always overwrites whatever came before with pure
    // argmax (confirmed by source-tracing `llama_cpp_2`'s
    // `dist_apply`/`greedy_apply`). `Temperature` filters to the
    // `top_k` highest-probability candidates, reshapes the
    // distribution by `temperature`, then samples from it via
    // `dist(seed)` as the actual final stage -- no `greedy()` after it,
    // so the sampled draw is what's actually used. When `grammar` is
    // `Some`, its stage goes *first* in the chain -- a chain applies
    // its stages in order, so the grammar must mask the logits down to
    // grammar-valid tokens before top_k/temp/dist (or greedy) ever see
    // them, not after a token's already been picked.
    let build_chain = |grammar: Option<&str>| -> Result<LlamaSampler, LlamaSessionError> {
        let mut stages: Vec<LlamaSampler> = Vec::new();
        if let Some(grammar_str) = grammar {
            stages.push(LlamaSampler::grammar(model, grammar_str, "root")?);
        }
        // A penalties stage, when its own fields aren't all "disabled",
        // goes before the final-selection stage regardless of which
        // `Sampling` variant is in play -- it adjusts logits, so it has
        // to run before greedy argmax or temperature/dist ever sees
        // them, the same ordering reasoning the grammar stage above
        // already follows.
        let push_penalties = |stages: &mut Vec<LlamaSampler>, penalties: Penalties| {
            if penalties.repeat != 1.0 || penalties.freq != 0.0 || penalties.present != 0.0 {
                stages.push(LlamaSampler::penalties(
                    penalties.last_n,
                    penalties.repeat,
                    penalties.freq,
                    penalties.present,
                ));
            }
        };
        match sampling {
            Sampling::Greedy { penalties } => {
                push_penalties(&mut stages, penalties);
                stages.push(LlamaSampler::greedy());
            }
            Sampling::Temperature {
                temperature,
                top_k,
                top_p,
                min_p,
                seed,
                penalties,
            } => {
                push_penalties(&mut stages, penalties);
                stages.push(LlamaSampler::top_k(top_k));
                // top_k, then top_p, then min_p, then temp, then dist --
                // llama.cpp's own default chain order. Each is a filter
                // over the candidate set and the order decides what each
                // one sees; a disabled stage is skipped rather than
                // pushed with a no-op value, so a chain built from a
                // model's card carries only what the card asked for.
                if top_p < 1.0 {
                    stages.push(LlamaSampler::top_p(top_p, 1));
                }
                if min_p > 0.0 {
                    stages.push(LlamaSampler::min_p(min_p, 1));
                }
                stages.push(LlamaSampler::temp(temperature));
                stages.push(LlamaSampler::dist(seed));
            }
        }
        Ok(LlamaSampler::chain_simple(stages))
    };

    // A grammar-constrained chain's `sample()` reliably crashes (a hard
    // process abort inside llama.cpp's own grammar internals --
    // `GGML_ASSERT(!stacks.empty())` inside `llama_grammar_reject_
    // candidates`, not a recoverable Rust `Err`) on the *second*
    // `sample()` call made against one grammar sampler instance --
    // reproduced live against even the simplest possible multi-token
    // grammar (`root ::= "AB"`, two ordinary letters, no alternation, no
    // repetition, nowhere near a completed match), so this isn't
    // specific to this crate's own grammars or to reaching the end of
    // one. The one call shape that *is* reliable: a freshly constructed
    // grammar sampler's very first `sample()`. So for a grammar-
    // constrained chain, `run_generation_loop` rebuilds the whole chain
    // from scratch before every token and replays every token accepted
    // so far into the fresh instance via `accept()` (cheap -- grammar
    // state advancement is in-memory bookkeeping, not model inference)
    // before sampling once and discarding it. More total work than one
    // persistent chain, but the only shape found that survives a
    // multi-token grammar-constrained response without crashing the
    // process. `plain_sampler` is the ordinary persistent-chain path,
    // unchanged, used whenever there's no grammar to work around.
    let mut plain_sampler = if grammar.is_none() {
        Some(build_chain(None)?)
    } else {
        None
    };
    let mut accepted_tokens: Vec<LlamaToken> = Vec::new();

    let mut text = String::new();
    // Every token actually fed back into the KV cache via `ctx.decode`
    // below, in order -- including the thinking-mode self-feed branch's
    // own EOG token, which never touches `text` at all. Needs to be
    // exactly what's physically in the cache, not just what counted as
    // "real" generated content, so a caller can later hand this to
    // `LlamaContext::state_save_file` (which pairs a saved KV cache with
    // the token sequence that produced it) without the two silently
    // disagreeing.
    let mut generated_tokens: Vec<LlamaToken> = Vec::new();
    let mut time_to_first_token = None;
    // Thinking-mode models (Qwen3.6, DeepSeek-R1-style) spend an
    // unpredictable, sometimes large chunk of generation on
    // `<think>...</think>` deliberation before the real answer even
    // starts. A fixed shared budget can run out mid-thought, before
    // any answer exists at all (real case: Qwen3.6-35B-A3B, piper
    // task 1 attempt 1 -- an 8,879-char response entirely inside an
    // unclosed `<think>`, cut off with no answer to extract; see
    // `taskpipe::backend::strip_thinking_block`, which is what turns
    // that case into a clean retry instead of a silent bad extract).
    //
    // `budget_used` only increments for tokens generated *outside* an
    // open, unclosed `<think>` block -- so deliberation is metered
    // against `n_ctx` alone (the hard physical ceiling -- the KV cache
    // literally cannot hold more), not against `max_new_tokens`, and
    // `max_new_tokens` ends up meaning exactly what it says: a budget
    // for the answer, not for the answer *and* however much thinking
    // happened to come first. A response with no `<think>` at all
    // (most models, including the current default) never has an open
    // block, so `budget_used` increments every token from the very
    // first one -- behavior is unchanged for those.
    let mut budget_used: i32 = 0;

    loop {
        if *n_cur >= n_ctx || budget_used >= max_new_tokens {
            break;
        }

        let token = match &mut plain_sampler {
            Some(sampler) => {
                let token = sampler.sample(ctx, batch.n_tokens() - 1);
                sampler.accept(token);
                token
            }
            None => {
                let mut sampler = build_chain(grammar)?;
                for &prior in &accepted_tokens {
                    sampler.accept(prior);
                }
                let token = sampler.sample(ctx, batch.n_tokens() - 1);
                sampler.accept(token);
                accepted_tokens.push(token);
                token
            }
        };

        if time_to_first_token.is_none() {
            time_to_first_token = Some(start.elapsed());
        }

        let inside_open_think = text.contains("<think>") && !text.contains("</think>");

        if model.is_eog_token(token) {
            if !inside_open_think {
                break;
            }
            // The model tried to end its turn while still "supposed
            // to be" thinking -- honoring that would leave a response
            // with deliberation but no answer at all, exactly the
            // failure this whole mechanism exists to avoid. Not
            // fabricating a substitute token and not just `continue`ing
            // either -- `llama-cpp-2` has no clean way to ban a token
            // mid-chain, and resampling from unchanged logits would
            // deterministically reselect the same EOG token forever
            // under greedy sampling. Feeding it back like an ordinary
            // token instead genuinely advances the KV cache, so the
            // *next* sample is conditioned on the model having "seen"
            // its own attempted stop -- out-of-distribution for what it
            // was trained on, but self-limiting (still bounded by
            // `n_cur < n_ctx` above) and never an infinite loop.
            batch.clear();
            batch.add(token, *n_cur, &[0], true)?;
            *n_cur += 1;
            ctx.decode(batch)?;
            generated_tokens.push(token);
            continue;
        }

        text.push_str(&model.token_to_piece(token, &mut decoder, true, None)?);

        if !inside_open_think {
            budget_used += 1;
        }

        // Grammar-constrained generation has a real, reproducible crash
        // (a hard process abort -- `GGML_ASSERT(!stacks.empty())` inside
        // llama.cpp's own `llama_grammar_reject_candidates`, not a
        // recoverable Rust error) if `sample()` is called again once the
        // grammar's `root` rule has already fully matched and the model
        // doesn't itself pick an end-of-generation token on the very
        // next draw. Reproduced live against even the simplest possible
        // closed grammar (`root ::= "YES"`, no alternation, nothing
        // envelope-specific) -- not a bug in this crate's own grammars.
        // `grammar_complete`, when the caller supplies one, is this
        // crate's own completion signal in place of relying on
        // llama.cpp's grammar-driven EOG forcing: check it against the
        // accumulated text and stop *before* ever sampling again, rather
        // than after.
        if let Some(is_complete) = grammar_complete
            && is_complete(&text)
        {
            break;
        }

        // The model has finished calling tools and moved on to inventing
        // their results. Everything from the last closer onward is
        // discarded, and the turn ends here rather than at the token cap.
        //
        // Checked on the accumulated text rather than the token, because
        // a closer is not a token: `}<tool_call|>` arrives in pieces and
        // may land alongside whatever follows it in the same piece.
        if let Some(cut) = turn_end.and_then(|end| end.reached(&text)) {
            text.truncate(cut);
            break;
        }

        batch.clear();
        batch.add(token, *n_cur, &[0], true)?;
        *n_cur += 1;
        ctx.decode(batch)?;
        generated_tokens.push(token);
    }

    Ok((
        text,
        generated_tokens,
        time_to_first_token.unwrap_or_default(),
    ))
}

/// How many of a model's transformer layers to put on GPU. A fixed count
/// baked into per-model config doesn't work for `rampiped`'s actual job:
/// `SwapRegistry` keeps several models resident under one shared memory
/// budget, so how many layers of *this* model fit depends on what else is
/// already resident on the GPU right now, not on the model alone.
#[derive(Debug, Clone, Copy, Default)]
pub enum GpuLayers {
    /// Ask llama.cpp's own `common_fit_params` to decide, from actual free
    /// device memory at load time (works for CUDA, Metal, or CPU-only --
    /// it queries whatever backend device is present). The default.
    #[default]
    Auto,
    /// Force exactly this many layers onto GPU, skipping the fit step --
    /// escape hatch for when the auto estimate is wrong for a given model.
    Fixed(u32),
}

/// Per-device memory margin `fit_params` leaves unused, and the minimum
/// context size it tries to preserve when shrinking allocations to fit --
/// same values llama.cpp's own CLI defaults to (`common/common.h`), kept
/// in step rather than picked fresh, since fitting is inherently a "guess
/// well" problem, not one this crate has a better answer to than upstream.
const GPU_FIT_MARGIN_BYTES: usize = 1024 * 1024 * 1024;
const GPU_FIT_MIN_CTX: u32 = 4096;

/// Resolves `gpu_layers` into ready-to-load `LlamaModelParams`. Returned
/// pinned because a successful `fit_params` call leaves raw pointers
/// inside `LlamaModelParams` pointing at its own `tensor_split`/
/// `tensor_buft_overrides` buffers -- moving the value afterward would
/// invalidate them, so it must stay behind `Pin` all the way through the
/// `load_from_file` call that actually reads it.
fn resolve_model_params(
    path: &Path,
    gpu_layers: GpuLayers,
) -> Result<Pin<Box<LlamaModelParams>>, LlamaSessionError> {
    match gpu_layers {
        GpuLayers::Fixed(n) => Ok(Box::pin(LlamaModelParams::default().with_n_gpu_layers(n))),
        GpuLayers::Auto => {
            let path_str = path
                .to_str()
                .ok_or_else(|| LlamaSessionError::PathNotUtf8(path.to_path_buf()))?;
            let path_c = CString::new(path_str)?;
            let mut params = Box::pin(LlamaModelParams::default());
            // n_ctx=0 (via `with_n_ctx(None)`) tells `fit_params` to pick
            // its own context size, floored at `GPU_FIT_MIN_CTX` -- using
            // this crate's own default of 512 here would let it offload
            // more layers than actually fit once a real conversation later
            // opens a much larger context.
            let mut cparams = LlamaContextParams::default().with_n_ctx(None);
            let mut margins =
                vec![GPU_FIT_MARGIN_BYTES; unsafe { llama_cpp_sys_2::llama_max_devices() }];
            match params.as_mut().fit_params(
                &path_c,
                &mut cparams,
                &mut margins,
                GPU_FIT_MIN_CTX,
                llama_cpp_sys_2::GGML_LOG_LEVEL_ERROR,
            ) {
                Ok(_) => Ok(params),
                // No allocation was projected to fit within the memory
                // margin -- fall back to CPU-only rather than failing the
                // whole load; matches `common_fit_params`'s own "assumes
                // system memory is unlimited" contract for the non-GPU
                // side.
                Err(llama_cpp_2::model::params::FitError::Failure) => {
                    Ok(Box::pin(LlamaModelParams::default().with_n_gpu_layers(0)))
                }
                Err(err) => Err(LlamaSessionError::Fit(err)),
            }
        }
    }
}

/// Free/total VRAM in bytes, per whichever GPU backend device is actually
/// present -- `ggml_backend_dev_by_type`/`ggml_backend_dev_memory` are the
/// same backend-agnostic ggml calls `resolve_model_params`'s `fit_params`
/// path relies on internally, so this reports exactly what auto-fitting
/// itself sees, not a separate/inconsistent measurement. `None` if no GPU
/// device is registered (CPU-only build, or none found at runtime).
pub fn gpu_memory_bytes() -> Option<(u64, u64)> {
    // Safety: `ggml_backend_dev_by_type` returns a null pointer (not a
    // dangling one) when no device of that type is registered, checked
    // below before the device handle is used for anything.
    let device = unsafe {
        llama_cpp_sys_2::ggml_backend_dev_by_type(llama_cpp_sys_2::GGML_BACKEND_DEVICE_TYPE_GPU)
    };
    if device.is_null() {
        return None;
    }
    let mut free: usize = 0;
    let mut total: usize = 0;
    // Safety: `device` was just checked non-null; `ggml_backend_dev_memory`
    // writes exactly these two `usize` out-params, both valid local
    // `&mut` targets.
    unsafe { llama_cpp_sys_2::ggml_backend_dev_memory(device, &raw mut free, &raw mut total) };
    Some((free as u64, total as u64))
}

impl LlamaSession {
    /// Loads `path` into `registry` (for residency accounting/eviction
    /// safety) and into llama.cpp (for actual inference), against
    /// `backend` -- shared via `Arc` so several sessions (and any
    /// `Conversation`s opened from them) can outlive a single call
    /// without each needing `backend` handed to them again.
    ///
    /// GPU offload is decided automatically (see [`GpuLayers::Auto`]) --
    /// use [`Self::load_with_gpu_layers`] to force a specific layer count
    /// instead.
    pub fn load(
        registry: &SwapRegistry,
        backend: Arc<LlamaBackend>,
        path: impl AsRef<Path>,
        residency: Residency,
    ) -> Result<Self, LlamaSessionError> {
        Self::load_with_gpu_layers(registry, backend, path, residency, GpuLayers::Auto)
    }

    /// Same as [`Self::load`], but with explicit control over how many
    /// layers to offload to GPU instead of always auto-fitting. See
    /// [`GpuLayers`].
    pub fn load_with_gpu_layers(
        registry: &SwapRegistry,
        backend: Arc<LlamaBackend>,
        path: impl AsRef<Path>,
        residency: Residency,
        gpu_layers: GpuLayers,
    ) -> Result<Self, LlamaSessionError> {
        let path = path.as_ref();
        let handle = registry.load(path, residency)?;
        let params = resolve_model_params(path, gpu_layers)?;
        // Free-device-bytes immediately before/after the one call that
        // actually touches the GPU -- the delta is this model's real GPU
        // footprint, same before/after-measurement shape
        // `SwapMetrics::rss_delta_bytes` already uses for host memory.
        // Only reported to the registry on success: a failed load never
        // makes it into `SharedState.sessions` (see `ensure_loaded`), so
        // there'd be nothing meaningful to attribute the measurement to.
        let gpu_free_before = gpu_memory_bytes().map(|(free, _)| free);
        let model = LlamaModel::load_from_file(&backend, path, &params)?;
        if let Some(free_before) = gpu_free_before
            && let Some((free_after, _)) = gpu_memory_bytes()
        {
            registry.record_device_bytes(handle.id(), free_before.saturating_sub(free_after));
        }
        Ok(Self {
            handle,
            model,
            backend,
            chat_wrap: None,
        })
    }

    /// Opts this session into a manual `ChatWrap` instead of the GGUF's
    /// own baked-in template -- see `ChatWrap`'s doc comment for why.
    /// Builder-style (consumes and returns `Self`) rather than a
    /// `&mut self` setter, so a caller can chain it directly onto `load`
    /// without an extra `let mut` binding.
    pub fn with_chat_wrap(mut self, chat_wrap: ChatWrap) -> Self {
        self.chat_wrap = Some(chat_wrap);
        self
    }

    /// Residency metrics from the `rampipe` side of this session (map
    /// latency, prefault latency, RSS delta, mapped bytes, warm flag).
    pub fn metrics(&self) -> SwapMetrics {
        self.handle.metrics()
    }

    pub fn path(&self) -> &Path {
        self.handle.path()
    }

    /// The underlying registry handle's id -- what a caller managing
    /// several resident sessions at once (`rampiped`) needs to match a
    /// session back to `SwapRegistry::resident_ids_by_lru()`'s output
    /// when deciding what to evict.
    pub fn id(&self) -> crate::ModelId {
        self.handle.id()
    }

    /// `LlamaModel::new_context`, with an out-of-memory failure that
    /// says so.
    ///
    /// Every context in this file goes through here rather than calling
    /// `new_context` directly, because the failure being translated is
    /// indistinguishable at the call site -- see
    /// [`LlamaSessionError::ContextAllocation`]. `asked` is passed in
    /// rather than read back off `params`, so the number in the message
    /// is the one the caller intended, including where that is zero
    /// meaning the model's own trained size.
    fn new_context_reporting(
        &self,
        params: LlamaContextParams,
        asked: u32,
    ) -> Result<LlamaContext<'_>, LlamaSessionError> {
        match self.model.new_context(&self.backend, params) {
            Ok(context) => Ok(context),
            // With no device to report on there is nothing to add, and
            // the plain error is still the honest one.
            Err(source) => Err(match gpu_memory_bytes() {
                Some((free, total)) => LlamaSessionError::ContextAllocation {
                    n_ctx: asked,
                    free_mib: free / (1024 * 1024),
                    total_mib: total / (1024 * 1024),
                    source,
                },
                None => LlamaSessionError::ContextLoad(source),
            }),
        }
    }

    /// GPU/device memory this session's model is using, if any -- see
    /// `ModelHandle::device_bytes`. `None` for a model that ran CPU-only.
    pub fn device_bytes(&self) -> Option<u64> {
        self.handle.device_bytes()
    }

    /// Embeds each of `texts` into one vector, using a fresh
    /// embeddings-enabled context for the whole batch.
    ///
    /// A separate context from the generation path on purpose:
    /// `with_embeddings(true)` changes what llama.cpp extracts from a
    /// decode, and the same model can legitimately be asked for both, so
    /// neither call gets to mutate the other's context configuration.
    /// The model itself is shared -- residency is per-model, and this
    /// costs no extra weights.
    ///
    /// Every token is added with logits enabled, matching llama.cpp's
    /// own `examples/embedding` (`batch_add_seq` passes `true` for every
    /// token): pooling reads across the whole sequence, not just the
    /// last position.
    pub fn embed(
        &self,
        texts: &[String],
        pooling: Pooling,
        normalize: bool,
    ) -> Result<Vec<Vec<f32>>, LlamaSessionError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        match self.embed_pooled(texts, pooling, normalize) {
            // The model declared no pooling type of its own and llama.cpp
            // resolved the unspecified request to NONE, which makes
            // `llama_get_embeddings_seq` return null. Mean is the
            // sensible fallback and the alternative is handing the caller
            // an error it has no way to act on.
            Err(LlamaSessionError::Embeddings(llama_cpp_2::EmbeddingsError::NonePoolType))
                if pooling == Pooling::Model =>
            {
                self.embed_pooled(texts, Pooling::Mean, normalize)
            }
            other => other,
        }
    }

    fn embed_pooled(
        &self,
        texts: &[String],
        pooling: Pooling,
        normalize: bool,
    ) -> Result<Vec<Vec<f32>>, LlamaSessionError> {
        // `with_n_ctx(None)` -> n_ctx=0 -> llama.cpp uses the model's own
        // training context, which for a retrieval encoder is the only
        // correct answer (bge-small is trained at 512; asking for 8192
        // would be meaningless).
        let mut ctx_params = LlamaContextParams::default()
            .with_n_ctx(None)
            .with_embeddings(true);
        if let Some(pooling_type) = pooling.to_llama() {
            ctx_params = ctx_params.with_pooling_type(pooling_type);
        }
        let mut ctx = self.new_context_reporting(ctx_params, 0)?;

        let capacity = (ctx.n_ctx() as usize).min(EMBED_MAX_TOKENS).max(1);
        let mut batch = LlamaBatch::new(capacity, 1);
        let mut out = Vec::with_capacity(texts.len());

        for text in texts {
            let mut tokens = self.model.str_to_token(text, AddBos::Always)?;
            tokens.truncate(capacity);
            batch.clear();
            for (pos, &token) in tokens.iter().enumerate() {
                batch.add(token, pos as i32, &[0], true)?;
            }
            // Each text is independent -- without this, sequence 0 keeps
            // accumulating and every vector after the first pools over
            // its own text *plus* every text before it.
            ctx.clear_kv_cache();
            ctx.decode(&mut batch)?;
            let mut vector = ctx.embeddings_seq_ith(0)?.to_vec();
            if normalize {
                l2_normalize(&mut vector);
            }
            out.push(vector);
        }
        Ok(out)
    }

    /// Runs a real generation, using a fresh context each call -- no
    /// state (KV cache, turn history) survives past this one call. See
    /// `open_conversation` for a session that keeps its KV cache alive
    /// across several calls instead of re-prefilling from scratch every
    /// time.
    pub fn generate(
        &self,
        prompt: &str,
        max_new_tokens: i32,
        sampling: Sampling,
        grammar: Option<&str>,
        assistant_prefill: Option<&str>,
        grammar_complete: Option<&dyn Fn(&str) -> bool>,
    ) -> Result<GenerationResult, LlamaSessionError> {
        let start = Instant::now();

        // Raised 2048 -> 4096 -> 8192, twice now for the same underlying
        // reason: `describe_dependency`'s API-summary budget
        // (`MAX_SUMMARY_CHARS`) is tuned to fit a annotated method list,
        // not any specific crate's real size -- a verbose crate's summary
        // (real case: `ratatui`, a real taskpipe run against piper task 3)
        // measured the full prompt at 5060 tokens, already over 4096
        // before any generation headroom, where `ropey`'s summary fit
        // comfortably. Qwen2.5-7B-Instruct's native trained context is
        // 32K, so 8192 still isn't a context-extension trick, just using
        // more of what the model already supports -- costs roughly double
        // the KV cache again (~224MiB -> ~448MiB at this model size),
        // still negligible next to the ~4.4GiB model weights.
        let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(8192));
        let mut ctx = self.new_context_reporting(ctx_params, 8192)?;

        let mut formatted_prompt = self.formatted_prompt(prompt)?;
        // Prefilling the assistant turn: `formatted_prompt` already ends
        // with the assistant's own turn opened (see this method's doc
        // comment), so appending here before tokenizing makes generation
        // resume *inside* `assistant_prefill` rather than at the start of
        // a fresh turn -- e.g. seeding "{" so a grammar-constrained JSON
        // response never has to open its own object. Prepended back onto
        // `text` below so the caller sees one complete, self-consistent
        // string, as if the model had generated it from scratch.
        if let Some(prefill) = assistant_prefill {
            formatted_prompt.push_str(prefill);
        }
        let tokens_list = self.model.str_to_token(&formatted_prompt, AddBos::Always)?;
        let n_ctx = ctx.n_ctx() as i32;
        // Not `+ max_new_tokens` any more: `max_new_tokens` no longer
        // bounds the whole generation, only the metered (non-thinking)
        // portion of it -- see `run_generation_loop`. The only thing that
        // has to fit up front is the prompt itself; there being at least
        // one token of room left is checked implicitly by the loop
        // condition (`n_cur < n_ctx`), so an oversized prompt just
        // generates zero tokens rather than erroring here. A prompt
        // that's already at or past `n_ctx` is the one real failure
        // worth surfacing early.
        if tokens_list.len() as i32 >= n_ctx {
            return Err(LlamaSessionError::PromptTooLong {
                prompt_tokens: tokens_list.len() as i32,
                n_ctx,
            });
        }

        let mut n_cur = 0i32;
        let mut batch = LlamaBatch::new(512, 1);
        decode_chunked(&mut ctx, &mut batch, &tokens_list, &mut n_cur)?;
        // No conversation, so no derived format to end a turn with.
        // This path is one-shot completion, not the agent loop.
        let turn_end: Option<crate::tool_format::TurnEnd> = None;
        let (text, generated_tokens, time_to_first_token) = run_generation_loop(
            &mut ctx,
            &self.model,
            &mut batch,
            &mut n_cur,
            n_ctx,
            max_new_tokens,
            sampling,
            grammar,
            grammar_complete,
            turn_end.as_ref(),
            start,
        )?;
        let tokens_generated = generated_tokens.len();
        let text = match assistant_prefill {
            Some(prefill) => format!("{prefill}{text}"),
            None => text,
        };

        Ok(GenerationResult {
            // A one-shot generate holds only this prompt and its reply.
            committed_tokens: tokens_list.len() + tokens_generated,
            context_size: n_ctx.max(0) as usize,
            text,
            time_to_first_token,
            tokens_generated,
            formatted_prompt,
            // A one-shot `generate()` has no conversation, and so no
            // tool list and no derived format -- tools are a
            // `Conversation`-level arrangement (see
            // `ConversationOptions::tools`).
            tool_calls: Vec::new(),
            truncated_tool_call: false,
        })
    }

    /// Opens a real multi-turn conversation: one `LlamaContext` (and its
    /// KV cache) that stays alive across every `Conversation::send` call
    /// on the value returned here, instead of `generate()`'s
    /// fresh-context-every-call shape. Each `send()` only tokenizes and
    /// decodes that turn's own new text -- everything earlier in the
    /// conversation is already sitting in the cache from a prior call.
    ///
    /// `defrag_thold` is set to [`CONVERSATION_DEFRAG_THOLD`], not left
    /// at llama.cpp's own default of `-1.0` (auto-defrag disabled) --
    /// see that constant's own doc comment for why a conversation
    /// context specifically needs it on.
    ///
    /// Requires a chat template this crate can prove renders each turn
    /// the same way regardless of how many turns came before it (see
    /// `derive_conversation_template`) -- `ChatWrap` (a single-shot,
    /// whole-prompt override) can't stand in for that the way it can
    /// for one-shot `generate()`, since it has no notion of "where one
    /// turn ends and the next begins." A model whose template can't be
    /// proven steady-state this way can still be used via `generate()`,
    /// just not `open_conversation`.
    pub fn open_conversation(
        &self,
        options: ConversationOptions,
    ) -> Result<Conversation<'_>, LlamaSessionError> {
        // Real, live-observed crash this guards against: a model using
        // some multi-position rope scheme hit `GGML_ASSERT(
        // hparams.n_pos_per_embd() == 1 && "seq_add() is only supported
        // for n_pos_per_embd() == 1")` and aborted the whole process --
        // confirmed live against the real model, this fires from
        // *either* `drop_oldest_turns_for`'s own `kv_cache_seq_add` call
        // on overflow, or llama.cpp's own internal auto-defrag doing the
        // same position-shift under the hood, so both have to be
        // disabled together, not just one.
        //
        // Allowlist, not a denylist, and deliberately so -- confirmed
        // live (reading llama.cpp's own `llama_model_rope_type()`
        // switch in `llama-model.cpp`) that the real model this crash
        // was found against reports `LLAMA_ROPE_TYPE_IMROPE`
        // ("interleaved M-RoPE", a newer constant llama.cpp added and
        // groups explicitly alongside its vision architectures) -- a
        // value `rope_type()`'s own match arms here don't have a case
        // for, so it silently fell through to `None`. A denylist keyed
        // on `Some(MRope) | Some(Vision)` would have missed this exact
        // case entirely and let the crash back in. Only the two rope
        // kinds confirmed ordinary, single-position-per-token text
        // models use (`Norm`, `NeoX`) are trusted with the risky path;
        // everything else -- `None`, `MRope`, `Vision`, or any future
        // rope kind this binding doesn't recognize yet -- plays it safe.
        // That trades away "auto-trim/auto-compact a long conversation"
        // for "never crash the whole daemon out from under every other
        // session on it, including for a rope kind added to llama.cpp
        // after this binding was last updated," which is not a close
        // call.
        let supports_kv_cache_position_shift = matches!(
            self.model.rope_type(),
            Some(RopeType::Norm) | Some(RopeType::NeoX)
        );
        let (overflow, defrag_thold) = if supports_kv_cache_position_shift {
            (options.overflow, CONVERSATION_DEFRAG_THOLD)
        } else {
            // llama.cpp's own `llama_context_default_params` value for
            // "defrag disabled entirely" -- see `CONVERSATION_DEFRAG_THOLD`'s
            // own doc comment.
            (OverflowPolicy::Fail, -1.0)
        };

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(options.n_ctx))
            .with_defrag_thold(defrag_thold);
        let ctx = self.new_context_reporting(ctx_params, options.n_ctx.get())?;
        let n_ctx = ctx.n_ctx() as i32;

        let template_text = self
            .model
            .chat_template(None)
            .ok()
            .and_then(|template| template.to_str().ok().map(str::to_string))
            .ok_or(LlamaSessionError::ConversationTemplateUnavailable)?;
        let template = derive_conversation_template(&template_text)
            .ok_or(LlamaSessionError::ConversationTemplateUnavailable)?;

        let ToolSetup {
            opening,
            system_for_first_turn,
            tool_format,
            tool_result_spans,
            capabilities,
        } = prepare_tools(
            &template_text,
            &template,
            options.system.as_deref(),
            &options.tools,
            options.tool_format.as_ref(),
        );

        Ok(Conversation {
            ctx,
            model: &self.model,
            _handle: self.handle.clone(),
            template,
            n_ctx,
            committed_pos: 0,
            turns: Vec::new(),
            overflow,
            tokens: Vec::new(),
            opening,
            protected_prefix: 0,
            system_for_first_turn,
            tool_format,
            tool_result_spans,
            capabilities,
            tools: options.tools,
            configured_tool_format: options.tool_format,
        })
    }

    /// The reverse of [`Conversation::save_state`] -- reopens a
    /// conversation with its KV cache already filled from `state_path`
    /// (llama.cpp's own state file) instead of starting cold, restoring
    /// the turn/role bookkeeping from the `meta_path` sidecar that
    /// accompanies it. `n_ctx` and `overflow` come from the snapshot
    /// itself, not a caller-supplied `ConversationOptions` -- the saved
    /// state's byte layout is already tied to the `n_ctx` it was saved
    /// with, so re-deriving it from the snapshot (rather than trusting a
    /// caller to pass a matching one) is what turns a possible mismatch
    /// into an explicit, named error here instead of a confusing failure
    /// deeper inside llama.cpp's own loader.
    ///
    /// Fails closed (`SnapshotModelMismatch`) if the snapshot's own
    /// recorded model path doesn't match this session's -- see that
    /// error variant's own doc comment for why that check happens here,
    /// before ever handing raw bytes to llama.cpp.
    pub fn open_conversation_from_state(
        &self,
        state_path: impl AsRef<Path>,
        meta_path: impl AsRef<Path>,
    ) -> Result<Conversation<'_>, LlamaSessionError> {
        let meta_bytes = std::fs::read(meta_path)?;
        let meta: ConversationSnapshotMeta = serde_json::from_slice(&meta_bytes)?;

        let current_path = self.path().to_path_buf();
        if meta.model_path != current_path {
            return Err(LlamaSessionError::SnapshotModelMismatch {
                saved: meta.model_path,
                current: current_path,
            });
        }

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(meta.n_ctx as u32))
            .with_defrag_thold(CONVERSATION_DEFRAG_THOLD);
        let mut ctx = self.new_context_reporting(ctx_params, meta.n_ctx as u32)?;
        let n_ctx = ctx.n_ctx() as i32;

        let template_text = self
            .model
            .chat_template(None)
            .ok()
            .and_then(|template| template.to_str().ok().map(str::to_string))
            .ok_or(LlamaSessionError::ConversationTemplateUnavailable)?;
        let template = derive_conversation_template(&template_text)
            .ok_or(LlamaSessionError::ConversationTemplateUnavailable)?;

        // `n_ctx` (the real, freshly-queried context size) rather than
        // `meta.n_ctx` as the upper bound: they should always agree, but
        // this is what llama.cpp itself is actually about to fill.
        let tokens = ctx.state_load_file(state_path, n_ctx as usize)?;
        let committed_pos = tokens.len() as i32;

        // A restored conversation's system block and tool list are
        // already *in* the KV cache being reloaded -- they were decoded
        // into the opening when it was first created, and re-rendering
        // them here would either duplicate them or, worse, disagree with
        // what is physically resident. So no system text and no tool
        // list is passed: `opening` and `system_for_first_turn` are
        // moot (`turns` is non-empty, so neither is ever consulted
        // again), while the tool *format* is re-derived from the same
        // template, since parsing the next reply still needs it.
        let ToolSetup {
            tool_format,
            tool_result_spans,
            capabilities,
            ..
        } = prepare_tools(
            &template_text,
            &template,
            None,
            &meta.tools,
            meta.tool_format.as_ref(),
        );

        Ok(Conversation {
            ctx,
            model: &self.model,
            _handle: self.handle.clone(),
            template,
            n_ctx,
            committed_pos,
            turns: meta.turns,
            overflow: meta.overflow,
            tokens,
            opening: String::new(),
            // See the field's doc: not recoverable from a snapshot.
            protected_prefix: 0,
            system_for_first_turn: None,
            tool_format,
            tool_result_spans,
            capabilities,
            tools: meta.tools,
            configured_tool_format: meta.tool_format,
        })
    }

    /// Wraps `prompt` in the model's own baked-in chat template as a
    /// single "user" turn (`add_ass: true`/`add_generation_prompt: true`,
    /// so the rendered text ends with the assistant turn already opened,
    /// ready for generation to continue into it) before tokenizing.
    /// Previously `generate` fed the raw instructional text straight to
    /// the tokenizer with no role structure at all -- `llama_cpp_2`'s own
    /// doc comment on `apply_chat_template` warns that skipping this "can
    /// result in really unexpected responses," and a real caller-observed
    /// case (Qwen3.6-35B-A3B leaving a referenced type undefined, one
    /// attempt producing no parseable output at all) matched that failure
    /// shape.
    ///
    /// Four-step fallback chain, each step only reached if every one
    /// before it couldn't produce an answer:
    /// 1. `render_with_minijinja` against the model's own real template
    ///    text -- a genuine Jinja engine, not llama.cpp's limited one, so
    ///    this is now the primary path for every model with a template,
    ///    not just ones known to need it (see that function's own doc
    ///    comment for why llama.cpp's own engine isn't good enough:
    ///    real, live case, AI21's Jamba Mini's template uses
    ///    macros/namespaces llama.cpp's Jinja subset can't execute at
    ///    all, `apply_chat_template` returning `ffi error -1`).
    /// 2. llama.cpp's own `apply_chat_template` -- kept as a fallback,
    ///    not removed, in case some template shape renders correctly
    ///    there but not through `minijinja` (untested, defense in depth
    ///    rather than a known real case).
    /// 3. `self.chat_wrap`, if the caller configured one -- a narrow,
    ///    hand-captured override for a template neither engine can
    ///    render (see `ChatWrap`'s own doc comment).
    /// 4. The untouched raw prompt -- strictly better than refusing to
    ///    run at all, the same reasoning that's applied since before any
    ///    of the above existed.
    fn formatted_prompt(&self, prompt: &str) -> Result<String, LlamaSessionError> {
        let template = match self.model.chat_template(None) {
            Ok(template) => template,
            Err(llama_cpp_2::ChatTemplateError::MissingTemplate) => return Ok(prompt.to_string()),
            Err(other) => return Err(other.into()),
        };

        if let Ok(template_text) = template.to_str()
            && let Some(rendered) = render_with_minijinja(template_text, prompt)
        {
            return Ok(rendered);
        }

        let message = LlamaChatMessage::new("user".to_string(), prompt.to_string())?;
        if let Ok(formatted) = self.model.apply_chat_template(&template, &[message], true) {
            return Ok(formatted);
        }

        if let Some(wrap) = &self.chat_wrap {
            return Ok(format!("{}{}{}", wrap.prefix, prompt, wrap.suffix));
        }

        Ok(prompt.to_string())
    }
}

/// What role a completed [`Conversation`] turn belongs to -- tracked per
/// [`TurnBoundary`] so `Conversation::drop_oldest_turns` only ever drops
/// a whole user+assistant pair, never half of one. `Serialize`/
/// `Deserialize` for [`Conversation::save_state`]'s sidecar metadata --
/// llama.cpp's own state file format has no notion of turn/role
/// boundaries, so that bookkeeping travels separately.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Role {
    User,
    Assistant,
}

/// One completed turn's span in the KV cache, `[start_pos, end_pos)`.
/// `Conversation` never truncates mid-span -- `drop_oldest_turns` only
/// ever removes whole entries from the front.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
struct TurnBoundary {
    role: Role,
    start_pos: i32,
    end_pos: i32,
}

/// What a [`Conversation`] does when the next turn wouldn't fit in
/// `n_ctx`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum OverflowPolicy {
    /// Refuse the call (`ConversationContextFull`) rather than lose any
    /// history.
    Fail,
    /// Drop whole turns from the front, oldest first, physically
    /// compacting the KV cache after each one (`kv_cache_seq_rm` +
    /// `kv_cache_seq_add`, the same "context shift" llama.cpp's own
    /// server uses), until the new turn fits.
    DropOldestTurns,
}

pub struct ConversationOptions {
    pub n_ctx: NonZeroU32,
    pub overflow: OverflowPolicy,
    /// Instructions rendered into the model's own *system* block rather
    /// than prepended to the first user message.
    ///
    /// Everything this crate's callers send was previously user text --
    /// `render_with_minijinja` renders exactly one `("user", prompt)`
    /// message, and there was no `"system"` role anywhere in the crate.
    /// So a caller's standing instructions ("you are a coding agent,
    /// here are your tools") competed for attention with the actual
    /// request, in the position a model is trained to treat as the
    /// request. Rendered once into the conversation's opening, since
    /// that is where a template puts it and where the KV cache can hold
    /// it for the whole conversation.
    ///
    /// Ignored, with the text folded into the first user turn instead,
    /// when the model's template has no system block --
    /// `crate::tool_format::ChatCapabilities::system` is how that is
    /// discovered rather than assumed.
    pub system: Option<String>,
    /// Tools to offer, rendered in whatever shape this model's template
    /// renders tools -- see `crate::tool_format`.
    ///
    /// Known at open time rather than per turn because that is where
    /// every template studied puts them: inside the system block, ahead
    /// of the first message. That also makes them free per turn, since
    /// the opening is decoded into the KV cache exactly once.
    ///
    /// Empty or `None` renders no tool list at all, which is not the
    /// same as an empty one: a template branching on `tools is defined`
    /// must see nothing when a caller offers nothing.
    pub tools: Vec<crate::protocol::ToolSpec>,
    /// A tool-call format supplied by the host, used when
    /// `crate::tool_format::derive_tool_call_format` declines to derive
    /// one from this model's template.
    ///
    /// This is the configured half of "ask the model if you can,
    /// otherwise be told": derivation handles the two families real
    /// templates use, and a model whose template renders calls some
    /// third way is a config entry rather than a silent wrong guess or
    /// a hard failure. Deliberately a *fallback* and not an override --
    /// a template that can answer for itself is more trustworthy than a
    /// config file that can drift from the model it describes.
    pub tool_format: Option<crate::protocol::ToolFormat>,
}

/// The fixed spans of literal text a [`Conversation`] composes each new
/// turn's text out of, recovered once (at `open_conversation` time) from
/// a model's real chat template -- see `derive_conversation_template`.
/// Deliberately *not* "re-render the whole transcript through Jinja
/// every turn and diff against last time": a template only exposes a
/// `messages` list, not an incremental-append primitive, and a naive
/// diff between two full re-renders breaks the moment a template renders
/// a *completed* turn differently from a still-*open* one (a real,
/// verified case: AI21's Jamba Mini renders a completed assistant
/// message with a leading space before its content that its
/// `add_generation_prompt` marker alone doesn't include) -- that
/// discrepancy shows up in the middle of the diffed string, not just at
/// the tail, breaking the prefix match on literally the second turn of
/// every conversation. Composing from fixed spans sidesteps that
/// entirely: nothing is ever re-rendered, so nothing can drift out of
/// sync with what's actually sitting in the KV cache.
struct ConversationTemplate {
    /// Text preceding the very first turn's content -- may include
    /// preamble a template only emits before the first message (a
    /// default empty system block, BOS-like framing) that the other two
    /// fields below must not repeat for every later turn. Decoded once,
    /// at the very start of a conversation.
    first_turn_open: String,
    /// Gap between the end of a user turn's content and the start of
    /// the assistant turn that follows it (i.e. what `add_generation_prompt`
    /// contributes) -- decoded once per `send()` call, right after that
    /// call's new user content and right before generation starts.
    generation_open: String,
    /// Gap between the end of a completed assistant turn's content and
    /// the start of the user turn that follows it. Decoded at the start
    /// of the *next* `send()` call, immediately before that call's new
    /// user content -- folding what would otherwise be two separate
    /// steps ("close the previous assistant turn" / "open the next user
    /// turn") into one, since nothing in a template's own rendering
    /// exposes an unambiguous split point between the two.
    turn_transition: String,
}

/// Recovers a [`ConversationTemplate`] from a model's real Jinja chat
/// template, or `None` if this crate can't prove it's safe to reuse
/// turn-by-turn. Renders two independent 4-message probe transcripts --
/// `[user, assistant, user, assistant]`, each message's `content` a
/// distinct sentinel string -- and requires both to agree before
/// trusting either:
///
/// - **Steady-state check**: the gap between the *first* user/assistant
///   pair must equal the gap between the *second* pair. A template that
///   special-cases the very first message (a default system preamble,
///   real and common) would make these differ; that's exactly the class
///   of bug this rejects, rather than silently baking first-turn-only
///   text into every later turn (see `ConversationTemplate::first_turn_open`'s
///   own doc comment).
/// - **Content-fidelity check**: two probes with different sentinel
///   content must recover byte-identical gap text. A template that
///   transforms content (escapes it, truncates it, branches on its
///   value) would make these differ, or make the sentinel fail to
///   appear verbatim at all (`.find` returning `None`) -- either way,
///   this rejects rather than composing turns around a wrap that isn't
///   actually fixed.
///
/// Sentinels use a Private Use Area wrapper plus HTML/JSON-special
/// characters specifically so any escaping/transformation a template
/// applies is likely to break the exact-match `.find()` below, turning
/// a silent miscomposition into a clean `None`.
fn derive_conversation_template(template_text: &str) -> Option<ConversationTemplate> {
    let probe_a = derive_from_probe(
        template_text,
        "\u{E000}RPA1<&\"'>\u{E000}",
        "\u{E000}RPA2<&\"'>\u{E000}",
        "\u{E000}RPA3<&\"'>\u{E000}",
        "\u{E000}RPA4<&\"'>\u{E000}",
    )?;
    let probe_b = derive_from_probe(
        template_text,
        "\u{E000}RPB1<&\"'>\u{E000}",
        "\u{E000}RPB2<&\"'>\u{E000}",
        "\u{E000}RPB3<&\"'>\u{E000}",
        "\u{E000}RPB4<&\"'>\u{E000}",
    )?;
    (probe_a.first_turn_open == probe_b.first_turn_open
        && probe_a.generation_open == probe_b.generation_open
        && probe_a.turn_transition == probe_b.turn_transition)
        .then_some(probe_a)
}

fn derive_from_probe(
    template_text: &str,
    u1: &str,
    a1: &str,
    u2: &str,
    a2: &str,
) -> Option<ConversationTemplate> {
    let messages = [
        ("user", u1),
        ("assistant", a1),
        ("user", u2),
        ("assistant", a2),
    ];
    let rendered = render_messages(template_text, &messages, false)?;

    let u1_start = rendered.find(u1)?;
    let a1_start = rendered.find(a1)?;
    let u2_start = rendered.find(u2)?;
    let a2_start = rendered.find(a2)?;
    // Every sentinel must appear, in order, with none overlapping --
    // rules out a template that reorders, drops, or merges messages.
    if !(u1_start < a1_start && a1_start + a1.len() <= u2_start && u2_start < a2_start) {
        return None;
    }

    let first_turn_open = rendered[..u1_start].to_string();
    let generation_open = rendered[u1_start + u1.len()..a1_start].to_string();
    let turn_transition = rendered[a1_start + a1.len()..u2_start].to_string();

    // The steady-state half of the check described on
    // `derive_conversation_template`: the *second* user/assistant gap
    // must match the first's.
    let generation_open_2 = &rendered[u2_start + u2.len()..a2_start];
    if generation_open != *generation_open_2 {
        return None;
    }

    Some(ConversationTemplate {
        first_turn_open,
        generation_open,
        turn_transition,
    })
}

/// Everything about a [`Conversation`] that llama.cpp's own state file
/// format doesn't carry -- saved as a small JSON sidecar alongside it by
/// [`Conversation::save_state`]. `model_path` exists purely so a later
/// reload can fail closed (`LlamaSessionError::SnapshotModelMismatch`)
/// against an obviously wrong session *before* ever handing the raw
/// state bytes to llama.cpp's own loader, which only checks byte-level
/// shape (`n_ctx`/`n_layer`/quantization), not "is this the model this
/// snapshot actually came from."
#[derive(serde::Serialize, serde::Deserialize)]
struct ConversationSnapshotMeta {
    model_path: PathBuf,
    n_ctx: i32,
    overflow: OverflowPolicy,
    turns: Vec<TurnBoundary>,
    /// The tool list this conversation was opened with, so a reopened
    /// one can re-derive how to parse the next reply's tool calls.
    ///
    /// `#[serde(default)]` is load-bearing, not tidiness: real snapshot
    /// sidecars written before this field existed are sitting on disk
    /// right now (`~/.agentpipe/hibernated/*.meta.json`), and a
    /// non-defaulting field would make every one of them fail to parse
    /// -- turning "this model gained tool support" into "every
    /// hibernated session is unreadable."
    #[serde(default)]
    tools: Vec<crate::protocol::ToolSpec>,
    /// See `ConversationOptions::tool_format`. `#[serde(default)]` for
    /// the same snapshot-compatibility reason as `tools` above.
    #[serde(default)]
    tool_format: Option<crate::protocol::ToolFormat>,
}

/// A single, still-open multi-turn exchange against one resident model --
/// see `LlamaSession::open_conversation`. Holds a real `LlamaContext`
/// (and so a real KV cache) alive for as long as this value lives;
/// dropping it frees the context the same way a `generate()` call's own
/// local context is freed at the end of that call.
pub struct Conversation<'a> {
    ctx: LlamaContext<'a>,
    model: &'a LlamaModel,
    /// Keeps `SwapRegistry` eviction blocked for as long as this
    /// conversation is alive -- same safety invariant `LlamaSession`
    /// itself relies on (see `ModelHandle`'s own doc comment), just also
    /// held here since a `Conversation` can outlive the specific
    /// `generate()`-style call that created it.
    _handle: ModelHandle,
    template: ConversationTemplate,
    /// Next decode for this conversation's (single, fixed) sequence
    /// starts here. llama.cpp sequence ids are always `0` throughout
    /// this crate -- a `Conversation` never multiplexes more than one
    /// logical exchange onto the same context.
    committed_pos: i32,
    n_ctx: i32,
    turns: Vec<TurnBoundary>,
    overflow: OverflowPolicy,
    /// Every token processed so far, in order -- kept in lockstep with
    /// `committed_pos`/the real KV cache (including being trimmed the
    /// same way `drop_oldest_turns_for` trims the cache itself) so
    /// [`Conversation::save_state`] always has a token sequence that
    /// actually matches what's physically resident, not just an
    /// approximation of it.
    tokens: Vec<LlamaToken>,
    /// Text opening the conversation, decoded once before the first
    /// turn's own content. Usually `template.first_turn_open`, but
    /// replaced by a real render when a system prompt or tool list has
    /// to go in it -- see [`prepare_tools`].
    ///
    /// Note that this is *concatenated into the first user turn's text*
    /// rather than decoded on its own -- see [`Conversation::send`]. Its
    /// tokens therefore sit at the front of `turns[0]`'s span, which is
    /// what `protected_prefix` exists to keep `drop_oldest_turns_for`
    /// from reclaiming.
    opening: String,
    /// How many tokens at the front of the KV cache are the `opening`,
    /// and so must never be evicted.
    ///
    /// Zero means "not known", and eviction then behaves as it did
    /// before this existed. That is the case for a conversation restored
    /// from a snapshot: its opening was decoded by the session that
    /// saved it, and the boundary is not recoverable from the snapshot
    /// without re-deriving a tokenisation that has to match the resident
    /// cache exactly. Guessing it wrong would remove the wrong positions
    /// and corrupt the cache, which is worse than the old behaviour, so
    /// the restore path does not guess.
    protected_prefix: i32,
    /// System text that could *not* be rendered as a system block
    /// because this model's template has none, to be folded into the
    /// first user turn instead. `None` whenever the template does
    /// support one (the normal case), since then it is already inside
    /// `opening`.
    system_for_first_turn: Option<String>,
    /// How this model writes a tool call, when it was derivable and
    /// tools were actually offered. `None` means [`Conversation::send`]
    /// reports no tool calls at all, which is correct for a
    /// conversation that offered no tools.
    tool_format: Option<crate::protocol::ToolFormat>,
    /// How this template renders a sequence of tool results -- see
    /// `crate::tool_format::ToolResultSpans`.
    tool_result_spans: Option<crate::tool_format::ToolResultSpans>,
    capabilities: crate::tool_format::ChatCapabilities,
    /// The tool list this conversation was opened with, kept only so
    /// [`Conversation::save_state`] can record it -- a reopened
    /// conversation needs it to re-derive how to parse tool calls,
    /// and nothing else here reads it.
    tools: Vec<crate::protocol::ToolSpec>,
    /// The host-supplied fallback format, kept only so
    /// [`Conversation::save_state`] can record it alongside `tools` --
    /// a reopened conversation needs both to rebuild the same parser.
    configured_tool_format: Option<crate::protocol::ToolFormat>,
}

/// What [`prepare_tools`] worked out about this conversation's opening
/// and this model's tool-calling surface.
struct ToolSetup {
    opening: String,
    system_for_first_turn: Option<String>,
    tool_format: Option<crate::protocol::ToolFormat>,
    tool_result_spans: Option<crate::tool_format::ToolResultSpans>,
    capabilities: crate::tool_format::ChatCapabilities,
}

/// Decides, once per conversation, how a system prompt and tool list
/// reach the model -- by asking the model's own template rather than
/// assuming any of it.
///
/// Everything here degrades rather than fails. A template with no
/// system block still gets the system text, folded into the first user
/// turn (which is what every caller did unconditionally before). A
/// template whose tool-call format isn't derivable still renders the
/// tool list, and simply reports no parsed calls -- leaving a caller
/// free to fall back to its own prompt-and-grammar arrangement. Nothing
/// about offering tools is allowed to make a conversation unopenable.
fn prepare_tools(
    template_text: &str,
    template: &ConversationTemplate,
    system: Option<&str>,
    tools: &[crate::protocol::ToolSpec],
    configured_format: Option<&crate::protocol::ToolFormat>,
) -> ToolSetup {
    let render = crate::chat_template::probe_renderer();
    let capabilities = crate::tool_format::derive_capabilities(template_text, &render);

    // An empty list is not the same as no list: a template branching on
    // `tools is defined` must see nothing when nothing is offered.
    let offered = (!tools.is_empty() && capabilities.tools).then_some(tools);
    let system_block = system.filter(|_| capabilities.system);

    // Only re-render the opening when something actually has to go in
    // it. Otherwise keep the derived span verbatim, so a conversation
    // that offers neither is byte-identical to before this existed.
    let rendered_opening = (system_block.is_some() || offered.is_some())
        .then(|| crate::tool_format::render_opening(template_text, system_block, offered, &render))
        .flatten();

    let tool_format = offered
        .is_some()
        .then(|| {
            crate::tool_format::derive_tool_call_format(template_text, &render)
                // Derivation first, config second -- see
                // `ConversationOptions::tool_format`.
                .or_else(|| configured_format.cloned())
        })
        .flatten();
    let tool_result_spans = (offered.is_some() && capabilities.tool_results)
        .then(|| crate::tool_format::derive_tool_result_spans(template_text, &render))
        .flatten();

    let opening_carries_system = system_block.is_some() && rendered_opening.is_some();

    ToolSetup {
        opening: rendered_opening.unwrap_or_else(|| template.first_turn_open.clone()),
        // Fold the system text into the first user turn when, and only
        // when, it did not make it into the opening -- either because
        // the template has no system block, or because rendering the
        // opening failed and the derived span (which contains no system
        // text) is being used instead.
        system_for_first_turn: system
            .filter(|_| !opening_carries_system)
            .map(str::to_string),
        tool_format,
        tool_result_spans,
        capabilities,
    }
}

impl<'a> Conversation<'a> {
    /// Number of completed user+assistant exchanges so far.
    pub fn turn_count(&self) -> usize {
        self.turns.len() / 2
    }

    /// Persists this conversation's full KV cache to `state_path` (via
    /// llama.cpp's own `state_save_file`) plus everything that format
    /// doesn't carry (turn/role boundaries, the model this came from) to
    /// a small JSON sidecar at `meta_path`. Both paths are caller-chosen
    /// -- no naming convention imposed here -- so a caller managing many
    /// saved sessions (e.g. evicting an idle resident agent without
    /// losing its context) can lay them out however its own retention
    /// policy wants. See [`LlamaSession::open_conversation_from_state`]
    /// for the reverse direction.
    ///
    /// Real, live-relevant cost: the state file is roughly proportional
    /// to context length used × layers × heads -- easily tens to
    /// hundreds of MB for an actual conversation, not a token-count-sized
    /// artifact. A caller doing this routinely needs its own cleanup
    /// policy for stale snapshots; nothing here expires them.
    pub fn save_state(
        &self,
        state_path: impl AsRef<Path>,
        meta_path: impl AsRef<Path>,
    ) -> Result<(), LlamaSessionError> {
        self.ctx.state_save_file(state_path, &self.tokens)?;
        let meta = ConversationSnapshotMeta {
            model_path: self._handle.path().to_path_buf(),
            n_ctx: self.n_ctx,
            overflow: self.overflow,
            turns: self.turns.clone(),
            tools: self.tools.clone(),
            tool_format: self.configured_tool_format.clone(),
        };
        let json = serde_json::to_vec_pretty(&meta)?;
        std::fs::write(meta_path, json)?;
        Ok(())
    }

    /// Sends one new user message and returns the model's reply, with
    /// both now part of this conversation's persistent KV cache. Unlike
    /// `LlamaSession::generate`, only *this* call's own new text is
    /// tokenized and decoded -- every earlier turn is already resident in
    /// the context from a prior `send()`.
    ///
    /// `grammar`/`assistant_prefill`/`grammar_complete` mirror
    /// `LlamaSession::generate`'s own parameters of the same names --
    /// see that method's doc comment for what each does. Grammar
    /// constraint and prefill both apply to this turn's assistant reply
    /// only; they don't persist to later `send()` calls on the same
    /// conversation.
    pub fn send(
        &mut self,
        message: &str,
        max_new_tokens: i32,
        sampling: Sampling,
        grammar: Option<&str>,
        assistant_prefill: Option<&str>,
        grammar_complete: Option<&dyn Fn(&str) -> bool>,
    ) -> Result<GenerationResult, LlamaSessionError> {
        let start = Instant::now();

        let (opening, add_bos) = if self.turns.is_empty() {
            (self.opening.as_str(), AddBos::Always)
        } else {
            (self.template.turn_transition.as_str(), AddBos::Never)
        };
        // Only ever on the very first turn, and only for a model whose
        // template has no system block of its own -- see
        // `ConversationOptions::system` and `prepare_tools`.
        let message = match (&self.system_for_first_turn, self.turns.is_empty()) {
            (Some(system), true) => std::borrow::Cow::Owned(format!("{system}\n\n{message}")),
            _ => std::borrow::Cow::Borrowed(message),
        };
        let mut user_text = format!("{opening}{message}{}", self.template.generation_open);
        // Prefilling the assistant turn: same technique as
        // `LlamaSession::generate`'s own `assistant_prefill` handling --
        // appending here before tokenizing makes generation resume
        // *inside* the prefill rather than at the start of a fresh turn.
        // Prepended back onto `text` below so the caller sees one
        // complete, self-consistent string.
        if let Some(prefill) = assistant_prefill {
            user_text.push_str(prefill);
        }
        let user_tokens = self.model.str_to_token(&user_text, add_bos)?;

        // Where the opening ends and this turn's own content begins.
        //
        // Established once, on the first turn, because that is the only
        // turn whose span contains the opening. `render_opening` cuts the
        // rendered template at the user content placeholder, so the seam
        // falls immediately after a turn-opening control token
        // (`<|im_start|>user\n` and its equivalents) -- a boundary BPE
        // does not merge across. Verified rather than assumed: if
        // tokenising the two halves separately does not reproduce the
        // whole, the boundary is not where it is believed to be, and the
        // prefix stays unprotected rather than being protected in the
        // wrong place. Removing the wrong KV positions would corrupt the
        // cache, which is a worse failure than the one this prevents.
        if self.turns.is_empty() && !opening.is_empty() {
            let opening_tokens = self.model.str_to_token(opening, add_bos)?;
            if user_tokens.starts_with(&opening_tokens) {
                self.protected_prefix = opening_tokens.len() as i32;
            } else {
                eprintln!(
                    "rampiped: the opening does not tokenise as a prefix of the first turn, so it \
                     cannot be protected from context eviction. A long conversation may lose its \
                     system block and tool definitions."
                );
            }
        }

        // `max_new_tokens`, not just the incoming message, or a
        // conversation whose running budget is merely *tight* (not yet
        // full) silently generates fewer tokens than asked -- possibly
        // none at all -- instead of `drop_oldest_turns_for` evicting
        // enough history upfront to leave real room for the reply. Real,
        // live bug: found via `examples/conversation_overflow_smoke.rs`
        // producing correct replies for its first ~22 turns, then
        // silently empty ones for the rest, well before `n_ctx` was
        // actually exhausted by the conversation's own content alone.
        self.ensure_room_for(user_tokens.len() as i32 + max_new_tokens)?;

        let mut batch = LlamaBatch::new(512, 1);
        let user_start = self.committed_pos;
        decode_chunked(
            &mut self.ctx,
            &mut batch,
            &user_tokens,
            &mut self.committed_pos,
        )?;
        self.turns.push(TurnBoundary {
            role: Role::User,
            start_pos: user_start,
            end_pos: self.committed_pos,
        });
        self.tokens.extend(user_tokens);

        let assistant_start = self.committed_pos;
        // Built per turn from the format this conversation derived, so a
        // model that keeps generating past its own tool call is stopped
        // rather than left to invent the results. See `TurnEnd`.
        let turn_end = self
            .tool_format
            .as_ref()
            .map(crate::tool_format::TurnEnd::of);
        let (text, generated_tokens, time_to_first_token) = run_generation_loop(
            &mut self.ctx,
            self.model,
            &mut batch,
            &mut self.committed_pos,
            self.n_ctx,
            max_new_tokens,
            sampling,
            grammar,
            grammar_complete,
            turn_end.as_ref(),
            start,
        )?;
        self.turns.push(TurnBoundary {
            role: Role::Assistant,
            start_pos: assistant_start,
            end_pos: self.committed_pos,
        });
        let tokens_generated = generated_tokens.len();
        self.tokens.extend(generated_tokens);

        let text = match assistant_prefill {
            Some(prefill) => format!("{prefill}{text}"),
            None => text,
        };

        let tool_calls = self
            .tool_format
            .as_ref()
            .map(|format| crate::tool_format::parse_tool_calls(&text, format))
            .unwrap_or_default();
        let truncated_tool_call = self
            .tool_format
            .as_ref()
            .is_some_and(|format| crate::tool_format::ends_mid_call(&text, format));

        Ok(GenerationResult {
            committed_tokens: self.committed_pos.max(0) as usize,
            context_size: self.n_ctx as usize,
            text,
            time_to_first_token,
            tokens_generated,
            formatted_prompt: user_text,
            tool_calls,
            truncated_tool_call,
        })
    }

    /// Whether tool calls emitted by this conversation can actually be
    /// decoded -- true only when tools were offered *and* the model's
    /// template yielded a derivable call format. A caller uses this to
    /// choose between the tool-calling path and its own
    /// prompt-and-grammar arrangement, rather than discovering the
    /// answer from an empty `tool_calls` on a turn that simply didn't
    /// call anything.
    pub fn supports_tool_calls(&self) -> bool {
        self.tool_format.is_some()
    }

    /// What this model's template was found to accept -- see
    /// `crate::tool_format::ChatCapabilities`.
    pub fn capabilities(&self) -> crate::tool_format::ChatCapabilities {
        self.capabilities
    }

    /// Feeds executed tool results back and generates the model's next
    /// turn, as the `tool` role its template defines rather than as an
    /// ordinary user message.
    ///
    /// This is the other half of tool calling, and the reason a caller
    /// can't just fold results into `send`: the model was trained to
    /// read results inside its own result markers (Qwen3-Coder wraps
    /// them in `<tool_response>` inside a user turn), and text arriving
    /// as a plain user message is a different thing to it than a result
    /// arriving as a result.
    ///
    /// `results` are concatenated in call order into one result turn,
    /// which is what every template studied renders for several results
    /// at once. Falls back to [`Conversation::send`] with the results
    /// joined as ordinary text when this model has no result markers to
    /// use -- degrading the same way everything else here does, rather
    /// than refusing.
    pub fn send_tool_results(
        &mut self,
        results: &[String],
        max_new_tokens: i32,
        sampling: Sampling,
        grammar: Option<&str>,
        grammar_complete: Option<&dyn Fn(&str) -> bool>,
    ) -> Result<GenerationResult, LlamaSessionError> {
        let Some(spans) = self.tool_result_spans.clone() else {
            return self.send(
                &results.join("\n"),
                max_new_tokens,
                sampling,
                grammar,
                None,
                grammar_complete,
            );
        };

        let start = Instant::now();
        // Each result as its own element, in this template's own shape
        // -- see `ToolResultSpans`.
        let text_in = spans.render(results);
        // Never `AddBos::Always`: a result turn can only ever follow a
        // model turn that asked for it, so there is always earlier
        // content and a second BOS would corrupt the sequence.
        let tokens = self.model.str_to_token(&text_in, AddBos::Never)?;
        self.ensure_room_for(tokens.len() as i32 + max_new_tokens)?;

        let mut batch = LlamaBatch::new(512, 1);
        let turn_start = self.committed_pos;
        decode_chunked(&mut self.ctx, &mut batch, &tokens, &mut self.committed_pos)?;
        // Recorded as a `User` turn deliberately: `TurnBoundary`'s roles
        // exist so `drop_oldest_turns_for` can evict whole exchanges,
        // and a result turn is evicted with the user turn it belongs to.
        // A third role would have to be threaded through the snapshot
        // sidecar (and every already-written one on disk) to express a
        // distinction nothing here acts on.
        self.turns.push(TurnBoundary {
            role: Role::User,
            start_pos: turn_start,
            end_pos: self.committed_pos,
        });
        self.tokens.extend(tokens);

        let assistant_start = self.committed_pos;
        // Built per turn from the format this conversation derived, so a
        // model that keeps generating past its own tool call is stopped
        // rather than left to invent the results. See `TurnEnd`.
        let turn_end = self
            .tool_format
            .as_ref()
            .map(crate::tool_format::TurnEnd::of);
        let (text, generated_tokens, time_to_first_token) = run_generation_loop(
            &mut self.ctx,
            self.model,
            &mut batch,
            &mut self.committed_pos,
            self.n_ctx,
            max_new_tokens,
            sampling,
            grammar,
            grammar_complete,
            turn_end.as_ref(),
            start,
        )?;
        self.turns.push(TurnBoundary {
            role: Role::Assistant,
            start_pos: assistant_start,
            end_pos: self.committed_pos,
        });
        let tokens_generated = generated_tokens.len();
        self.tokens.extend(generated_tokens);

        let tool_calls = self
            .tool_format
            .as_ref()
            .map(|format| crate::tool_format::parse_tool_calls(&text, format))
            .unwrap_or_default();
        let truncated_tool_call = self
            .tool_format
            .as_ref()
            .is_some_and(|format| crate::tool_format::ends_mid_call(&text, format));

        Ok(GenerationResult {
            committed_tokens: self.committed_pos.max(0) as usize,
            context_size: self.n_ctx as usize,
            text,
            time_to_first_token,
            tokens_generated,
            formatted_prompt: text_in,
            tool_calls,
            truncated_tool_call,
        })
    }

    /// Ensures `needed` more tokens will fit before `committed_pos`
    /// reaches `n_ctx`, dropping oldest turns first if `overflow` allows
    /// it.
    fn ensure_room_for(&mut self, needed: i32) -> Result<(), LlamaSessionError> {
        if self.committed_pos + needed < self.n_ctx {
            return Ok(());
        }
        match self.overflow {
            OverflowPolicy::Fail => Err(LlamaSessionError::ConversationContextFull {
                committed_pos: self.committed_pos,
                needed,
                n_ctx: self.n_ctx,
            }),
            OverflowPolicy::DropOldestTurns => self.drop_oldest_turns_for(needed),
        }
    }

    /// Drops whole user+assistant turn pairs from the front, oldest
    /// first, until `needed` more tokens fit -- physically compacting the
    /// KV cache after each drop so the freed span doesn't just sit there
    /// as a hole: `kv_cache_seq_rm` removes the positions, then
    /// `kv_cache_seq_add` with a negative delta shifts every later
    /// position down to close the gap (RoPE-aware, per llama.cpp -- the
    /// same "context shift" its own server implements for the same
    /// reason).
    ///
    /// The opening is exempt. This used to drop it along with the first
    /// turn, because `send` concatenates it into that turn's text and so
    /// its tokens sit at the front of `turns[0]`'s span -- and that was
    /// documented as an accepted approximation, "the same a real
    /// context-shifting inference server already makes". It is not an
    /// acceptable approximation for a caller whose system block *is* the
    /// task. Measured: an agent whose window filled at turn 12 had its
    /// system block and tool definitions removed and then emitted the
    /// same malformed tool call several hundred times, because what
    /// remained was file contents with nothing describing what a tool
    /// call looked like or what the work was. Now `protected_prefix`
    /// tokens at the front are never reclaimed, and the first turn is
    /// evicted from the end of the opening rather than from position 0.
    fn drop_oldest_turns_for(&mut self, needed: i32) -> Result<(), LlamaSessionError> {
        let mut freed = 0i32;
        while self.committed_pos + needed - freed >= self.n_ctx {
            if self.turns.len() < 2 {
                return Err(LlamaSessionError::ConversationTooLargeToTrim);
            }
            let user = self.turns.remove(0);
            let assistant = self.turns.remove(0);
            // Never into the opening. For every turn but the first this
            // is a no-op, since their spans start past it.
            let start = user.start_pos.max(self.protected_prefix);
            // Said out loud, because a caller cannot infer it from the
            // response and the consequence is severe: an exchange it
            // believes the model can still see is gone. The first
            // `protected_prefix` tokens -- system block and tool
            // definitions -- are excluded, so what is named here really
            // is only conversation.
            eprintln!(
                "rampiped: context full ({}/{}) -- dropping the oldest exchange (positions {start}..{}), \
                 keeping the first {} tokens of opening. Anything a caller sent in that exchange is no \
                 longer visible to the model.",
                self.committed_pos, self.n_ctx, assistant.end_pos, self.protected_prefix
            );
            // `turns` must strictly alternate User/Assistant starting
            // with User -- every push site above maintains that, so a
            // mismatch here means the bookkeeping itself is broken, not
            // just an unlucky conversation shape.
            debug_assert_eq!(user.role, Role::User);
            debug_assert_eq!(assistant.role, Role::Assistant);
            let removed = assistant.end_pos - start;
            if removed <= 0 {
                // The whole pair is inside the protected prefix. Nothing
                // can be freed by dropping it, and looping would spin.
                return Err(LlamaSessionError::ConversationTooLargeToTrim);
            }

            self.ctx
                .kv_cache_seq_rm(0, Some(start as u32), Some(assistant.end_pos as u32))?;
            self.ctx
                .kv_cache_seq_add(0, Some(assistant.end_pos as u32), None, -removed)?;

            for turn in &mut self.turns {
                turn.start_pos -= removed;
                turn.end_pos -= removed;
            }
            self.committed_pos -= removed;
            // Keeps `self.tokens` in the same lockstep with the real KV
            // cache the position bookkeeping above maintains. The dropped
            // span begins at `start`, not at zero: the opening stays put
            // at the front, and every earlier drop has already shifted
            // what follows down to meet it.
            self.tokens
                .drain(start as usize..assistant.end_pos as usize);
            freed += removed;
        }
        Ok(())
    }
}

/// KV cache defragmentation threshold for every `Conversation` context
/// (`open_conversation`/`open_conversation_from_state`).
///
/// llama.cpp's own default (`llama_context_default_params`) is `-1.0`,
/// which disables automatic defragmentation entirely -- fine for a
/// context that only ever appends, but `drop_oldest_turns_for`'s
/// `kv_cache_seq_rm`/`kv_cache_seq_add` surgery is exactly the workload
/// that fragments a cache over a long-running session: real, live
/// failure, a long tool-use conversation with many turns dropped over
/// time hit `llama.cpp decode error: Decode Error 1: NoKvCacheSlot` on
/// an ordinary turn, despite `committed_pos` staying well under `n_ctx`
/// by `run_generation_loop`'s own position-based accounting -- the
/// *positions* had room, but the physical KV cache cells didn't have a
/// large enough contiguous free span left for the batch, since nothing
/// was ever compacting the fragmentation `kv_cache_seq_rm`/`_add` leaves
/// behind beyond shifting position *numbers*. `0.1` matches
/// `llama-cpp-2`'s own doctest example and llama.cpp's conventional
/// default when defrag is explicitly enabled.
const CONVERSATION_DEFRAG_THOLD: f32 = 0.1;

/// A model callable in-process or over some other transport -- the
/// common seam `taskpipe::backend::InferenceClient` used to be the only
/// implementation of; living here instead means any caller (not just
/// taskpipe) gets the same in-process/daemon/remote dispatch without
/// reimplementing it. `LlamaSession` is the in-process implementation
/// (below); a `rampiped`-socket-backed and a remote-HTTP-backed
/// implementation are the other two real shapes this is meant for, each
/// living wherever that transport's own code lives.
pub trait LocalModel: Send + Sync {
    fn complete(
        &self,
        prompt: &str,
        max_new_tokens: i32,
        sampling: Sampling,
    ) -> Result<GenerationResult, LlamaSessionError>;

    /// Opens a real multi-turn session against this model. Boxed and
    /// trait-object-safe since different `LocalModel` implementations
    /// back this with genuinely different session types (an in-process
    /// `Conversation` holding a live `LlamaContext`; a socket-backed
    /// implementation would hold a conversation id instead) -- a caller
    /// generic over `LocalModel` (or holding `Box<dyn LocalModel>`, the
    /// way `taskpipe::backend::Executor` holds `Box<dyn InferenceClient>`
    /// today) only ever needs `ConversationHandle`'s common surface, not
    /// which concrete type is behind it.
    fn open_conversation(
        &self,
        options: ConversationOptions,
    ) -> Result<Box<dyn ConversationHandle + '_>, LlamaSessionError>;
}

impl ConversationHandle for Conversation<'_> {
    fn send(
        &mut self,
        message: &str,
        max_new_tokens: i32,
        sampling: Sampling,
        grammar: Option<&str>,
        assistant_prefill: Option<&str>,
        grammar_completion: Option<crate::protocol::GrammarCompletion>,
    ) -> Result<GenerationResult, ConversationError> {
        let grammar_complete =
            grammar_completion.map(crate::protocol::GrammarCompletion::into_predicate);
        Conversation::send(
            self,
            message,
            max_new_tokens,
            sampling,
            grammar,
            assistant_prefill,
            grammar_complete.as_deref(),
        )
        .map_err(ConversationError::Llama)
    }

    fn supports_tool_calls(&self) -> bool {
        Conversation::supports_tool_calls(self)
    }

    fn send_tool_results(
        &mut self,
        results: &[String],
        max_new_tokens: i32,
        sampling: Sampling,
        grammar: Option<&str>,
        grammar_completion: Option<crate::protocol::GrammarCompletion>,
    ) -> Result<GenerationResult, ConversationError> {
        let grammar_complete =
            grammar_completion.map(crate::protocol::GrammarCompletion::into_predicate);
        Conversation::send_tool_results(
            self,
            results,
            max_new_tokens,
            sampling,
            grammar,
            grammar_complete.as_deref(),
        )
        .map_err(ConversationError::Llama)
    }

    fn turn_count(&self) -> usize {
        Conversation::turn_count(self)
    }

    fn snapshot(&mut self, state_path: &Path, meta_path: &Path) -> Result<(), ConversationError> {
        Conversation::save_state(self, state_path, meta_path).map_err(ConversationError::Llama)
    }
}

impl LocalModel for LlamaSession {
    fn complete(
        &self,
        prompt: &str,
        max_new_tokens: i32,
        sampling: Sampling,
    ) -> Result<GenerationResult, LlamaSessionError> {
        self.generate(prompt, max_new_tokens, sampling, None, None, None)
    }

    fn open_conversation(
        &self,
        options: ConversationOptions,
    ) -> Result<Box<dyn ConversationHandle + '_>, LlamaSessionError> {
        Ok(Box::new(LlamaSession::open_conversation(self, options)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A model's real `tokenizer.chat_template` GGUF metadata, extracted
    /// once via Python's `gguf` package against the real, downloaded
    /// `bartowski/ai21labs_AI21-Jamba-Mini-1.7-GGUF` file (not hand-
    /// assembled) -- the same template whose rendering failure through
    /// llama.cpp's own engine (`ffi error -1`) motivated
    /// `render_with_minijinja` existing at all. Real macros, a
    /// `namespace()`, `is not defined` checks, `tojson`, tool/document
    /// handling this test's own call never exercises -- exactly the
    /// shape `render_with_minijinja`'s doc comment claims llama.cpp's
    /// engine can't execute but this can.
    const JAMBA_CHAT_TEMPLATE: &str =
        include_str!("../tests/fixtures/jamba_mini_1_7_chat_template.jinja");

    const CHATML_TEMPLATE: &str = "{% for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n\
                         {% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";

    /// Real, live-observed template that fails without
    /// `set_unknown_method_callback` -- confirmed live against
    /// `agentpiped`, `open_conversation` rejected this model with "no
    /// template at all, or not provably steady-state" because
    /// `render_messages` silently swallowed the real error
    /// (`.render(ctx).ok()`); the actual failure, found by removing
    /// that `.ok()` in a standalone probe, was `unknown method: string
    /// has no method named startswith` at the template's own
    /// `content.startswith('<tool_response>')` call.
    const QWEN_3_8_CHAT_TEMPLATE: &str =
        include_str!("../tests/fixtures/qwen3_8_chat_template.jinja");

    #[test]
    fn renders_jambas_real_template_for_a_single_user_turn() {
        let rendered = render_with_minijinja(JAMBA_CHAT_TEMPLATE, "Hello, how are you?")
            .expect("should render");
        assert_eq!(
            rendered,
            "<|bom|><|system|> <|eom|><|bom|><|user|> Hello, how are you?<|eom|><|bom|><|assistant|>"
        );
    }

    /// Not just Jamba-specific -- a plain ChatML-style template (the
    /// shape most instruct-tuned GGUFs actually ship, and one llama.cpp's
    /// own engine already handles fine) needs to keep working too, since
    /// this is now the *primary* renderer for every model, not a
    /// Jamba-only escape hatch.
    #[test]
    fn renders_a_plain_chatml_style_template() {
        let rendered = render_with_minijinja(CHATML_TEMPLATE, "hi").expect("should render");
        assert_eq!(
            rendered,
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn returns_none_for_a_template_with_genuinely_invalid_syntax() {
        assert_eq!(
            render_with_minijinja("{% this is not valid jinja %}", "hi"),
            None
        );
    }

    /// `raise_exception` must be registered -- a template calling it
    /// (even one that would never do so for a real single-user-turn
    /// input) shouldn't fail with "unknown function" instead of the
    /// template's own intended error.
    #[test]
    fn a_template_defining_raise_exception_as_a_call_does_not_fail_on_an_unknown_function() {
        let template =
            "{% if false %}{{ raise_exception(\"unreachable\") }}{% endif %}{{ prompt }}";
        // `prompt` isn't part of the context this function builds (only
        // `messages`/`add_generation_prompt` are) -- this asserts the
        // render doesn't fail on `raise_exception` being undefined, not
        // that this exact template produces a particular string.
        assert!(render_with_minijinja(template, "hi").is_some());
    }

    #[test]
    fn derives_a_conversation_template_for_plain_chatml() {
        let template = derive_conversation_template(CHATML_TEMPLATE).expect("should derive");
        // `generation_open` spans from the end of a user turn's content
        // to the start of the following assistant turn's content in a
        // *completed* render -- that includes the user's own closing
        // `<|im_end|>\n`, not just the bare `add_generation_prompt`
        // marker, since that's what a real multi-turn transcript
        // actually has between the two spans.
        assert_eq!(template.first_turn_open, "<|im_start|>user\n");
        assert_eq!(
            template.generation_open,
            "<|im_end|>\n<|im_start|>assistant\n"
        );
        assert_eq!(template.turn_transition, "<|im_end|>\n<|im_start|>user\n");

        // Composing turn 1 out of these spans must reproduce exactly
        // what `render_with_minijinja` (real `generate()`'s own
        // one-shot renderer) already produces for a single open turn --
        // the two paths ought to agree on what "send the first message"
        // looks like.
        let composed_turn_1 = format!(
            "{}{}{}",
            template.first_turn_open, "hi", template.generation_open
        );
        assert_eq!(
            composed_turn_1,
            render_with_minijinja(CHATML_TEMPLATE, "hi").unwrap()
        );
    }

    /// The real case that motivated composing turns from fixed spans
    /// instead of diffing two full re-renders: Jamba always prepends a
    /// `<|bom|><|system|> <|eom|>` block before the first real message,
    /// but the steady-state check must NOT let that leak into
    /// `generation_open`/`turn_transition`, which apply to every turn.
    #[test]
    fn derives_a_conversation_template_for_jamba_without_leaking_its_first_turn_preamble() {
        let template = derive_conversation_template(JAMBA_CHAT_TEMPLATE).expect("should derive");
        assert!(
            template.first_turn_open.contains("<|system|>"),
            "system preamble belongs only in first_turn_open, got {:?}",
            template.first_turn_open
        );
        assert!(!template.generation_open.contains("<|system|>"));
        assert!(!template.turn_transition.contains("<|system|>"));
    }

    /// The actual bug this session's `pycompat` fix closes -- a small,
    /// readable reproduction rather than relying solely on the full real
    /// Qwen 3.8 template below to prove the mechanism. Python's own
    /// Jinja2 lets a template call native `str` methods it never
    /// defines itself (like `.startswith()`); minijinja has no such
    /// fallback without `set_unknown_method_callback`.
    #[test]
    fn renders_a_template_that_calls_pythons_native_startswith_method() {
        let template = "{% if messages[0].content.startswith('hi') %}yes{% else %}no{% endif %}";
        let rendered = render_messages(template, &[("user", "hi there")], false)
            .expect("pycompat should make .startswith() resolve, not error");
        assert_eq!(rendered, "yes");
    }

    /// The real, live-observed case -- confirms the actual model
    /// template that motivated this fix now derives a genuine,
    /// steady-state conversation template end to end, not just that the
    /// isolated `.startswith()` call works in a toy template.
    #[test]
    fn derives_a_conversation_template_for_the_real_qwen_3_8_template() {
        let template = derive_conversation_template(QWEN_3_8_CHAT_TEMPLATE)
            .expect("should derive now that pycompat covers .startswith()/.endswith()");
        assert!(template.generation_open.contains("<|im_start|>assistant"));
    }

    /// The real, live-observed crash `open_conversation`'s own
    /// `rope_type()` guard exists for -- and the real reason it's an
    /// allowlist, not a denylist: confirmed live, this actual model
    /// doesn't report `Some(MRope)`/`Some(Vision)` at all. It reports
    /// `None`, because llama.cpp's own `llama_model_rope_type()`
    /// returns `LLAMA_ROPE_TYPE_IMROPE` for its architecture (`QWEN35`,
    /// grouped with the vision archs in llama.cpp's own source) -- a
    /// rope kind this binding's `rope_type()` has no match arm for, so
    /// it silently falls through to the catch-all `None` case. A
    /// denylist keyed on the two *named* multimodal variants would have
    /// missed this and let the crash back in; this is exactly why
    /// `open_conversation` only trusts the two *positively confirmed*
    /// ordinary rope kinds (`Norm`, `NeoX`) with the risky
    /// `kv_cache_seq_add` path, not "everything except a fixed list of
    /// known-bad ones." Needs the real, large GGUF this crash was found
    /// against -- set `QWEN_3_8_GGUF_PATH` to run; skips (not fails)
    /// everywhere else, same pattern `agentpipe::cli_agent`'s own
    /// credentialed tests use.
    #[test]
    #[ignore = "needs the real, large Qwen 3.8 GGUF -- set QWEN_3_8_GGUF_PATH to run"]
    fn qwen_3_8_does_not_report_an_ordinary_single_position_rope_type() {
        let Ok(path) = std::env::var("QWEN_3_8_GGUF_PATH") else {
            eprintln!("skipping: QWEN_3_8_GGUF_PATH not set");
            return;
        };
        let registry = SwapRegistry::new();
        let backend = Arc::new(LlamaBackend::init().expect("init backend"));
        let session =
            LlamaSession::load(&registry, backend, &path, Residency::Lazy).expect("load model");
        let rope_type = session.model.rope_type();
        assert!(
            !matches!(rope_type, Some(RopeType::Norm) | Some(RopeType::NeoX)),
            "expected a non-ordinary rope type (this model's real value is None, via the \
             unmapped LLAMA_ROPE_TYPE_IMROPE case), got {rope_type:?} -- if this now reports \
             Norm/NeoX, either the model changed or llama-cpp-2 started mapping IMROPE, and \
             open_conversation's own allowlist comment should be revisited",
        );
    }

    /// There is no real bug here -- see this crate's own commit history
    /// for the false alarm this test was originally written to chase
    /// down: `GpuLayers::Auto` was observed offloading 0/66 layers for
    /// this model, which looked like a `fit_params` (llama.cpp's own
    /// `common_fit_params`) estimation bug for this new architecture.
    /// It wasn't -- confirmed live, the actual cause was mundane: an
    /// already-running `rampiped` process (a *different* model, left
    /// over from earlier testing) was holding ~14GB of this box's ~16GB
    /// VRAM resident the entire time, so `fit_params` correctly saw
    /// almost no free memory and correctly declined to offload anything.
    /// With that process killed and real VRAM free, plain
    /// `GpuLayers::Auto` offloads all 66/66 layers with no forcing
    /// needed at all. This test now exists only as a plain sanity check
    /// that the `GpuLayers::Fixed` escape hatch itself still works for
    /// this model, not to guard against anything broken.
    #[test]
    #[ignore = "needs the real, large Qwen 3.8 GGUF -- set QWEN_3_8_GGUF_PATH to run"]
    fn qwen_3_8_loads_with_an_explicit_forced_gpu_layer_count() {
        let Ok(path) = std::env::var("QWEN_3_8_GGUF_PATH") else {
            eprintln!("skipping: QWEN_3_8_GGUF_PATH not set");
            return;
        };
        let registry = SwapRegistry::new();
        let backend = Arc::new(LlamaBackend::init().expect("init backend"));
        let session = LlamaSession::load_with_gpu_layers(
            &registry,
            backend,
            &path,
            Residency::Lazy,
            GpuLayers::Fixed(30),
        );
        assert!(
            session.is_ok(),
            "forced offload should at least not error: {:?}",
            session.err()
        );
    }

    #[test]
    fn rejects_a_template_with_no_chat_syntax_at_all() {
        // A template with no `messages` loop at all still renders (as
        // literal text with no sentinel present), so this only documents
        // the shape rather than asserting a specific outcome -- the real
        // safety net is `derive_from_probe`'s sentinel-position checks,
        // covered by the tests above using real templates.
        let _ = derive_conversation_template("just plain text, no {{ }} or {% %} anywhere");
    }

    #[test]
    fn rejects_a_template_that_transforms_content_instead_of_embedding_it_verbatim() {
        // Reverses each message's content -- a stand-in for any
        // content-dependent transformation (escaping, truncation,
        // case-folding). The two probes' sentinels are different
        // strings, so their reversed forms differ in a way that isn't a
        // fixed function of position alone, and `.find()` for the
        // literal (non-reversed) sentinel fails outright against
        // reversed output -- either way this must reject, not silently
        // compose turns around a wrap that doesn't actually hold.
        let template = "{% for message in messages %}<|im_start|>{{ message.role }}\n\
                         {{ message.content[::-1] }}<|im_end|>\n{% endfor %}\
                         {% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";
        assert!(derive_conversation_template(template).is_none());
    }
}
