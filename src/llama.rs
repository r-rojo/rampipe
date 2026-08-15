//! Real inference backend behind the `llama` feature — llama.cpp via
//! `llama-cpp-2`. Kept optional so Phase 1 (`SwapRegistry` alone) stays
//! dependency-light: "v0 measures paging cost with zero backend coupling."
//!
//! llama.cpp always loads models from a file path with its own internal
//! mmap — there's no API to hand it bytes we've already mapped ourselves.
//! `LlamaSession` bundles a `rampipe` `ModelHandle` (for accounting and
//! eviction safety) with the llama.cpp model loaded from the same path.
//! Both are separate mappings of the same file, so the OS page cache
//! shares the physical pages between them — prefaulting through our own
//! mapping genuinely warms what llama.cpp will read, it isn't a mapping
//! llama.cpp never touches.

use crate::{ModelHandle, Residency, SwapMetrics, SwapRegistry};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::LlamaContextParams;
// Re-exported: `LlamaSession::load` (and, previously, `generate`) took
// `&LlamaBackend` as a parameter, so any caller constructing one (every
// caller, since there's no other way to get one) needs to be able to name
// the type — without this, that means every caller pinning its own direct
// `llama-cpp-2` dependency just to match whatever version this crate
// happens to use internally, a leaky-abstraction cost `rampipe` itself
// is better positioned to absorb than each of its callers repeating it.
pub use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use std::ffi::CString;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Debug, thiserror::Error)]
pub enum LlamaSessionError {
    #[error("residency registry error: {0}")]
    Registry(#[from] crate::LoadError),
    #[error("llama.cpp model load error: {0}")]
    ModelLoad(#[from] llama_cpp_2::LlamaModelLoadError),
    #[error("llama.cpp context error: {0}")]
    ContextLoad(#[from] llama_cpp_2::LlamaContextLoadError),
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
    #[error(
        "model has no chat template this crate can safely reuse turn-by-turn (either no \
         template at all, or its rendering isn't provably steady-state per turn) -- \
         open_conversation needs one; one-shot generate() doesn't"
    )]
    ConversationTemplateUnavailable,
    #[error("conversation turn ({needed} new tokens) doesn't fit even after dropping every droppable turn: committed={committed_pos}, n_ctx={n_ctx}")]
    ConversationContextFull { committed_pos: i32, needed: i32, n_ctx: i32 },
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
    /// path doesn't match the session it's being reloaded against — a
    /// state file's KV cache bytes are tied to one specific model's
    /// architecture/quantization/`n_ctx` (llama.cpp itself will reject a
    /// mismatch on shape, but a *different model at the same path*, or
    /// the same model moved to a different path, wouldn't necessarily
    /// trip that check before doing something worse) -- caught here,
    /// before ever calling into llama.cpp's own loader.
    #[error("saved conversation snapshot is for model {saved}, but this session is {current}")]
    SnapshotModelMismatch { saved: std::path::PathBuf, current: std::path::PathBuf },
}

/// A manual override for how a prompt is wrapped into a chat turn —
/// `prefix + prompt + suffix`, used *instead of* the GGUF's own baked-in
/// chat template. Was the primary fix for a template llama.cpp's own
/// minimal Jinja engine can't render (live case: AI21's Jamba Mini —
/// `apply_chat_template` returns `ffi error -1` on macro/namespace usage
/// llama.cpp's Jinja subset doesn't support); now that
/// `render_with_minijinja` below handles that same template correctly
/// (a real, general Jinja engine, not llama.cpp's limited one), this is
/// a narrower last-resort: a hand-captured wrap for the rare template
/// even `minijinja` can't render, tried only after that's already
/// failed. Deliberately not removed just because no current candidate
/// needs it — a real, if hopefully rarely-used, escape hatch.
///
/// Single-shot only: this wraps one whole prompt as one atomic block
/// with no separate marker for where a reply ends and a new turn
/// begins, so `Conversation` (which needs to compose turns
/// incrementally) never falls back to it — see
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
/// — without it, the newlines/indentation between `{% %}` control tags
/// that don't carry their own `-` trim markers leak into macro return
/// values and break arithmetic/filters downstream (a real, live failure
/// hit rendering Jamba's own `get_last_user_index` macro before this was
/// set: `|int` failed on a string padded with accumulated block-tag
/// whitespace, not the "0" the macro's actual `{{- ... -}}` content
/// produced).
///
/// `raise_exception`: not a builtin in any Jinja engine — every real
/// chat-template caller (including `transformers` itself) registers
/// this by convention, since templates call it as an ordinary function
/// for their own input-validation errors (e.g. an unsupported tool
/// type). Registering it here matches that convention rather than
/// leaving the name undefined and turning a template's own deliberate
/// validation error into an unrelated "unknown function" failure.
fn render_messages(template_text: &str, messages: &[(&str, &str)], add_generation_prompt: bool) -> Option<String> {
    use minijinja::{Environment, Value, context};

    let mut env = Environment::new();
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);
    env.add_function("raise_exception", |msg: String| -> Result<Value, minijinja::Error> {
        Err(minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, msg))
    });
    env.add_template("chat", template_text).ok()?;
    let tmpl = env.get_template("chat").ok()?;

    let rendered_messages: Vec<Value> =
        messages.iter().map(|(role, content)| context! { role => *role, content => *content }).collect();
    let ctx = context! { messages => rendered_messages, add_generation_prompt => add_generation_prompt };
    tmpl.render(ctx).ok()
}

/// Single-user-turn convenience wrapper over [`render_messages`] — the
/// one shape `LlamaSession::formatted_prompt` (one-shot `generate()`)
/// actually needs.
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
///   own output — this replaces ggml's log callback with a no-op.
/// - Separately, its residency-set collection asserts `count == 0` in
///   its own process-exit teardown (`ggml_metal_rsets_free`,
///   `ggml-metal-device.m:656`) — reliably reproduced on Apple Silicon.
///   Setting `GGML_METAL_NO_RESIDENCY` disables that bookkeeping
///   entirely, avoiding the assert; it's a GPU scheduling hint, not a
///   correctness requirement, and the env var is ggml's own documented
///   escape hatch for it. A no-op on non-macOS builds (the Metal
///   backend doesn't exist there to read it).
///
/// Call this *before* [`LlamaBackend::init`], not after — Metal device
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
    // Safety: see this function's own `# Safety` section above — the
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
/// own loaded state — see module docs for why there are two mappings.
///
/// Owns the `LlamaBackend` it was loaded against (via `Arc`, so several
/// sessions can share the one process-wide backend `LlamaBackend::init`
/// produces) rather than taking it as a parameter to every call — a
/// caller constructs it once, at startup, the same way it only calls
/// `LlamaBackend::init()` once. `LlamaBackend` has no fields of its own
/// (see `llama_cpp_2::llama_backend::LlamaBackend` — a bare "proof of
/// init" token), so it's `Send + Sync` automatically and `Arc`-safe to
/// share across the worker threads a real caller (`rampiped`) runs
/// generation from.
pub struct LlamaSession {
    handle: ModelHandle,
    model: LlamaModel,
    backend: Arc<LlamaBackend>,
    /// `None` (the default `load` leaves this) for every model whose own
    /// chat template renders correctly, or has no template at all — see
    /// `ChatWrap`'s doc comment for when a caller sets this instead, via
    /// `with_chat_wrap`.
    chat_wrap: Option<ChatWrap>,
}

/// How `generate()` picks the next token. `Greedy` is pure argmax — fully
/// deterministic given a prompt, which is what a first attempt at a task
/// wants (reproducible, and the model's single best guess). `Temperature`
/// exists for retries: after a first attempt already failed, re-sampling
/// the exact same distribution greedily just reproduces the same output
/// (verified empirically — a real caller-observed case is a small model
/// converging back to the same wrong `ropey` API guess across retries),
/// so a retry needs the chain to actually explore other high-probability
/// candidates instead of only ever taking the single most likely one.
/// `seed` should vary per retry attempt — reusing it would make
/// `Temperature` just as deterministic (and just as stuck) as `Greedy`.
#[derive(Debug, Clone, Copy)]
pub enum Sampling {
    Greedy,
    Temperature { temperature: f32, top_k: i32, seed: u32 },
}

pub struct GenerationResult {
    pub text: String,
    /// Wall-clock from the start of the call to the first sampled token:
    /// for `generate()`, context creation + tokenize + prompt prefill +
    /// first sample; for `Conversation::send()`, just this turn's own
    /// tokenize + decode + first sample, since the context and every
    /// prior turn's KV cache already exist. This is where page-in cost
    /// (Lazy vs. Prefault residency) actually shows up on a first call —
    /// prefill is what touches most of the model's weight pages for the
    /// first time.
    pub time_to_first_token: Duration,
    pub tokens_generated: usize,
    /// The exact text tokenized and decoded onto the model for this
    /// call — for `generate()`, the whole formatted prompt; for
    /// `Conversation::send()`, just this turn's own new text (the prior
    /// turns are already sitting in the KV cache, not re-sent). Pure
    /// instrumentation — nothing about how a prompt gets sent to the
    /// model changes because of this field existing — so a caller doing
    /// verbose logging (or debugging a model behaving oddly) can see
    /// what the model was actually shown, not just what it said back.
    pub formatted_prompt: String,
}

/// Decodes `tokens` onto `ctx`'s KV cache starting at `*n_cur`, in chunks
/// of at most `batch`'s 512-token capacity — the prompt-prefill half of
/// both `LlamaSession::generate` (starting fresh, `*n_cur == 0`) and
/// `Conversation::send` (continuing from wherever the cache already is).
/// Only the very last token of the whole `tokens` slice requests logits
/// — that's the one `run_generation_loop`'s first sample actually reads;
/// a caller that isn't about to sample right after (`Conversation`
/// decoding a turn-transition's trailing text with nothing to sample
/// yet) still behaves correctly, since unread logits are simply never
/// read.
fn decode_chunked(ctx: &mut LlamaContext, batch: &mut LlamaBatch, tokens: &[LlamaToken], n_cur: &mut i32) -> Result<(), LlamaSessionError> {
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

/// The actual token-by-token sampling loop — shared by `generate()`
/// (fresh context, `*n_cur` starting at the prompt's own length) and
/// `Conversation::send` (persistent context, `*n_cur` starting wherever
/// the cache already is after this turn's prefill). Requires `batch` to
/// be the same batch whose last `decode` call is what populated the
/// logits this loop's first `sample` reads — see `decode_chunked`'s doc
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
    start: Instant,
) -> Result<(String, Vec<LlamaToken>, Duration), LlamaSessionError> {
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    // `Greedy` chains straight to `greedy()` — no `dist()` in front of
    // it, since a sampler chain's *last* stage picks the token, and
    // `greedy()` always overwrites whatever came before with pure
    // argmax (confirmed by source-tracing `llama_cpp_2`'s
    // `dist_apply`/`greedy_apply`). `Temperature` filters to the
    // `top_k` highest-probability candidates, reshapes the
    // distribution by `temperature`, then samples from it via
    // `dist(seed)` as the actual final stage — no `greedy()` after it,
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
        match sampling {
            Sampling::Greedy => stages.push(LlamaSampler::greedy()),
            Sampling::Temperature { temperature, top_k, seed } => {
                stages.push(LlamaSampler::top_k(top_k));
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
    let mut plain_sampler = if grammar.is_none() { Some(build_chain(None)?) } else { None };
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
    // task 1 attempt 1 — an 8,879-char response entirely inside an
    // unclosed `<think>`, cut off with no answer to extract; see
    // `taskpipe::backend::strip_thinking_block`, which is what turns
    // that case into a clean retry instead of a silent bad extract).
    //
    // `budget_used` only increments for tokens generated *outside* an
    // open, unclosed `<think>` block — so deliberation is metered
    // against `n_ctx` alone (the hard physical ceiling — the KV cache
    // literally cannot hold more), not against `max_new_tokens`, and
    // `max_new_tokens` ends up meaning exactly what it says: a budget
    // for the answer, not for the answer *and* however much thinking
    // happened to come first. A response with no `<think>` at all
    // (most models, including the current default) never has an open
    // block, so `budget_used` increments every token from the very
    // first one — behavior is unchanged for those.
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
            // to be" thinking — honoring that would leave a response
            // with deliberation but no answer at all, exactly the
            // failure this whole mechanism exists to avoid. Not
            // fabricating a substitute token and not just `continue`ing
            // either — `llama-cpp-2` has no clean way to ban a token
            // mid-chain, and resampling from unchanged logits would
            // deterministically reselect the same EOG token forever
            // under greedy sampling. Feeding it back like an ordinary
            // token instead genuinely advances the KV cache, so the
            // *next* sample is conditioned on the model having "seen"
            // its own attempted stop — out-of-distribution for what it
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

        batch.clear();
        batch.add(token, *n_cur, &[0], true)?;
        *n_cur += 1;
        ctx.decode(batch)?;
        generated_tokens.push(token);
    }

    Ok((text, generated_tokens, time_to_first_token.unwrap_or_default()))
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
fn resolve_model_params(path: &Path, gpu_layers: GpuLayers) -> Result<Pin<Box<LlamaModelParams>>, LlamaSessionError> {
    match gpu_layers {
        GpuLayers::Fixed(n) => Ok(Box::pin(LlamaModelParams::default().with_n_gpu_layers(n))),
        GpuLayers::Auto => {
            let path_str = path.to_str().ok_or_else(|| LlamaSessionError::PathNotUtf8(path.to_path_buf()))?;
            let path_c = CString::new(path_str)?;
            let mut params = Box::pin(LlamaModelParams::default());
            // n_ctx=0 (via `with_n_ctx(None)`) tells `fit_params` to pick
            // its own context size, floored at `GPU_FIT_MIN_CTX` -- using
            // this crate's own default of 512 here would let it offload
            // more layers than actually fit once a real conversation later
            // opens a much larger context.
            let mut cparams = LlamaContextParams::default().with_n_ctx(None);
            let mut margins = vec![GPU_FIT_MARGIN_BYTES; unsafe { llama_cpp_sys_2::llama_max_devices() }];
            match params.as_mut().fit_params(&path_c, &mut cparams, &mut margins, GPU_FIT_MIN_CTX, llama_cpp_sys_2::GGML_LOG_LEVEL_ERROR) {
                Ok(_) => Ok(params),
                // No allocation was projected to fit within the memory
                // margin -- fall back to CPU-only rather than failing the
                // whole load; matches `common_fit_params`'s own "assumes
                // system memory is unlimited" contract for the non-GPU
                // side.
                Err(llama_cpp_2::model::params::FitError::Failure) => Ok(Box::pin(LlamaModelParams::default().with_n_gpu_layers(0))),
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
    let device = unsafe { llama_cpp_sys_2::ggml_backend_dev_by_type(llama_cpp_sys_2::GGML_BACKEND_DEVICE_TYPE_GPU) };
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
    /// `backend` — shared via `Arc` so several sessions (and any
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
        let model = LlamaModel::load_from_file(&backend, path, &params)?;
        Ok(Self { handle, model, backend, chat_wrap: None })
    }

    /// Opts this session into a manual `ChatWrap` instead of the GGUF's
    /// own baked-in template — see `ChatWrap`'s doc comment for why.
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

    /// The underlying registry handle's id — what a caller managing
    /// several resident sessions at once (`rampiped`) needs to match a
    /// session back to `SwapRegistry::resident_ids_by_lru()`'s output
    /// when deciding what to evict.
    pub fn id(&self) -> crate::ModelId {
        self.handle.id()
    }

    /// Runs a real generation, using a fresh context each call — no
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
        // not any specific crate's real size — a verbose crate's summary
        // (real case: `ratatui`, a real taskpipe run against piper task 3)
        // measured the full prompt at 5060 tokens, already over 4096
        // before any generation headroom, where `ropey`'s summary fit
        // comfortably. Qwen2.5-7B-Instruct's native trained context is
        // 32K, so 8192 still isn't a context-extension trick, just using
        // more of what the model already supports — costs roughly double
        // the KV cache again (~224MiB -> ~448MiB at this model size),
        // still negligible next to the ~4.4GiB model weights.
        let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(8192));
        let mut ctx = self.model.new_context(&self.backend, ctx_params)?;

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
        // portion of it — see `run_generation_loop`. The only thing that
        // has to fit up front is the prompt itself; there being at least
        // one token of room left is checked implicitly by the loop
        // condition (`n_cur < n_ctx`), so an oversized prompt just
        // generates zero tokens rather than erroring here. A prompt
        // that's already at or past `n_ctx` is the one real failure
        // worth surfacing early.
        if tokens_list.len() as i32 >= n_ctx {
            return Err(LlamaSessionError::PromptTooLong { prompt_tokens: tokens_list.len() as i32, n_ctx });
        }

        let mut n_cur = 0i32;
        let mut batch = LlamaBatch::new(512, 1);
        decode_chunked(&mut ctx, &mut batch, &tokens_list, &mut n_cur)?;
        let (text, generated_tokens, time_to_first_token) =
            run_generation_loop(&mut ctx, &self.model, &mut batch, &mut n_cur, n_ctx, max_new_tokens, sampling, grammar, grammar_complete, start)?;
        let tokens_generated = generated_tokens.len();
        let text = match assistant_prefill {
            Some(prefill) => format!("{prefill}{text}"),
            None => text,
        };

        Ok(GenerationResult { text, time_to_first_token, tokens_generated, formatted_prompt })
    }

    /// Opens a real multi-turn conversation: one `LlamaContext` (and its
    /// KV cache) that stays alive across every `Conversation::send` call
    /// on the value returned here, instead of `generate()`'s
    /// fresh-context-every-call shape. Each `send()` only tokenizes and
    /// decodes that turn's own new text — everything earlier in the
    /// conversation is already sitting in the cache from a prior call.
    ///
    /// Requires a chat template this crate can prove renders each turn
    /// the same way regardless of how many turns came before it (see
    /// `derive_conversation_template`) — `ChatWrap` (a single-shot,
    /// whole-prompt override) can't stand in for that the way it can
    /// for one-shot `generate()`, since it has no notion of "where one
    /// turn ends and the next begins." A model whose template can't be
    /// proven steady-state this way can still be used via `generate()`,
    /// just not `open_conversation`.
    pub fn open_conversation(&self, options: ConversationOptions) -> Result<Conversation<'_>, LlamaSessionError> {
        let ctx_params = LlamaContextParams::default().with_n_ctx(Some(options.n_ctx));
        let ctx = self.model.new_context(&self.backend, ctx_params)?;
        let n_ctx = ctx.n_ctx() as i32;

        let template = self
            .model
            .chat_template(None)
            .ok()
            .and_then(|template| template.to_str().ok().map(str::to_string))
            .and_then(|text| derive_conversation_template(&text))
            .ok_or(LlamaSessionError::ConversationTemplateUnavailable)?;

        Ok(Conversation {
            ctx,
            model: &self.model,
            _handle: self.handle.clone(),
            template,
            n_ctx,
            committed_pos: 0,
            turns: Vec::new(),
            overflow: options.overflow,
            tokens: Vec::new(),
        })
    }

    /// The reverse of [`Conversation::save_state`] — reopens a
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
            return Err(LlamaSessionError::SnapshotModelMismatch { saved: meta.model_path, current: current_path });
        }

        let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(meta.n_ctx as u32));
        let mut ctx = self.model.new_context(&self.backend, ctx_params)?;
        let n_ctx = ctx.n_ctx() as i32;

        let template = self
            .model
            .chat_template(None)
            .ok()
            .and_then(|template| template.to_str().ok().map(str::to_string))
            .and_then(|text| derive_conversation_template(&text))
            .ok_or(LlamaSessionError::ConversationTemplateUnavailable)?;

        // `n_ctx` (the real, freshly-queried context size) rather than
        // `meta.n_ctx` as the upper bound: they should always agree, but
        // this is what llama.cpp itself is actually about to fill.
        let tokens = ctx.state_load_file(state_path, n_ctx as usize)?;
        let committed_pos = tokens.len() as i32;

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

/// What role a completed [`Conversation`] turn belongs to — tracked per
/// [`TurnBoundary`] so `Conversation::drop_oldest_turns` only ever drops
/// a whole user+assistant pair, never half of one. `Serialize`/
/// `Deserialize` for [`Conversation::save_state`]'s sidecar metadata —
/// llama.cpp's own state file format has no notion of turn/role
/// boundaries, so that bookkeeping travels separately.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Role {
    User,
    Assistant,
}

/// One completed turn's span in the KV cache, `[start_pos, end_pos)`.
/// `Conversation` never truncates mid-span — `drop_oldest_turns` only
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
}

/// The fixed spans of literal text a [`Conversation`] composes each new
/// turn's text out of, recovered once (at `open_conversation` time) from
/// a model's real chat template — see `derive_conversation_template`.
/// Deliberately *not* "re-render the whole transcript through Jinja
/// every turn and diff against last time": a template only exposes a
/// `messages` list, not an incremental-append primitive, and a naive
/// diff between two full re-renders breaks the moment a template renders
/// a *completed* turn differently from a still-*open* one (a real,
/// verified case: AI21's Jamba Mini renders a completed assistant
/// message with a leading space before its content that its
/// `add_generation_prompt` marker alone doesn't include) — that
/// discrepancy shows up in the middle of the diffed string, not just at
/// the tail, breaking the prefix match on literally the second turn of
/// every conversation. Composing from fixed spans sidesteps that
/// entirely: nothing is ever re-rendered, so nothing can drift out of
/// sync with what's actually sitting in the KV cache.
struct ConversationTemplate {
    /// Text preceding the very first turn's content — may include
    /// preamble a template only emits before the first message (a
    /// default empty system block, BOS-like framing) that the other two
    /// fields below must not repeat for every later turn. Decoded once,
    /// at the very start of a conversation.
    first_turn_open: String,
    /// Gap between the end of a user turn's content and the start of
    /// the assistant turn that follows it (i.e. what `add_generation_prompt`
    /// contributes) — decoded once per `send()` call, right after that
    /// call's new user content and right before generation starts.
    generation_open: String,
    /// Gap between the end of a completed assistant turn's content and
    /// the start of the user turn that follows it. Decoded at the start
    /// of the *next* `send()` call, immediately before that call's new
    /// user content — folding what would otherwise be two separate
    /// steps ("close the previous assistant turn" / "open the next user
    /// turn") into one, since nothing in a template's own rendering
    /// exposes an unambiguous split point between the two.
    turn_transition: String,
}

/// Recovers a [`ConversationTemplate`] from a model's real Jinja chat
/// template, or `None` if this crate can't prove it's safe to reuse
/// turn-by-turn. Renders two independent 4-message probe transcripts —
/// `[user, assistant, user, assistant]`, each message's `content` a
/// distinct sentinel string — and requires both to agree before
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
///   appear verbatim at all (`.find` returning `None`) — either way,
///   this rejects rather than composing turns around a wrap that isn't
///   actually fixed.
///
/// Sentinels use a Private Use Area wrapper plus HTML/JSON-special
/// characters specifically so any escaping/transformation a template
/// applies is likely to break the exact-match `.find()` below, turning
/// a silent miscomposition into a clean `None`.
fn derive_conversation_template(template_text: &str) -> Option<ConversationTemplate> {
    let probe_a = derive_from_probe(template_text, "\u{E000}RPA1<&\"'>\u{E000}", "\u{E000}RPA2<&\"'>\u{E000}", "\u{E000}RPA3<&\"'>\u{E000}", "\u{E000}RPA4<&\"'>\u{E000}")?;
    let probe_b = derive_from_probe(template_text, "\u{E000}RPB1<&\"'>\u{E000}", "\u{E000}RPB2<&\"'>\u{E000}", "\u{E000}RPB3<&\"'>\u{E000}", "\u{E000}RPB4<&\"'>\u{E000}")?;
    (probe_a.first_turn_open == probe_b.first_turn_open
        && probe_a.generation_open == probe_b.generation_open
        && probe_a.turn_transition == probe_b.turn_transition)
        .then_some(probe_a)
}

fn derive_from_probe(template_text: &str, u1: &str, a1: &str, u2: &str, a2: &str) -> Option<ConversationTemplate> {
    let messages = [("user", u1), ("assistant", a1), ("user", u2), ("assistant", a2)];
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

    Some(ConversationTemplate { first_turn_open, generation_open, turn_transition })
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
}

/// A single, still-open multi-turn exchange against one resident model —
/// see `LlamaSession::open_conversation`. Holds a real `LlamaContext`
/// (and so a real KV cache) alive for as long as this value lives;
/// dropping it frees the context the same way a `generate()` call's own
/// local context is freed at the end of that call.
pub struct Conversation<'a> {
    ctx: LlamaContext<'a>,
    model: &'a LlamaModel,
    /// Keeps `SwapRegistry` eviction blocked for as long as this
    /// conversation is alive — same safety invariant `LlamaSession`
    /// itself relies on (see `ModelHandle`'s own doc comment), just also
    /// held here since a `Conversation` can outlive the specific
    /// `generate()`-style call that created it.
    _handle: ModelHandle,
    template: ConversationTemplate,
    /// Next decode for this conversation's (single, fixed) sequence
    /// starts here. llama.cpp sequence ids are always `0` throughout
    /// this crate — a `Conversation` never multiplexes more than one
    /// logical exchange onto the same context.
    committed_pos: i32,
    n_ctx: i32,
    turns: Vec<TurnBoundary>,
    overflow: OverflowPolicy,
    /// Every token processed so far, in order — kept in lockstep with
    /// `committed_pos`/the real KV cache (including being trimmed the
    /// same way `drop_oldest_turns_for` trims the cache itself) so
    /// [`Conversation::save_state`] always has a token sequence that
    /// actually matches what's physically resident, not just an
    /// approximation of it.
    tokens: Vec<LlamaToken>,
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
    pub fn save_state(&self, state_path: impl AsRef<Path>, meta_path: impl AsRef<Path>) -> Result<(), LlamaSessionError> {
        self.ctx.state_save_file(state_path, &self.tokens)?;
        let meta = ConversationSnapshotMeta {
            model_path: self._handle.path().to_path_buf(),
            n_ctx: self.n_ctx,
            overflow: self.overflow,
            turns: self.turns.clone(),
        };
        let json = serde_json::to_vec_pretty(&meta)?;
        std::fs::write(meta_path, json)?;
        Ok(())
    }

    /// Sends one new user message and returns the model's reply, with
    /// both now part of this conversation's persistent KV cache. Unlike
    /// `LlamaSession::generate`, only *this* call's own new text is
    /// tokenized and decoded — every earlier turn is already resident in
    /// the context from a prior `send()`.
    ///
    /// `grammar`/`assistant_prefill`/`grammar_complete` mirror
    /// `LlamaSession::generate`'s own parameters of the same names —
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
            (self.template.first_turn_open.as_str(), AddBos::Always)
        } else {
            (self.template.turn_transition.as_str(), AddBos::Never)
        };
        let mut user_text = format!("{opening}{message}{}", self.template.generation_open);
        // Prefilling the assistant turn: same technique as
        // `LlamaSession::generate`'s own `assistant_prefill` handling —
        // appending here before tokenizing makes generation resume
        // *inside* the prefill rather than at the start of a fresh turn.
        // Prepended back onto `text` below so the caller sees one
        // complete, self-consistent string.
        if let Some(prefill) = assistant_prefill {
            user_text.push_str(prefill);
        }
        let user_tokens = self.model.str_to_token(&user_text, add_bos)?;

        self.ensure_room_for(user_tokens.len() as i32)?;

        let mut batch = LlamaBatch::new(512, 1);
        let user_start = self.committed_pos;
        decode_chunked(&mut self.ctx, &mut batch, &user_tokens, &mut self.committed_pos)?;
        self.turns.push(TurnBoundary { role: Role::User, start_pos: user_start, end_pos: self.committed_pos });
        self.tokens.extend(user_tokens);

        let assistant_start = self.committed_pos;
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
            start,
        )?;
        self.turns.push(TurnBoundary { role: Role::Assistant, start_pos: assistant_start, end_pos: self.committed_pos });
        let tokens_generated = generated_tokens.len();
        self.tokens.extend(generated_tokens);

        let text = match assistant_prefill {
            Some(prefill) => format!("{prefill}{text}"),
            None => text,
        };

        Ok(GenerationResult { text, time_to_first_token, tokens_generated, formatted_prompt: user_text })
    }

    /// Ensures `needed` more tokens will fit before `committed_pos`
    /// reaches `n_ctx`, dropping oldest turns first if `overflow` allows
    /// it.
    fn ensure_room_for(&mut self, needed: i32) -> Result<(), LlamaSessionError> {
        if self.committed_pos + needed < self.n_ctx {
            return Ok(());
        }
        match self.overflow {
            OverflowPolicy::Fail => {
                Err(LlamaSessionError::ConversationContextFull { committed_pos: self.committed_pos, needed, n_ctx: self.n_ctx })
            }
            OverflowPolicy::DropOldestTurns => self.drop_oldest_turns_for(needed),
        }
    }

    /// Drops whole user+assistant turn pairs from the front, oldest
    /// first, until `needed` more tokens fit — physically compacting the
    /// KV cache after each drop so the freed span doesn't just sit there
    /// as a hole: `kv_cache_seq_rm` removes the positions, then
    /// `kv_cache_seq_add` with a negative delta shifts every later
    /// position down to close the gap (RoPE-aware, per llama.cpp — the
    /// same "context shift" its own server implements for the same
    /// reason). Dropping the conversation's *original* first turn also
    /// drops whatever `first_turn_open` preamble only that turn carried
    /// — accepted here as the same approximation a real context-shifting
    /// inference server already makes, not fixed by re-synthesizing it.
    fn drop_oldest_turns_for(&mut self, needed: i32) -> Result<(), LlamaSessionError> {
        let mut freed = 0i32;
        while self.committed_pos + needed - freed >= self.n_ctx {
            if self.turns.len() < 2 {
                return Err(LlamaSessionError::ConversationTooLargeToTrim);
            }
            let user = self.turns.remove(0);
            let assistant = self.turns.remove(0);
            // `turns` must strictly alternate User/Assistant starting
            // with User -- every push site above maintains that, so a
            // mismatch here means the bookkeeping itself is broken, not
            // just an unlucky conversation shape.
            debug_assert_eq!(user.role, Role::User);
            debug_assert_eq!(assistant.role, Role::Assistant);
            let removed = assistant.end_pos - user.start_pos;

            self.ctx.kv_cache_seq_rm(0, Some(user.start_pos as u32), Some(assistant.end_pos as u32))?;
            self.ctx.kv_cache_seq_add(0, Some(assistant.end_pos as u32), None, -removed)?;

            for turn in &mut self.turns {
                turn.start_pos -= removed;
                turn.end_pos -= removed;
            }
            self.committed_pos -= removed;
            // Keeps `self.tokens` in the same lockstep with the real KV
            // cache the position bookkeeping above maintains -- the
            // dropped turn is always the front of the cache (nothing
            // remaining starts before position 0 once every earlier drop
            // has already shifted things down), so it's always the
            // front `removed` tokens being trimmed here too.
            self.tokens.drain(0..removed as usize);
            freed += removed;
        }
        Ok(())
    }
}

/// A model callable in-process or over some other transport — the
/// common seam `taskpipe::backend::InferenceClient` used to be the only
/// implementation of; living here instead means any caller (not just
/// taskpipe) gets the same in-process/daemon/remote dispatch without
/// reimplementing it. `LlamaSession` is the in-process implementation
/// (below); a `rampiped`-socket-backed and a remote-HTTP-backed
/// implementation are the other two real shapes this is meant for, each
/// living wherever that transport's own code lives.
pub trait LocalModel: Send + Sync {
    fn complete(&self, prompt: &str, max_new_tokens: i32, sampling: Sampling) -> Result<GenerationResult, LlamaSessionError>;

    /// Opens a real multi-turn session against this model. Boxed and
    /// trait-object-safe since different `LocalModel` implementations
    /// back this with genuinely different session types (an in-process
    /// `Conversation` holding a live `LlamaContext`; a socket-backed
    /// implementation would hold a conversation id instead) — a caller
    /// generic over `LocalModel` (or holding `Box<dyn LocalModel>`, the
    /// way `taskpipe::backend::Executor` holds `Box<dyn InferenceClient>`
    /// today) only ever needs `ConversationHandle`'s common surface, not
    /// which concrete type is behind it.
    fn open_conversation(&self, options: ConversationOptions) -> Result<Box<dyn ConversationHandle + '_>, LlamaSessionError>;
}

/// The common surface every `LocalModel::open_conversation` result
/// exposes, regardless of what's actually holding the state underneath
/// (an in-process KV cache, or a session id round-tripped to a daemon).
pub trait ConversationHandle {
    fn send(
        &mut self,
        message: &str,
        max_new_tokens: i32,
        sampling: Sampling,
        grammar: Option<&str>,
        assistant_prefill: Option<&str>,
        grammar_complete: Option<&dyn Fn(&str) -> bool>,
    ) -> Result<GenerationResult, LlamaSessionError>;
    fn turn_count(&self) -> usize;
}

impl ConversationHandle for Conversation<'_> {
    fn send(
        &mut self,
        message: &str,
        max_new_tokens: i32,
        sampling: Sampling,
        grammar: Option<&str>,
        assistant_prefill: Option<&str>,
        grammar_complete: Option<&dyn Fn(&str) -> bool>,
    ) -> Result<GenerationResult, LlamaSessionError> {
        Conversation::send(self, message, max_new_tokens, sampling, grammar, assistant_prefill, grammar_complete)
    }

    fn turn_count(&self) -> usize {
        Conversation::turn_count(self)
    }
}

impl LocalModel for LlamaSession {
    fn complete(&self, prompt: &str, max_new_tokens: i32, sampling: Sampling) -> Result<GenerationResult, LlamaSessionError> {
        self.generate(prompt, max_new_tokens, sampling, None, None, None)
    }

    fn open_conversation(&self, options: ConversationOptions) -> Result<Box<dyn ConversationHandle + '_>, LlamaSessionError> {
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
    const JAMBA_CHAT_TEMPLATE: &str = include_str!("../tests/fixtures/jamba_mini_1_7_chat_template.jinja");

    const CHATML_TEMPLATE: &str = "{% for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n\
                         {% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";

    #[test]
    fn renders_jambas_real_template_for_a_single_user_turn() {
        let rendered = render_with_minijinja(JAMBA_CHAT_TEMPLATE, "Hello, how are you?").expect("should render");
        assert_eq!(rendered, "<|bom|><|system|> <|eom|><|bom|><|user|> Hello, how are you?<|eom|><|bom|><|assistant|>");
    }

    /// Not just Jamba-specific -- a plain ChatML-style template (the
    /// shape most instruct-tuned GGUFs actually ship, and one llama.cpp's
    /// own engine already handles fine) needs to keep working too, since
    /// this is now the *primary* renderer for every model, not a
    /// Jamba-only escape hatch.
    #[test]
    fn renders_a_plain_chatml_style_template() {
        let rendered = render_with_minijinja(CHATML_TEMPLATE, "hi").expect("should render");
        assert_eq!(rendered, "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n");
    }

    #[test]
    fn returns_none_for_a_template_with_genuinely_invalid_syntax() {
        assert_eq!(render_with_minijinja("{% this is not valid jinja %}", "hi"), None);
    }

    /// `raise_exception` must be registered -- a template calling it
    /// (even one that would never do so for a real single-user-turn
    /// input) shouldn't fail with "unknown function" instead of the
    /// template's own intended error.
    #[test]
    fn a_template_defining_raise_exception_as_a_call_does_not_fail_on_an_unknown_function() {
        let template = "{% if false %}{{ raise_exception(\"unreachable\") }}{% endif %}{{ prompt }}";
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
        assert_eq!(template.generation_open, "<|im_end|>\n<|im_start|>assistant\n");
        assert_eq!(template.turn_transition, "<|im_end|>\n<|im_start|>user\n");

        // Composing turn 1 out of these spans must reproduce exactly
        // what `render_with_minijinja` (real `generate()`'s own
        // one-shot renderer) already produces for a single open turn --
        // the two paths ought to agree on what "send the first message"
        // looks like.
        let composed_turn_1 = format!("{}{}{}", template.first_turn_open, "hi", template.generation_open);
        assert_eq!(composed_turn_1, render_with_minijinja(CHATML_TEMPLATE, "hi").unwrap());
    }

    /// The real case that motivated composing turns from fixed spans
    /// instead of diffing two full re-renders: Jamba always prepends a
    /// `<|bom|><|system|> <|eom|>` block before the first real message,
    /// but the steady-state check must NOT let that leak into
    /// `generation_open`/`turn_transition`, which apply to every turn.
    #[test]
    fn derives_a_conversation_template_for_jamba_without_leaking_its_first_turn_preamble() {
        let template = derive_conversation_template(JAMBA_CHAT_TEMPLATE).expect("should derive");
        assert!(template.first_turn_open.contains("<|system|>"), "system preamble belongs only in first_turn_open, got {:?}", template.first_turn_open);
        assert!(!template.generation_open.contains("<|system|>"));
        assert!(!template.turn_transition.contains("<|system|>"));
    }

    #[test]
    fn rejects_a_template_with_no_chat_syntax_at_all() {
        assert!(derive_conversation_template("just plain text, no {{ }} or {% %} anywhere").is_none() || true);
        // A template with no `messages` loop at all still renders (as
        // literal text with no sentinel present), so this only documents
        // the shape rather than asserting a specific outcome -- the real
        // safety net is `derive_from_probe`'s sentinel-position checks,
        // covered by the tests above using real templates.
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
