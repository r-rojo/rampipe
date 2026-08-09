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
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use std::num::NonZeroU32;
use std::path::Path;
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
}

/// A model resident in both `rampipe`'s accounting mmap and llama.cpp's
/// own loaded state — see module docs for why there are two mappings.
pub struct LlamaSession {
    handle: ModelHandle,
    model: LlamaModel,
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
    /// Wall-clock from the start of `generate()` to the first sampled
    /// token: context creation + tokenize + prompt prefill + first sample.
    /// This is where page-in cost (Lazy vs. Prefault residency) actually
    /// shows up — prefill is what touches most of the model's weight
    /// pages for the first time.
    pub time_to_first_token: Duration,
    pub tokens_generated: usize,
}

impl LlamaSession {
    /// Loads `path` into `registry` (for residency accounting/eviction
    /// safety) and into llama.cpp (for actual inference).
    pub fn load(
        registry: &SwapRegistry,
        backend: &LlamaBackend,
        path: impl AsRef<Path>,
        residency: Residency,
    ) -> Result<Self, LlamaSessionError> {
        let path = path.as_ref();
        let handle = registry.load(path, residency)?;
        let model = LlamaModel::load_from_file(backend, path, &LlamaModelParams::default())?;
        Ok(Self { handle, model })
    }

    /// Residency metrics from the `rampipe` side of this session (map
    /// latency, prefault latency, RSS delta, mapped bytes, warm flag).
    pub fn metrics(&self) -> SwapMetrics {
        self.handle.metrics()
    }

    pub fn path(&self) -> &Path {
        self.handle.path()
    }

    /// Runs a real generation, using greedy sampling over a fresh context
    /// each call. Attributes time-to-first-token separately from total
    /// generation time so page-in cost (see `metrics()`) can be correlated
    /// against it.
    pub fn generate(
        &self,
        backend: &LlamaBackend,
        prompt: &str,
        max_new_tokens: i32,
        sampling: Sampling,
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
        let mut ctx = self.model.new_context(backend, ctx_params)?;

        let formatted_prompt = self.formatted_prompt(prompt)?;
        let tokens_list = self.model.str_to_token(&formatted_prompt, AddBos::Always)?;
        let n_ctx = ctx.n_ctx() as i32;
        // Not `+ max_new_tokens` any more: `max_new_tokens` no longer
        // bounds the whole generation, only the metered (non-thinking)
        // portion of it — see the loop below. The only thing that has to
        // fit up front is the prompt itself; there being at least one
        // token of room left is checked implicitly by the loop condition
        // (`n_cur < n_ctx`), so an oversized prompt just generates zero
        // tokens rather than erroring here. A prompt that's already at or
        // past `n_ctx` is the one real failure worth surfacing early.
        if tokens_list.len() as i32 >= n_ctx {
            return Err(LlamaSessionError::PromptTooLong { prompt_tokens: tokens_list.len() as i32, n_ctx });
        }

        // Prompt decode in chunks of at most the batch's token capacity
        // (512): `LlamaBatch::add` only has room for 512 tokens per
        // `decode` call regardless of `n_ctx`, so a prompt longer than
        // that (already allowed by the `n_ctx`-only check above) needs
        // multiple decode calls. Only the very last token of the whole
        // prompt requests logits — that's the one sampling starts from.
        let mut batch = LlamaBatch::new(512, 1);
        let last_index = (tokens_list.len() - 1) as i32;
        for chunk_start in (0..tokens_list.len()).step_by(512) {
            let chunk_end = (chunk_start + 512).min(tokens_list.len());
            batch.clear();
            for (offset, &token) in tokens_list[chunk_start..chunk_end].iter().enumerate() {
                let i = (chunk_start + offset) as i32;
                batch.add(token, i, &[0], i == last_index)?;
            }
            ctx.decode(&mut batch)?;
        }

        let mut decoder = encoding_rs::UTF_8.new_decoder();
        // `Greedy` chains straight to `greedy()` — no `dist()` in front of
        // it, since a sampler chain's *last* stage picks the token, and
        // `greedy()` always overwrites whatever came before with pure
        // argmax (confirmed by source-tracing `llama_cpp_2`'s
        // `dist_apply`/`greedy_apply`). `Temperature` filters to the
        // `top_k` highest-probability candidates, reshapes the
        // distribution by `temperature`, then samples from it via
        // `dist(seed)` as the actual final stage — no `greedy()` after it,
        // so the sampled draw is what's actually used.
        let mut sampler = match sampling {
            Sampling::Greedy => LlamaSampler::chain_simple([LlamaSampler::greedy()]),
            Sampling::Temperature { temperature, top_k, seed } => {
                LlamaSampler::chain_simple([LlamaSampler::top_k(top_k), LlamaSampler::temp(temperature), LlamaSampler::dist(seed)])
            }
        };

        // Not `batch.n_tokens()`: the batch was cleared and refilled per
        // chunk above, so it only reflects the size of the *last* chunk,
        // not the full prompt — using it here would make the generation
        // loop's positions diverge from the KV cache's actual last
        // position by the size of every chunk before the final one.
        let mut n_cur = tokens_list.len() as i32;
        let mut text = String::new();
        let mut tokens_generated = 0usize;
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
            if n_cur >= n_ctx || budget_used >= max_new_tokens {
                break;
            }

            let token = sampler.sample(&ctx, batch.n_tokens() - 1);
            sampler.accept(token);

            if time_to_first_token.is_none() {
                time_to_first_token = Some(start.elapsed());
            }

            let inside_open_think = text.contains("<think>") && !text.contains("</think>");

            if self.model.is_eog_token(token) {
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
                batch.add(token, n_cur, &[0], true)?;
                n_cur += 1;
                ctx.decode(&mut batch)?;
                continue;
            }

            text.push_str(&self.model.token_to_piece(token, &mut decoder, true, None)?);
            tokens_generated += 1;

            if !inside_open_think {
                budget_used += 1;
            }

            batch.clear();
            batch.add(token, n_cur, &[0], true)?;
            n_cur += 1;
            ctx.decode(&mut batch)?;
        }

        Ok(GenerationResult {
            text,
            time_to_first_token: time_to_first_token.unwrap_or_default(),
            tokens_generated,
        })
    }

    /// Wraps `prompt` in the model's own baked-in chat template as a
    /// single "user" turn (`add_ass: true`, so the rendered text ends
    /// with the assistant turn already opened, ready for generation to
    /// continue into it) before tokenizing. Previously `generate` fed the
    /// raw instructional text straight to the tokenizer with no role
    /// structure at all -- `llama_cpp_2`'s own doc comment on
    /// `apply_chat_template` warns that skipping this "can result in
    /// really unexpected responses," and a real caller-observed case
    /// (Qwen3.6-35B-A3B leaving a referenced type undefined, one attempt
    /// producing no parseable output at all) matched that failure shape.
    /// Not every GGUF has a template baked in, though -- falls back to
    /// the untouched raw prompt on `MissingTemplate` rather than hard
    /// erroring, since that was the only behavior available before this
    /// existed and is still strictly better than refusing to run.
    fn formatted_prompt(&self, prompt: &str) -> Result<String, LlamaSessionError> {
        let template = match self.model.chat_template(None) {
            Ok(template) => template,
            Err(llama_cpp_2::ChatTemplateError::MissingTemplate) => return Ok(prompt.to_string()),
            Err(other) => return Err(other.into()),
        };
        let message = LlamaChatMessage::new("user".to_string(), prompt.to_string())?;
        Ok(self.model.apply_chat_template(&template, &[message], true)?)
    }
}
