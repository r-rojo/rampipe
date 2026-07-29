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
use llama_cpp_2::model::{AddBos, LlamaModel};
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
    #[error("prompt plus max_new_tokens ({requested}) exceeds context size ({n_ctx})")]
    PromptTooLong { requested: i32, n_ctx: i32 },
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

        let tokens_list = self.model.str_to_token(prompt, AddBos::Always)?;
        let n_ctx = ctx.n_ctx() as i32;
        let requested = tokens_list.len() as i32 + max_new_tokens;
        if requested > n_ctx {
            return Err(LlamaSessionError::PromptTooLong { requested, n_ctx });
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
        let end = tokens_list.len() as i32 + max_new_tokens;
        let mut text = String::new();
        let mut tokens_generated = 0usize;
        let mut time_to_first_token = None;

        while n_cur <= end {
            let token = sampler.sample(&ctx, batch.n_tokens() - 1);
            sampler.accept(token);

            if time_to_first_token.is_none() {
                time_to_first_token = Some(start.elapsed());
            }

            if self.model.is_eog_token(token) {
                break;
            }

            text.push_str(&self.model.token_to_piece(token, &mut decoder, true, None)?);
            tokens_generated += 1;

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
}
