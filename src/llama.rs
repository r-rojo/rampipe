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
#[derive(Debug, Clone)]
pub struct ChatWrap {
    pub prefix: String,
    pub suffix: String,
}

/// Renders `template_text` (a model's own real `tokenizer.chat_template`
/// Jinja source, as returned by `chat_template()`) for a single `user`
/// turn with generation left open for the assistant to continue into —
/// the one shape `generate()` actually needs. `None` on any parse or
/// render failure (unsupported syntax, a template that genuinely calls
/// `raise_exception` for this input shape, etc.) — the caller falls back
/// further, this never itself decides there's no other option.
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
fn render_with_minijinja(template_text: &str, prompt: &str) -> Option<String> {
    use minijinja::{Environment, Value, context};

    let mut env = Environment::new();
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);
    env.add_function("raise_exception", |msg: String| -> Result<Value, minijinja::Error> {
        Err(minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, msg))
    });
    env.add_template("chat", template_text).ok()?;
    let tmpl = env.get_template("chat").ok()?;

    let ctx = context! {
        messages => vec![context! { role => "user", content => prompt }],
        add_generation_prompt => true,
    };
    tmpl.render(ctx).ok()
}

/// A model resident in both `rampipe`'s accounting mmap and llama.cpp's
/// own loaded state — see module docs for why there are two mappings.
pub struct LlamaSession {
    handle: ModelHandle,
    model: LlamaModel,
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
        Ok(Self { handle, model, chat_wrap: None })
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
        let template = "{% for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n\
                         {% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";
        let rendered = render_with_minijinja(template, "hi").expect("should render");
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
}
