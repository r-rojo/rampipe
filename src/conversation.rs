//! The backend-agnostic conversation seam: the trait a multi-turn
//! conversation is driven through, and the small value types that
//! trait's surface is written in.
//!
//! Lives here rather than in `crate::llama` because it is the one
//! part of that module a build without the `llama` feature still needs
//! to name. A daemon-only host -- a GUI panel talking to an already-
//! running `rampiped` over its socket -- drives
//! `crate::client::RampipedConversation` through
//! [`ConversationHandle`] and has no use for an in-process model, but
//! while the trait sat inside the `llama`-gated module there was no way
//! to say so: a `client`-only build had no name for the thing
//! `RampipedConversation` implements, and so had to compile llama.cpp
//! to get one. Moving the seam out is what makes `llama` genuinely
//! optional for a caller, not just for this crate's own internals.

use std::time::Duration;

/// How `generate()` picks the next token. `Greedy` is pure argmax -- fully
/// deterministic given a prompt, which is what a first attempt at a task
/// wants (reproducible, and the model's single best guess). `Temperature`
/// exists for retries: after a first attempt already failed, re-sampling
/// the exact same distribution greedily just reproduces the same output
/// (verified empirically -- a real caller-observed case is a small model
/// converging back to the same wrong `ropey` API guess across retries),
/// so a retry needs the chain to actually explore other high-probability
/// candidates instead of only ever taking the single most likely one.
/// `seed` should vary per retry attempt -- reusing it would make
/// `Temperature` just as deterministic (and just as stuck) as `Greedy`.
#[derive(Debug, Clone, Copy)]
pub enum Sampling {
    Greedy {
        penalties: Penalties,
    },
    Temperature {
        temperature: f32,
        top_k: i32,
        /// Nucleus sampling: keep the smallest set of candidates whose
        /// probabilities sum to this. `1.0` disables it.
        ///
        /// Added because it was not expressible, and the models this
        /// serves ask for it by name -- Qwen3-Coder's own card says
        /// `temperature=0.7, top_p=0.8, top_k=20`. Two of those three
        /// could be set and the third could not, so "run it the way its
        /// card says" was not a thing this crate could do.
        top_p: f32,
        /// Floor on a candidate's probability relative to the most
        /// likely one. `0.0` disables it. Qwen3's guidance pairs it with
        /// the above as `min_p=0`, which is the disabled value -- but a
        /// setting a model asks for explicitly should be sayable even
        /// when what it asks for is "off".
        min_p: f32,
        seed: u32,
        penalties: Penalties,
    },
}

/// Discourages repeating tokens already seen recently in this turn's own
/// context -- independent of which final-selection strategy `Sampling`
/// picks, since it adjusts logits *before* greedy argmax or temperature/
/// dist ever runs, so even pure `Greedy` decoding is affected. That
/// matters concretely: greedy is exactly the shape most prone to
/// repetition (no randomness to escape a self-reinforcing loop once one
/// starts) -- confirmed live, a real manager turn against Qwen 3.8
/// generated a single shell command that degenerated into the same four
/// `find`/`cat`/`ls` invocations repeated dozens of times, burning its
/// entire token budget before ever finishing, using plain `Greedy` with
/// no penalty applied at all.
///
/// Field meanings and "disabled" values match llama.cpp's own
/// `llama_sampler_init_penalties` directly -- see that function's own
/// doc comment in `llama_cpp_2::sampling::LlamaSampler::penalties`.
/// `Default` is fully disabled, so a caller that never opts in gets
/// byte-identical behavior to before this existed.
#[derive(Debug, Clone, Copy)]
pub struct Penalties {
    /// How many of the most recent tokens count toward the penalty (0 =
    /// disabled, -1 = the whole context).
    pub last_n: i32,
    /// 1.0 = disabled.
    pub repeat: f32,
    /// 0.0 = disabled.
    pub freq: f32,
    /// 0.0 = disabled.
    pub present: f32,
}

impl Default for Penalties {
    fn default() -> Self {
        Self {
            last_n: 0,
            repeat: 1.0,
            freq: 0.0,
            present: 0.0,
        }
    }
}

pub struct GenerationResult {
    pub text: String,
    /// Wall-clock from the start of the call to the first sampled token:
    /// for `generate()`, context creation + tokenize + prompt prefill +
    /// first sample; for `Conversation::send()`, just this turn's own
    /// tokenize + decode + first sample, since the context and every
    /// prior turn's KV cache already exist. This is where page-in cost
    /// (Lazy vs. Prefault residency) actually shows up on a first call --
    /// prefill is what touches most of the model's weight pages for the
    /// first time.
    pub time_to_first_token: Duration,
    pub tokens_generated: usize,
    /// How much of the context window this conversation now occupies,
    /// and how big it is.
    ///
    /// Reported per turn because it was not, and that absence cost real
    /// time. A model repeatedly collapsed into repeated tokens partway
    /// through agentic runs, and "the context is filling up" was
    /// proposed as the cause three separate times without anyone being
    /// able to see the number -- the daemon tracked it and only ever
    /// mentioned it in the text of an overflow error.
    ///
    /// A caller that can watch this can tell a context problem from a
    /// sampling problem by looking, which is the whole difference
    /// between measuring and guessing.
    pub committed_tokens: usize,
    pub context_size: usize,
    /// The exact text tokenized and decoded onto the model for this
    /// call -- for `generate()`, the whole formatted prompt; for
    /// `Conversation::send()`, just this turn's own new text (the prior
    /// turns are already sitting in the KV cache, not re-sent). Pure
    /// instrumentation -- nothing about how a prompt gets sent to the
    /// model changes because of this field existing -- so a caller doing
    /// verbose logging (or debugging a model behaving oddly) can see
    /// what the model was actually shown, not just what it said back.
    pub formatted_prompt: String,
    /// Tool calls decoded out of `text`, when this conversation was
    /// opened with tools *and* the model's own template yielded a
    /// derivable call format (see `crate::tool_format`).
    ///
    /// Always empty otherwise -- including for a conversation that
    /// offered tools and simply got a prose answer, which is the
    /// ordinary "the model chose to reply rather than act" case and not
    /// distinguishable from "no tools offered" by looking here. `text`
    /// is left completely untouched either way: the raw generation is
    /// what a caller logs and what a grammar-based fallback still
    /// parses, so this is strictly additive information rather than a
    /// replacement for it.
    pub tool_calls: Vec<crate::protocol::ToolCall>,
    /// Generation stopped part-way into a tool call rather than because
    /// the model finished -- see `crate::tool_format::ends_mid_call`.
    ///
    /// Computed here, where the model's derived call format lives,
    /// rather than by the caller: a daemon-backed client deliberately
    /// does not link the template machinery (that is the whole point of
    /// the `client` feature), so it could not work this out itself.
    pub truncated_tool_call: bool,
}

/// What a [`ConversationHandle`] call can fail with, across every
/// backend rather than any one of them.
///
/// Separate from `crate::llama::LlamaSessionError` because that type
/// names `llama_cpp_2`'s own error types in about twenty of its
/// variants and therefore cannot exist at all without the `llama`
/// feature -- which is precisely what kept this trait trapped inside
/// the gated module. The concrete types keep their rich errors on their
/// *inherent* methods (`llama::Conversation::send` still returns a
/// `LlamaSessionError`; `client::RampipedConversation::send` still
/// returns a `RampipedError`); this is the vocabulary a caller who
/// deliberately does not know which backend it got can actually act on.
#[derive(Debug, thiserror::Error)]
pub enum ConversationError {
    /// A non-in-process implementation's own transport/remote failure,
    /// flattened to a message. Replaces the `LlamaSessionError::Backend`
    /// variant that existed for exactly this and is now unnecessary.
    #[error("conversation backend error: {0}")]
    Backend(String),
    /// An in-process llama.cpp failure, carried through structurally
    /// rather than stringified, so a caller that *does* link `llama` can
    /// still match on the underlying cause.
    #[cfg(feature = "llama")]
    #[error(transparent)]
    Llama(#[from] crate::llama::LlamaSessionError),
}

/// The common surface every `LocalModel::open_conversation` result
/// exposes, regardless of what's actually holding the state underneath
/// (an in-process KV cache, or a session id round-tripped to a daemon).
///
/// `grammar_completion` is the *serializable* [`crate::protocol::
/// GrammarCompletion`] rather than the `&dyn Fn(&str) -> bool` predicate
/// `Conversation::send` itself takes, because a closure can't cross a
/// socket: any implementation that isn't in-process has to receive this
/// as data. The in-process implementation in `crate::llama` converts
/// it back into a predicate with
/// [`crate::protocol::GrammarCompletion::into_predicate`], which is
/// exactly what every caller was already doing by hand at its own call
/// site.
pub trait ConversationHandle {
    fn send(
        &mut self,
        message: &str,
        max_new_tokens: i32,
        sampling: Sampling,
        grammar: Option<&str>,
        assistant_prefill: Option<&str>,
        grammar_completion: Option<crate::protocol::GrammarCompletion>,
    ) -> Result<GenerationResult, ConversationError>;
    /// Whether tool calls emitted in this conversation can actually be
    /// decoded -- true only when tools were offered at open time *and*
    /// the backend can parse this model's call format.
    ///
    /// A caller uses this to choose between the tool-calling path and
    /// its own prompt-and-grammar arrangement. It deliberately answers
    /// a question an empty `GenerationResult::tool_calls` cannot: a turn
    /// with no calls is the ordinary "the model chose to answer rather
    /// than act" case, indistinguishable from "calls are never coming."
    ///
    /// Defaults to `false` so an implementation predating tool calling
    /// keeps compiling and correctly reports that it has none.
    fn supports_tool_calls(&self) -> bool {
        false
    }

    /// Feeds executed tool results back and generates the next turn --
    /// see `crate::llama::Conversation::send_tool_results`, which this
    /// mirrors.
    ///
    /// Defaults to an error rather than silently falling back to
    /// [`ConversationHandle::send`]: an implementation that cannot do
    /// this also returns `false` from `supports_tool_calls`, so a
    /// caller reaching here has ignored that and wants to know, not to
    /// have its results quietly reshaped into user text.
    fn send_tool_results(
        &mut self,
        _results: &[String],
        _max_new_tokens: i32,
        _sampling: Sampling,
        _grammar: Option<&str>,
        _grammar_completion: Option<crate::protocol::GrammarCompletion>,
    ) -> Result<GenerationResult, ConversationError> {
        Err(ConversationError::Backend(
            "this conversation backend does not support tool results".to_string(),
        ))
    }

    fn turn_count(&self) -> usize;
    /// Persists this conversation's KV cache to `state_path`/`meta_path`
    /// and, on success, leaves it unusable for anything further -- a
    /// caller must treat any subsequent `send()` as a bug, not a
    /// retryable failure (the in-process implementation could technically
    /// still be sent to afterward; the daemon-backed one physically
    /// can't, since `rampiped` closes the connection right after
    /// replying -- this trait's contract is the stricter of the two, so
    /// a caller holding a bare `Box<dyn ConversationHandle>` gets the
    /// same rule regardless of which backend it actually got).
    /// `&mut self` rather than consuming `self`: a `Box<dyn
    /// ConversationHandle>` can't cleanly offer a by-value trait method
    /// through a trait object, and the caller-discipline cost of "don't
    /// use this afterward" is the same either way.
    fn snapshot(
        &mut self,
        state_path: &std::path::Path,
        meta_path: &std::path::Path,
    ) -> Result<(), ConversationError>;
}
