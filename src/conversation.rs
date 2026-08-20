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
    Greedy,
    Temperature {
        temperature: f32,
        top_k: i32,
        seed: u32,
    },
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
    /// The exact text tokenized and decoded onto the model for this
    /// call -- for `generate()`, the whole formatted prompt; for
    /// `Conversation::send()`, just this turn's own new text (the prior
    /// turns are already sitting in the KV cache, not re-sent). Pure
    /// instrumentation -- nothing about how a prompt gets sent to the
    /// model changes because of this field existing -- so a caller doing
    /// verbose logging (or debugging a model behaving oddly) can see
    /// what the model was actually shown, not just what it said back.
    pub formatted_prompt: String,
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
    fn turn_count(&self) -> usize;
}
