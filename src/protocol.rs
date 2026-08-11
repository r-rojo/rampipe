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
}

/// One line back per request — an enum (not a bare struct + separate
/// error channel) so a caller can never observe a response that's
/// simultaneously a success and a failure; matches `taskpipe::daemon::
/// QueryResponse`'s own shape.
#[derive(Debug, Serialize, Deserialize)]
pub enum GenerateResponse {
    Ok { text: String, tokens_generated: usize, time_to_first_token_ms: u64 },
    Err { message: String },
}
