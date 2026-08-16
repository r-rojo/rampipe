//! `rampiped`: a small daemon that holds GGUF model(s) resident in
//! memory (CPU/Metal) and serves generation requests from any number of
//! local client processes over a Unix socket.
//!
//! This exists to structurally fix a real, repeatedly-observed failure:
//! two separate local processes (a running `taskpipe` task and a harness
//! comparison run) each mapping and loading their own full copy of the
//! same multi-GB model into the same machine's Metal memory pool
//! crashed with `kIOGPUCommandBufferCallbackErrorOutOfMemory`, twice, in
//! one evening. One process holding the model, serving every local
//! client, removes the failure mode entirely instead of relying on the
//! convention "don't run two things at once."
//!
//! Wire protocol: `rampipe::protocol` (one JSON value + `\n` per
//! message over the socket), matching `taskpipe::daemon`'s own
//! established local convention rather than a new shared IPC crate for
//! ~100 lines of socket boilerplate.
//!
//! Connection handling is concurrent (one thread per connection); the
//! actual GPU-touching work — deciding what to load/evict and running a
//! generation — is not: it all happens inside one critical section,
//! guarded by `SharedState`'s own `Mutex`, so at most one load, evict, or
//! decode is ever in flight process-wide. A GPU can only usefully run one
//! decode at a time anyway, and loading a second model onto it while a
//! decode is using it risks the same kind of memory pressure this daemon
//! exists to avoid — so this is a deliberate choice, not a missing
//! optimization: real gain is a slow-to-send or slow-to-receive client no
//! longer blocking every *other* client's request while its own
//! connection is merely open (the previous, fully single-threaded
//! version's actual limitation), and eviction never touching a model
//! that's mid-generation for someone else, enforced structurally (the
//! whole load-then-generate turn is one atomic critical section) rather
//! than true only by accident of nothing else being able to run at all.
//!
//!     cargo run --release --features llama --bin rampiped -- \
//!         [--socket <path>] [--budget-fraction <0.0-1.0>]

use anyhow::{Context, Result, bail};
use llama_cpp_2::llama_backend::LlamaBackend;
use rampipe::llama::{Conversation, ConversationOptions, LlamaSession, OverflowPolicy, Sampling};
use rampipe::protocol::{
    ClientMessage, ConversationResponse, ConversationTurnRequest, GenerateRequest, GenerateResponse, OpenConversationRequest,
    WireOverflowPolicy, WireSampling,
};
use rampipe::{ModelId, Residency, SwapRegistry};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::num::NonZeroU32;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

const DEFAULT_BUDGET_FRACTION: f64 = 0.8;

const USAGE: &str = "\
rampiped: a daemon that holds GGUF model(s) resident in memory and
serves generation requests from local clients over a Unix socket.

USAGE:
    rampiped [--socket <path>] [--budget-fraction <0.0-1.0>]

OPTIONS:
    --socket <path>            Unix socket to listen on
                                (default: ~/.rampipe/rampiped.sock)
    --budget-fraction <float>  fraction of available memory the swap
                                registry may use before evicting
                                (default: 0.8)
    -h, --help                 print this message and exit";

enum ParsedArgs {
    Help,
    Run(Args),
}

struct Args {
    socket: PathBuf,
    budget_fraction: f64,
}

fn take_flag_value(args: &mut Vec<String>, flag: &str) -> Result<Option<String>> {
    let Some(pos) = args.iter().position(|a| a == flag) else { return Ok(None) };
    if pos + 1 >= args.len() {
        bail!("{flag} requires a value");
    }
    args.remove(pos);
    Ok(Some(args.remove(pos)))
}

fn parse_args() -> Result<ParsedArgs> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    if let Some(pos) = args.iter().position(|a| a == "-h" || a == "--help") {
        args.remove(pos);
        return Ok(ParsedArgs::Help);
    }
    let socket = take_flag_value(&mut args, "--socket")?.map(PathBuf::from);
    let budget_fraction = take_flag_value(&mut args, "--budget-fraction")?
        .map(|s| s.parse::<f64>().map_err(|_| anyhow::anyhow!("--budget-fraction must be a number, got {s:?}")))
        .transpose()?
        .unwrap_or(DEFAULT_BUDGET_FRACTION);
    if !args.is_empty() {
        bail!("unrecognized argument(s): {}", args.join(" "));
    }
    let socket = match socket {
        Some(path) => path,
        // `rampipe::protocol::default_socket_path` -- the one shared
        // source of truth every client falls back to as well (see that
        // function's own doc comment), not a value re-derived here.
        None => rampipe::protocol::default_socket_path().context("--socket not given, and could not determine a default (~/.rampipe/rampiped.sock) -- HOME not set")?,
    };
    Ok(ParsedArgs::Run(Args { socket, budget_fraction }))
}

fn wire_sampling_to_sampling(sampling: WireSampling) -> Sampling {
    match sampling {
        WireSampling::Greedy => Sampling::Greedy,
        WireSampling::Temperature { temperature, top_k, seed } => Sampling::Temperature { temperature, top_k, seed },
    }
}

fn wire_overflow_to_overflow(overflow: WireOverflowPolicy) -> OverflowPolicy {
    match overflow {
        WireOverflowPolicy::Fail => OverflowPolicy::Fail,
        WireOverflowPolicy::DropOldestTurns => OverflowPolicy::DropOldestTurns,
    }
}

/// Everything a load, evict, or generate call touches — bundled into one
/// struct specifically so it can live behind one `Mutex` (see module doc
/// comment). `backend` lives in here too, not as a separate `&LlamaBackend`
/// passed alongside: `llama_cpp_2::llama_backend::LlamaBackend` is
/// expected to be `Send` but not necessarily `Sync` (see brush's `aish`
/// builtin, `ai.rs::ModelState`'s own doc comment, which this mirrors) —
/// putting it behind the same `Mutex` as everything else means only
/// `Send` is ever required, never `Sync`, and it's never touched from two
/// threads at once by construction.
struct SharedState {
    backend: Arc<LlamaBackend>,
    registry: SwapRegistry,
    /// `Arc`-wrapped, not a bare `LlamaSession`, so a conversation
    /// connection's own thread (see `handle_conversation`) can clone one
    /// out and keep it alive on its own stack for the conversation's
    /// whole lifetime, independent of whatever this map does to its own
    /// entry afterward -- the same `ModelHandle`-pinning mechanism
    /// `make_room_for`'s eviction skip (below) already relies on.
    sessions: HashMap<PathBuf, Arc<LlamaSession>>,
    budget_fraction: f64,
    /// This process's own executable path and its mtime *at the moment
    /// this struct was built* (i.e. daemon startup) — not re-derived on
    /// every `Status` request, which would just reflect whatever
    /// happens to be on disk right now regardless of what code this
    /// already-running process actually loaded. See
    /// `protocol::StatusResponse::exe_modified_unix_secs`'s own doc
    /// comment for why that distinction is the entire point of this
    /// field.
    exe_snapshot: (Option<PathBuf>, Option<u64>),
    started_at: Instant,
    requests_served: u64,
    requests_failed: u64,
    total_tokens_generated: u64,
    /// Per-model counters, keyed the same way `sessions` is and with the
    /// same lifetime: removed in `make_room_for` whenever its `sessions`
    /// entry is evicted, so these describe the model's *current*
    /// residency, not a lifetime history that would otherwise persist
    /// across an evict-and-later-reload as if it were one continuous stay.
    model_stats: HashMap<PathBuf, ModelStats>,
}

#[derive(Default)]
struct ModelStats {
    requests_served: u64,
    tokens_generated: u64,
    last_used_unix_secs: u64,
}

/// `std::env::current_exe()` plus that file's own mtime, in Unix
/// seconds — best-effort: `None` for either half just means this
/// platform or environment didn't have an answer, not an error worth
/// failing startup over.
fn current_exe_snapshot() -> (Option<PathBuf>, Option<u64>) {
    let Ok(path) = std::env::current_exe() else { return (None, None) };
    let modified = std::fs::metadata(&path)
        .and_then(|meta| meta.modified())
        .ok()
        .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|duration| duration.as_secs());
    (Some(path), modified)
}

impl SharedState {
    fn new(backend: LlamaBackend, budget_fraction: f64) -> Self {
        Self {
            backend: Arc::new(backend),
            registry: SwapRegistry::new(),
            sessions: HashMap::new(),
            budget_fraction,
            exe_snapshot: current_exe_snapshot(),
            started_at: Instant::now(),
            requests_served: 0,
            requests_failed: 0,
            total_tokens_generated: 0,
            model_stats: HashMap::new(),
        }
    }

    /// Records the outcome of one request/turn against `path` -- the one
    /// place every success/failure counter (daemon-level and per-model)
    /// gets updated, called with the state lock already held by the
    /// caller (matching every other mutation on this type).
    fn record_outcome(&mut self, path: &Path, tokens_generated: Option<usize>) {
        match tokens_generated {
            Some(tokens) => {
                self.requests_served += 1;
                self.total_tokens_generated += tokens as u64;
                let stats = self.model_stats.entry(path.to_path_buf()).or_default();
                stats.requests_served += 1;
                stats.tokens_generated += tokens as u64;
                stats.last_used_unix_secs =
                    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
            }
            None => self.requests_failed += 1,
        }
    }

    /// Ensures `path` is resident in `self.sessions`, loading it (and
    /// evicting, if the budget requires it) if it isn't already. Returns
    /// nothing — deliberately not `&LlamaSession`, since a reference tied
    /// to `&mut self` here would keep the whole `SharedState` borrowed
    /// mutably for as long as the caller holds it, blocking the caller
    /// from also borrowing `self.backend` immutably alongside
    /// `self.sessions.get(path)` for the actual `generate()` call right
    /// after. Two separate immutable field borrows once this returns is
    /// simpler than threading a reference through.
    fn ensure_loaded(&mut self, path: &Path) -> Result<()> {
        if self.sessions.contains_key(path) {
            // A cache hit still needs to count as an access for LRU
            // purposes -- `SwapRegistry::load` already treats a
            // dedup-hit against an already-resident path as a real
            // touch (see `RegistryState::last_accessed`'s doc comment),
            // so re-calling it here (cheap: no new mmap, no reload) is
            // what keeps `resident_ids_by_lru` honest.
            self.registry.load(path, Residency::Lazy).context("touching cached model for LRU accounting")?;
            return Ok(());
        }

        self.make_room_for(path)?;

        eprintln!("rampiped: loading {}", path.display());
        let load_start = Instant::now();
        let session = LlamaSession::load(&self.registry, Arc::clone(&self.backend), path, Residency::Lazy)
            .with_context(|| format!("loading model {}", path.display()))?;
        eprintln!("rampiped: loaded {} in {:?} (now {} model(s) resident, {} bytes mapped)",
            path.display(), load_start.elapsed(), self.sessions.len() + 1, self.registry.mapped_bytes());
        self.sessions.insert(path.to_path_buf(), Arc::new(session));
        Ok(())
    }

    /// Evicts least-recently-used resident models, one at a time, until
    /// loading `new_size_bytes` more would stay within budget.
    ///
    /// Walks the LRU list (not just the single oldest entry) and skips
    /// any session a live conversation still holds its own `Arc` clone
    /// of (`Arc::strong_count(session) > 1` -- the only other place this
    /// map's `Arc<LlamaSession>` values are ever cloned is
    /// `handle_conversation`, which keeps its clone alive for exactly as
    /// long as that conversation's connection stays open) rather than
    /// removing the entry and then failing outright on
    /// `EvictError::HandleOutstanding` — a background conversation
    /// sitting idle shouldn't be able to make an unrelated one-shot
    /// request against a *different* model fail. Checking the count
    /// first (instead of removing speculatively and catching the evict
    /// error) also avoids ever dropping this map's own entry for a
    /// session eviction that was going to fail anyway.
    fn make_room_for(&mut self, path: &Path) -> Result<()> {
        let new_size_bytes = std::fs::metadata(path).map(|meta| meta.len()).unwrap_or(0);
        loop {
            let host_fits = match self.registry.fits_within_budget(new_size_bytes, self.budget_fraction) {
                // `None`: can't measure free memory on this platform --
                // best-effort proceed rather than refuse a request over
                // something unmeasurable.
                None => true,
                Some(fits) => fits,
            };
            // `new_size_bytes` (the file's on-disk size) doubles as the
            // device-memory estimate too, same "file size as a stand-in
            // for what full residency would cost" reasoning
            // `fits_within_budget` already applies to host memory -- see
            // that call's own precedent, not a separately invented
            // number. `None` (no GPU device present) means there's no
            // device budget to violate.
            let device_fits = match rampipe::llama::gpu_memory_bytes() {
                Some((free, _total)) => self.registry.fits_within_device_budget(new_size_bytes, free, self.budget_fraction),
                None => true,
            };
            if host_fits && device_fits {
                return Ok(());
            }

            let lru_ids = self.registry.resident_ids_by_lru();
            let evictable = lru_ids.into_iter().find_map(|id| {
                let evict_path = self.path_for_id(id)?;
                let session = self.sessions.get(&evict_path)?;
                if Arc::strong_count(session) != 1 {
                    return None;
                }
                // Host memory is already fine and only device memory is
                // under pressure: evicting a model that never touched the
                // GPU (no recorded device bytes) wouldn't free any VRAM,
                // so skip it and look further down the LRU list for one
                // that actually holds device memory.
                if host_fits && !device_fits && session.device_bytes().is_none() {
                    return None;
                }
                Some((id, evict_path))
            });
            let Some((evict_id, evict_path)) = evictable else {
                // Over budget but nothing evictable -- either nothing's
                // resident, everything resident is pinned by a live
                // conversation, or (device-only pressure) nothing
                // resident actually holds GPU memory to reclaim. A single
                // model bigger than the whole budget (or a budget fully
                // consumed by pinned models) must still be servable, not
                // refused outright, so proceed anyway.
                return Ok(());
            };

            eprintln!("rampiped: evicting {} (LRU) to make room for {}", evict_path.display(), path.display());
            self.sessions.remove(&evict_path); // drops LlamaSession -> drops its ModelHandle
            self.model_stats.remove(&evict_path);
            self.registry.evict(evict_id).context("evicting LRU model")?;
        }
    }

    fn path_for_id(&self, id: ModelId) -> Option<PathBuf> {
        self.sessions.iter().find(|(_, session)| session.id() == id).map(|(path, _)| path.clone())
    }
}

/// Request reading and response writing happen with no lock held —
/// only `handle_request` (the actual GPU-touching work) takes
/// `state`'s lock, and only for as long as that one request's turn
/// takes. A slow-to-send or slow-to-receive client blocks only its own
/// thread, never another connection's request from being read or
/// another already-queued request's turn at the lock.
///
/// The first line on every connection is a [`ClientMessage`]:
/// `Generate` keeps today's exact one-shot shape (read one line, reply
/// one line, done); `OpenConversation` hands the rest of this
/// connection's lifetime to [`handle_conversation`], which reads and
/// replies to any number of further turns on the same connection.
fn handle_connection(stream: UnixStream, state: &Mutex<SharedState>) -> Result<()> {
    let mut reader = BufReader::new(stream.try_clone().context("cloning connection stream")?);
    let mut line = String::new();
    reader.read_line(&mut line).context("reading request")?;
    let message: ClientMessage = serde_json::from_str(line.trim()).context("decoding request")?;

    match message {
        ClientMessage::Generate(request) => {
            let response = match handle_request(state, &request) {
                Ok(response) => response,
                Err(error) => GenerateResponse::Err { message: format!("{error:#}") },
            };
            let tokens_generated = match &response {
                GenerateResponse::Ok { tokens_generated, .. } => Some(*tokens_generated),
                GenerateResponse::Err { .. } => None,
            };
            state.lock().expect("rampiped model store lock poisoned").record_outcome(&request.model_path, tokens_generated);
            let mut stream = stream;
            serde_json::to_writer(&mut stream, &response).context("encoding response")?;
            stream.write_all(b"\n").context("writing response")?;
            Ok(())
        }
        ClientMessage::OpenConversation(request) => handle_conversation(stream, reader, state, request),
        ClientMessage::Status => {
            let response = handle_status(state);
            let mut stream = stream;
            serde_json::to_writer(&mut stream, &response).context("encoding response")?;
            stream.write_all(b"\n").context("writing response")?;
            Ok(())
        }
    }
}

/// Builds a [`StatusResponse`] from this process's own pid, its
/// startup-time exe snapshot (see [`current_exe_snapshot`]), and
/// whatever's currently resident — locked only long enough to read
/// `sessions`' keys, not for the duration of any generation.
fn handle_status(state: &Mutex<SharedState>) -> rampipe::protocol::StatusResponse {
    let state = state.lock().expect("rampiped model store lock poisoned");
    let (gpu_free_bytes, gpu_total_bytes) = match rampipe::llama::gpu_memory_bytes() {
        Some((free, total)) => (Some(free), Some(total)),
        None => (None, None),
    };
    let models = state
        .sessions
        .keys()
        .map(|path| {
            let stats = state.model_stats.get(path);
            rampipe::protocol::ModelStatus {
                path: path.clone(),
                requests_served: stats.map_or(0, |s| s.requests_served),
                tokens_generated: stats.map_or(0, |s| s.tokens_generated),
                last_used_unix_secs: stats.map_or(0, |s| s.last_used_unix_secs),
            }
        })
        .collect();
    rampipe::protocol::StatusResponse {
        pid: std::process::id(),
        exe_path: state.exe_snapshot.0.clone(),
        exe_modified_unix_secs: state.exe_snapshot.1,
        uptime_secs: state.started_at.elapsed().as_secs(),
        requests_served: state.requests_served,
        requests_failed: state.requests_failed,
        total_tokens_generated: state.total_tokens_generated,
        resident_bytes: state.registry.mapped_bytes(),
        gpu_free_bytes,
        gpu_total_bytes,
        models,
    }
}

/// The one critical section: locks `state` for exactly as long as it
/// takes to ensure the requested model is resident (loading/evicting if
/// needed) and run one generation against it, then releases before
/// `handle_connection` writes the response.
fn handle_request(state: &Mutex<SharedState>, request: &GenerateRequest) -> Result<GenerateResponse> {
    let mut state = state.lock().expect("rampiped model store lock poisoned");
    state.ensure_loaded(&request.model_path)?;
    let session = state.sessions.get(&request.model_path).expect("ensure_loaded just guaranteed this");
    let sampling = wire_sampling_to_sampling(request.sampling);
    let grammar_complete = request.grammar_completion.clone().map(rampipe::protocol::GrammarCompletion::into_predicate);
    let result = session
        .generate(
            &request.prompt,
            request.max_new_tokens,
            sampling,
            request.grammar.as_deref(),
            request.assistant_prefill.as_deref(),
            grammar_complete.as_deref(),
        )
        .with_context(|| format!("generating against {}", request.model_path.display()))?;

    Ok(GenerateResponse::Ok {
        text: result.text,
        tokens_generated: result.tokens_generated,
        time_to_first_token_ms: result.time_to_first_token.as_millis() as u64,
        formatted_prompt: result.formatted_prompt,
    })
}

/// Serves one whole conversation for as long as its connection stays
/// open — one thread per connection (same as the accept loop's own
/// convention), so `session` (an `Arc<LlamaSession>` clone) and the
/// `Conversation` opened from it both live as plain locals on *this*
/// thread's own stack for the conversation's entire lifetime. This is
/// what lets a `Conversation` (which borrows from the `LlamaModel`
/// inside `session`) be held across many requests without storing it
/// inside `SharedState` itself, which would need a self-referential
/// struct (`SharedState.sessions` both owns sessions and would need to
/// hold something borrowing from one of them at the same time) — an
/// ordinary function-local owner-and-borrower pair sidesteps that
/// entirely.
fn handle_conversation(
    stream: UnixStream,
    mut reader: BufReader<UnixStream>,
    state: &Mutex<SharedState>,
    request: OpenConversationRequest,
) -> Result<()> {
    let mut writer = stream;

    // Caught and turned into a real `ConversationResponse::Err` here,
    // not propagated via `?` -- unlike a request that fails to *decode*
    // at all (nothing meaningful to reply with in that case), this is a
    // request that parsed fine but failed during processing, same as
    // `handle_request`'s own `Err -> GenerateResponse::Err` conversion
    // for the one-shot path. Letting this propagate instead left a
    // client's `RampipedConversation::open` staring at a silently closed
    // connection -- a real, live-reproduced failure that surfaced as a
    // confusing "EOF while parsing a value" decode error instead of the
    // actual problem.
    let session = {
        let mut state = state.lock().expect("rampiped model store lock poisoned");
        let loaded = state.ensure_loaded(&request.model_path);
        match loaded {
            Ok(()) => Arc::clone(state.sessions.get(&request.model_path).expect("ensure_loaded just guaranteed this")),
            Err(error) => {
                drop(state);
                let response = ConversationResponse::Err { message: format!("{error:#}") };
                serde_json::to_writer(&mut writer, &response).context("encoding response")?;
                writer.write_all(b"\n").context("writing response")?;
                return Ok(());
            }
        }
    };

    let n_ctx = NonZeroU32::new(request.n_ctx).context("n_ctx must be nonzero")?;
    let options = ConversationOptions { n_ctx, overflow: wire_overflow_to_overflow(request.overflow) };

    let mut conversation = match session.open_conversation(options) {
        Ok(conversation) => conversation,
        Err(error) => {
            let response = ConversationResponse::Err { message: format!("{error:#}") };
            serde_json::to_writer(&mut writer, &response).context("encoding response")?;
            writer.write_all(b"\n").context("writing response")?;
            return Ok(());
        }
    };

    let response = ConversationResponse::Opened;
    serde_json::to_writer(&mut writer, &response).context("encoding response")?;
    writer.write_all(b"\n").context("writing response")?;

    let mut line = String::new();
    loop {
        line.clear();
        let bytes_read = reader.read_line(&mut line).context("reading conversation turn")?;
        if bytes_read == 0 {
            // Client dropped the connection -- ordinary end of this
            // conversation, not an error. No explicit close message is
            // needed.
            return Ok(());
        }

        let turn_response = match serde_json::from_str::<ConversationTurnRequest>(line.trim()) {
            Ok(turn) => {
                let response = run_conversation_turn(state, &mut conversation, &turn);
                let tokens_generated = match &response {
                    ConversationResponse::Turn { tokens_generated, .. } => Some(*tokens_generated),
                    ConversationResponse::Opened | ConversationResponse::Err { .. } => None,
                };
                state.lock().expect("rampiped model store lock poisoned").record_outcome(&request.model_path, tokens_generated);
                response
            }
            // A turn that never decoded never touched the model at all --
            // nothing to record against it, same reasoning as `Ok(())`
            // never being reached in `handle_request` when the model
            // itself fails to load before generation is ever attempted.
            Err(error) => ConversationResponse::Err { message: format!("decoding conversation turn: {error:#}") },
        };

        serde_json::to_writer(&mut writer, &turn_response).context("encoding response")?;
        writer.write_all(b"\n").context("writing response")?;
    }
}

/// The one critical section for a conversation turn: locks `state`
/// purely as a serialization gate (matching `handle_request`'s own "at
/// most one decode in flight" invariant, extended to conversation
/// turns), held only for the duration of this one `send()` call — never
/// across the idle time between turns while this thread is blocked
/// reading the next line from its own socket, so a conversation sitting
/// idle never blocks an unrelated request's turn at the lock.
fn run_conversation_turn(state: &Mutex<SharedState>, conversation: &mut Conversation<'_>, turn: &ConversationTurnRequest) -> ConversationResponse {
    let _guard = state.lock().expect("rampiped model store lock poisoned");
    let sampling = wire_sampling_to_sampling(turn.sampling);
    let grammar_complete = turn.grammar_completion.clone().map(rampipe::protocol::GrammarCompletion::into_predicate);
    let result = conversation.send(
        &turn.message,
        turn.max_new_tokens,
        sampling,
        turn.grammar.as_deref(),
        turn.assistant_prefill.as_deref(),
        grammar_complete.as_deref(),
    );
    match result {
        Ok(result) => ConversationResponse::Turn {
            text: result.text,
            tokens_generated: result.tokens_generated,
            time_to_first_token_ms: result.time_to_first_token.as_millis() as u64,
            formatted_prompt: result.formatted_prompt,
        },
        Err(error) => ConversationResponse::Err { message: format!("{error:#}") },
    }
}

/// Binds `path`, first removing any stale socket file left behind by an
/// unclean previous exit (`UnixListener::bind` on an existing path
/// otherwise fails outright) -- matches `taskpipe::daemon::bind_fresh`.
fn bind_fresh(path: &Path) -> Result<UnixListener> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("creating socket directory {}", parent.display()))?;
    }
    if path.exists() {
        std::fs::remove_file(path).with_context(|| format!("removing stale socket {}", path.display()))?;
    }
    UnixListener::bind(path).with_context(|| format!("binding socket {}", path.display()))
}

fn main() -> Result<()> {
    let args = match parse_args()? {
        ParsedArgs::Help => {
            println!("{USAGE}");
            return Ok(());
        }
        ParsedArgs::Run(args) => args,
    };
    println!("rampiped: socket {}", args.socket.display());
    println!("rampiped: budget_fraction {}", args.budget_fraction);

    // Bind before backend init, not after: `LlamaBackend::init()` can take
    // several real seconds on a cold Metal shader cache (observed:
    // ~10.5s), and a client connecting during that window should see its
    // connection queue in the kernel accept backlog and then get served,
    // not "no such file" because the socket path didn't exist yet.
    let listener = bind_fresh(&args.socket)?;
    let backend = LlamaBackend::init().context("llama.cpp backend init")?;
    let state = Arc::new(Mutex::new(SharedState::new(backend, args.budget_fraction)));

    for stream in listener.incoming() {
        let stream = match stream {
            Ok(stream) => stream,
            Err(error) => {
                eprintln!("rampiped: accept error: {error}");
                continue;
            }
        };
        let state = Arc::clone(&state);
        thread::spawn(move || {
            if let Err(error) = handle_connection(stream, &state) {
                eprintln!("rampiped: connection error: {error:#}");
            }
        });
    }
    Ok(())
}
