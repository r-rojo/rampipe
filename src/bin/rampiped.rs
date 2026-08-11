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
//! Serial by design: every request is handled to completion, one at a
//! time, in the single accept loop below — no worker threads, no
//! request queue. A GPU can only usefully run one decode at a time
//! anyway, so serializing is simpler and strictly safer than
//! concurrency here, not a missing feature. The real cost is that a
//! slow or silent client blocks every other client while its connection
//! is open; acceptable for a local, single-machine, trusted-clients
//! daemon, not defended against here.
//!
//!     cargo run --release --features llama --bin rampiped -- \
//!         [--socket <path>] [--budget-fraction <0.0-1.0>]

use anyhow::{Context, Result, bail};
use llama_cpp_2::llama_backend::LlamaBackend;
use rampipe::llama::{LlamaSession, Sampling};
use rampipe::protocol::{GenerateRequest, GenerateResponse, WireSampling};
use rampipe::{ModelId, Residency, SwapRegistry};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::time::Instant;

const DEFAULT_BUDGET_FRACTION: f64 = 0.8;

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

fn default_socket_path() -> Result<PathBuf> {
    let home = std::env::var_os("HOME").map(PathBuf::from).context("could not determine home directory (HOME not set)")?;
    Ok(home.join(".rampipe").join("rampiped.sock"))
}

fn parse_args() -> Result<Args> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
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
        None => default_socket_path()?,
    };
    Ok(Args { socket, budget_fraction })
}

fn wire_sampling_to_sampling(sampling: WireSampling) -> Sampling {
    match sampling {
        WireSampling::Greedy => Sampling::Greedy,
        WireSampling::Temperature { temperature, top_k, seed } => Sampling::Temperature { temperature, top_k, seed },
    }
}

/// Every model this daemon currently holds resident, keyed by path —
/// the natural key for clients (which only know a file path, never a
/// `ModelId`), with `SwapRegistry` underneath doing the actual
/// residency accounting/eviction-safety bookkeeping `LlamaSession`
/// itself relies on.
struct ModelStore {
    registry: SwapRegistry,
    sessions: HashMap<PathBuf, LlamaSession>,
    budget_fraction: f64,
}

impl ModelStore {
    fn new(budget_fraction: f64) -> Self {
        Self { registry: SwapRegistry::new(), sessions: HashMap::new(), budget_fraction }
    }

    fn get_or_load(&mut self, backend: &LlamaBackend, path: &Path) -> Result<&LlamaSession> {
        if self.sessions.contains_key(path) {
            // A cache hit still needs to count as an access for LRU
            // purposes -- `SwapRegistry::load` already treats a
            // dedup-hit against an already-resident path as a real
            // touch (see `RegistryState::last_accessed`'s doc comment),
            // so re-calling it here (cheap: no new mmap, no reload) is
            // what keeps `resident_ids_by_lru` honest.
            self.registry.load(path, Residency::Lazy).context("touching cached model for LRU accounting")?;
            return Ok(self.sessions.get(path).expect("checked contains_key above"));
        }

        self.make_room_for(path)?;

        eprintln!("rampiped: loading {}", path.display());
        let load_start = Instant::now();
        let session =
            LlamaSession::load(&self.registry, backend, path, Residency::Lazy).with_context(|| format!("loading model {}", path.display()))?;
        eprintln!("rampiped: loaded {} in {:?} (now {} model(s) resident, {} bytes mapped)",
            path.display(), load_start.elapsed(), self.sessions.len() + 1, self.registry.mapped_bytes());
        self.sessions.insert(path.to_path_buf(), session);
        Ok(self.sessions.get(path).expect("just inserted"))
    }

    /// Evicts least-recently-used resident models, one at a time, until
    /// loading `new_size_bytes` more would stay within budget --
    /// respects `SwapRegistry::evict`'s own `HandleOutstanding` safety
    /// check implicitly, since every session's only outstanding handle
    /// is the one this store itself holds, dropped right before
    /// eviction.
    fn make_room_for(&mut self, path: &Path) -> Result<()> {
        let new_size_bytes = std::fs::metadata(path).map(|meta| meta.len()).unwrap_or(0);
        loop {
            match self.registry.fits_within_budget(new_size_bytes, self.budget_fraction) {
                // `None`: can't measure free memory on this platform --
                // best-effort proceed rather than refuse a request over
                // something unmeasurable. `Some(true)`: already fits.
                None | Some(true) => return Ok(()),
                Some(false) => {}
            }

            let lru_ids = self.registry.resident_ids_by_lru();
            let Some(&evict_id) = lru_ids.first() else {
                // Over budget but nothing resident to evict -- a single
                // model bigger than the whole budget must still be
                // servable, not refused outright, so proceed anyway.
                return Ok(());
            };
            let evict_path = self.path_for_id(evict_id);
            let Some(evict_path) = evict_path else {
                // Resident in the registry but not one of this store's
                // own sessions -- shouldn't happen (this store is the
                // registry's only caller), but don't loop forever on it.
                return Ok(());
            };

            eprintln!("rampiped: evicting {} (LRU) to make room for {}", evict_path.display(), path.display());
            self.sessions.remove(&evict_path); // drops LlamaSession -> drops its ModelHandle
            self.registry.evict(evict_id).context("evicting LRU model")?;
        }
    }

    fn path_for_id(&self, id: ModelId) -> Option<PathBuf> {
        self.sessions.iter().find(|(_, session)| session.id() == id).map(|(path, _)| path.clone())
    }
}

fn handle_connection(stream: UnixStream, backend: &LlamaBackend, store: &mut ModelStore) -> Result<()> {
    let mut reader = BufReader::new(stream.try_clone().context("cloning connection stream")?);
    let mut line = String::new();
    reader.read_line(&mut line).context("reading request")?;
    let request: GenerateRequest = serde_json::from_str(line.trim()).context("decoding request")?;

    let response = match handle_request(backend, store, &request) {
        Ok(response) => response,
        Err(error) => GenerateResponse::Err { message: format!("{error:#}") },
    };

    let mut stream = stream;
    serde_json::to_writer(&mut stream, &response).context("encoding response")?;
    stream.write_all(b"\n").context("writing response")?;
    Ok(())
}

fn handle_request(backend: &LlamaBackend, store: &mut ModelStore, request: &GenerateRequest) -> Result<GenerateResponse> {
    let session = store.get_or_load(backend, &request.model_path)?;
    let sampling = wire_sampling_to_sampling(request.sampling);
    let result = session
        .generate(backend, &request.prompt, request.max_new_tokens, sampling)
        .with_context(|| format!("generating against {}", request.model_path.display()))?;

    Ok(GenerateResponse::Ok {
        text: result.text,
        tokens_generated: result.tokens_generated,
        time_to_first_token_ms: result.time_to_first_token.as_millis() as u64,
    })
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
    let args = parse_args()?;
    println!("rampiped: socket {}", args.socket.display());
    println!("rampiped: budget_fraction {}", args.budget_fraction);

    // Bind before backend init, not after: `LlamaBackend::init()` can take
    // several real seconds on a cold Metal shader cache (observed:
    // ~10.5s), and a client connecting during that window should see its
    // connection queue in the kernel accept backlog and then get served,
    // not "no such file" because the socket path didn't exist yet.
    let listener = bind_fresh(&args.socket)?;
    let backend = LlamaBackend::init().context("llama.cpp backend init")?;
    let mut store = ModelStore::new(args.budget_fraction);
    for stream in listener.incoming() {
        let stream = match stream {
            Ok(stream) => stream,
            Err(error) => {
                eprintln!("rampiped: accept error: {error}");
                continue;
            }
        };
        if let Err(error) = handle_connection(stream, &backend, &mut store) {
            eprintln!("rampiped: connection error: {error:#}");
        }
    }
    Ok(())
}
