//! Per-model sampling settings, for the half that cannot be discovered.
//!
//! # Why this exists beside template discovery, not instead of it
//!
//! A GGUF carries what it carries. Reading one on this machine:
//!
//! ```text
//! 44 metadata keys
//!   general.architecture = qwen3moe
//!   qwen3moe.context_length = 262144
//!   tokenizer.chat_template = {# ... #}
//!   ...
//! anything that looks like a sampling default: NONE
//! ```
//!
//! The chat template is *in the file*, which is why
//! [`crate::tool_format`] can derive how a model emits tool calls rather
//! than being told. There is no sampling metadata at all -- no
//! temperature, no top-k, no repetition penalty, and no standard GGUF
//! field to put them in. Those numbers live on a model card, in prose,
//! on a webpage, and nothing in this stack reads webpages.
//!
//! So: discovered where discoverable, configured where not. That split
//! is deliberate and this module is the second half of it.
//!
//! # Why the settings belong to the daemon
//!
//! Repetition penalty is a fact about a *model*, established by reading
//! that model's card. Before this, it was a literal in one example
//! program that got copied into a real worker -- two copies of one
//! number, in a codebase that also had a properly reasoned
//! `SamplingConfig` in `agentpipe` that neither of them read, and whose
//! values disagreed with both.
//!
//! `rampiped` is the process that owns the model. Putting the numbers
//! here means every client -- `aish`, `agentpiped`, `agent99`, an
//! example -- gets the model's own settings without holding a constant,
//! and a client that genuinely knows better can still override per turn.

use crate::protocol::{WirePenalties, WireSampling};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Everything a config file says about models.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelSettings {
    /// Used for any model no entry matches. Absent means the built-in
    /// defaults, which are llama.cpp's own: no penalty at all.
    #[serde(default)]
    pub default: Option<Entry>,
    /// Keyed by a substring of the model's file name.
    ///
    /// A substring rather than an exact path because the number is a
    /// property of the *model*, and one model ships as many files --
    /// `Qwen3-Coder-30B-A3B-Instruct` is `Q4_K_M` today and may be
    /// `Q5_K_M` next week, with the same card and the same recommended
    /// sampling. Keying on the path would mean re-entering the settings
    /// on every requantisation, which is how a config goes stale.
    ///
    /// **Longest match wins**, so a general entry and a specific one can
    /// coexist and the specific one is reached.
    #[serde(default)]
    pub models: BTreeMap<String, Entry>,
}

/// What one model's card says.
///
/// Plain descriptive fields, converted to [`WireSampling`] at the edge.
/// Same reason `agentpipe::daemon_config` keeps its own shape: a config
/// describes what a file *says*, not what a wire type happens to look
/// like this month.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Entry {
    /// `None` means greedy. A card that recommends against greedy
    /// decoding is the reason this is expressible at all.
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_k: Option<i32>,
    #[serde(default)]
    pub seed: Option<u32>,
    #[serde(default = "one")]
    pub repeat_penalty: f32,
    #[serde(default)]
    pub frequency_penalty: f32,
    #[serde(default)]
    pub presence_penalty: f32,
    #[serde(default)]
    pub penalty_last_n: i32,
    /// How much a model may generate in one turn.
    ///
    /// A fact about the model in the same way the penalties are -- what
    /// it can produce coherently in one go, and how much room a caller
    /// has to give it. Here rather than in the caller for the reason
    /// this whole module exists: `agent99` held `1500` as a literal
    /// copied out of an example, and a model asked to write a 136-line
    /// test file was guillotined at exactly that number, mid-expression,
    /// with the fragment written to disk.
    ///
    /// `None` leaves it to the caller, which is what a classifier turn
    /// wanting sixteen tokens should do.
    #[serde(default)]
    pub max_new_tokens: Option<i32>,
    /// Free text, so the file records *where a number came from*. Not
    /// decoration: the whole failure this module fixes was a value
    /// nobody could trace back to its source.
    #[serde(default)]
    pub note: Option<String>,
}

const fn one() -> f32 {
    1.0
}

impl Entry {
    /// This entry as the daemon will apply it.
    #[must_use]
    pub fn sampling(&self) -> WireSampling {
        let penalties = WirePenalties {
            last_n: self.penalty_last_n,
            repeat: self.repeat_penalty,
            freq: self.frequency_penalty,
            present: self.presence_penalty,
        };
        match self.temperature {
            Some(temperature) => WireSampling::Temperature {
                temperature,
                top_k: self.top_k.unwrap_or(40),
                seed: self.seed.unwrap_or(0),
                penalties,
            },
            None => WireSampling::Greedy { penalties },
        }
    }
}

impl ModelSettings {
    /// Where settings live when nothing says otherwise -- beside the
    /// socket, because both belong to the daemon.
    #[must_use]
    pub fn default_path() -> Option<PathBuf> {
        std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".rampipe").join("models.toml"))
    }

    /// Reads `path`, or answers empty settings when it is not there.
    ///
    /// A missing file is not an error: a host that has never written one
    /// should behave exactly as it did before this existed. A file that
    /// is there and malformed *is* an error, because silently ignoring
    /// it would apply defaults while the operator believed otherwise --
    /// which is the failure mode this module is about.
    pub fn load(path: &Path) -> Result<Self, SettingsError> {
        let raw = match std::fs::read_to_string(path) {
            Ok(raw) => raw,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Self::default()),
            Err(source) => return Err(SettingsError::Io { path: path.to_path_buf(), source }),
        };
        toml::from_str(&raw).map_err(|source| SettingsError::Parse { path: path.to_path_buf(), source })
    }

    /// The entry for `model_path`: longest matching key, else `default`.
    #[must_use]
    pub fn entry_for(&self, model_path: &Path) -> Option<&Entry> {
        let name = model_path.file_name()?.to_string_lossy().into_owned();
        self.models
            .iter()
            .filter(|(key, _)| name.contains(key.as_str()))
            .max_by_key(|(key, _)| key.len())
            .map(|(_, entry)| entry)
            .or(self.default.as_ref())
    }

    /// How much `model_path` may generate in one turn, when a caller did
    /// not say.
    #[must_use]
    pub fn max_new_tokens_for(&self, model_path: &Path) -> Option<i32> {
        self.entry_for(model_path).and_then(|entry| entry.max_new_tokens)
    }

    /// What to sample with for `model_path` when a caller did not say.
    #[must_use]
    pub fn sampling_for(&self, model_path: &Path) -> WireSampling {
        self.entry_for(model_path)
            .map_or_else(|| WireSampling::Greedy { penalties: WirePenalties::default() }, Entry::sampling)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SettingsError {
    #[error("reading {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("parsing {path}: {source}")]
    Parse {
        path: PathBuf,
        #[source]
        source: toml::de::Error,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings(text: &str) -> ModelSettings {
        toml::from_str(text).expect("parse")
    }

    /// The cap belongs to the model, not to whichever program is
    /// driving it. See `Entry::max_new_tokens`.
    #[test]
    fn a_generation_cap_can_be_configured_per_model() {
        let config = settings("[models.big]\nmax_new_tokens = 8192\n\n[models.small]\nrepeat_penalty = 1.1\n");
        assert_eq!(config.max_new_tokens_for(Path::new("/m/big-Q4.gguf")), Some(8192));
        assert_eq!(
            config.max_new_tokens_for(Path::new("/m/small-Q4.gguf")),
            None,
            "an entry that says nothing about it leaves the caller's own value alone"
        );
        assert_eq!(config.max_new_tokens_for(Path::new("/m/unknown.gguf")), None);
    }

    #[test]
    fn a_model_gets_the_settings_its_card_recommends() {
        let config = settings(
            r#"
            [models."Qwen3-Coder-30B-A3B-Instruct"]
            repeat_penalty = 1.18
            frequency_penalty = 0.1
            presence_penalty = 1.0
            penalty_last_n = 512
            note = "from the model card"
            "#,
        );
        let WireSampling::Greedy { penalties } =
            config.sampling_for(Path::new("/m/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf"))
        else {
            panic!("no temperature configured, so greedy");
        };
        assert_eq!(penalties.repeat, 1.18);
        assert_eq!(penalties.last_n, 512);
        assert_eq!(
            config.entry_for(Path::new("/m/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf")).and_then(|e| e.note.as_deref()),
            Some("from the model card"),
            "the file records where a number came from"
        );
    }

    /// One model, many quantisations, one card. See `models`.
    #[test]
    fn every_quantisation_of_one_model_matches_the_same_entry() {
        let config = settings("[models.\"Qwen3-Coder-30B\"]\nrepeat_penalty = 1.18\n");
        for quant in ["Q4_K_M", "Q5_K_M", "IQ4_XS"] {
            let WireSampling::Greedy { penalties } =
                config.sampling_for(Path::new(&format!("/m/Qwen3-Coder-30B-A3B-{quant}.gguf")))
            else {
                panic!("greedy");
            };
            assert_eq!(penalties.repeat, 1.18, "{quant} is the same model");
        }
    }

    #[test]
    fn the_more_specific_entry_wins() {
        let config = settings(
            "[models.\"Qwen3\"]\nrepeat_penalty = 1.1\n\n[models.\"Qwen3-Coder-30B\"]\nrepeat_penalty = 1.18\n",
        );
        let repeat = |name: &str| match config.sampling_for(Path::new(name)) {
            WireSampling::Greedy { penalties } => penalties.repeat,
            WireSampling::Temperature { penalties, .. } => penalties.repeat,
        };
        assert_eq!(repeat("/m/Qwen3-Coder-30B-A3B-Q4.gguf"), 1.18);
        assert_eq!(repeat("/m/Qwen3.8-27B-UD-IQ4_XS.gguf"), 1.1, "the general entry still covers its siblings");
    }

    /// A card that says not to decode greedily has to be expressible.
    #[test]
    fn a_temperature_entry_produces_temperature_sampling() {
        let config = settings("[models.thinking]\ntemperature = 0.7\ntop_k = 20\nrepeat_penalty = 1.05\n");
        let WireSampling::Temperature { temperature, top_k, penalties, .. } =
            config.sampling_for(Path::new("/m/thinking.gguf"))
        else {
            panic!("a temperature was configured");
        };
        assert_eq!(temperature, 0.7);
        assert_eq!(top_k, 20);
        assert_eq!(penalties.repeat, 1.05);
    }

    /// A host that never wrote one must behave exactly as before.
    #[test]
    fn no_file_means_no_penalty_which_is_what_llama_cpp_itself_defaults_to() {
        let config = ModelSettings::load(Path::new("/nonexistent/models.toml")).expect("missing is not an error");
        let WireSampling::Greedy { penalties } = config.sampling_for(Path::new("/m/anything.gguf")) else {
            panic!("greedy");
        };
        assert_eq!(penalties.repeat, 1.0);
        assert_eq!(penalties.last_n, 0);
    }

    /// Silently ignoring a broken file would apply defaults while the
    /// operator believed their settings were in force.
    #[test]
    fn a_malformed_file_is_an_error_rather_than_silence() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let path = dir.path().join("models.toml");
        std::fs::write(&path, "[models.x]\nrepeat_penalty = \"not a number\"\n").expect("write");
        assert!(matches!(ModelSettings::load(&path), Err(SettingsError::Parse { .. })));
    }

    /// A typo in a key must not be accepted as "no setting".
    #[test]
    fn an_unknown_field_is_refused() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let path = dir.path().join("models.toml");
        std::fs::write(&path, "[models.x]\nrepetition_penalty = 1.1\n").expect("write");
        assert!(matches!(ModelSettings::load(&path), Err(SettingsError::Parse { .. })));
    }
}
