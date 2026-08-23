fn main() {
    let settings = rampipe::model_settings::ModelSettings::load(
        &rampipe::model_settings::ModelSettings::default_path().unwrap()).unwrap();
    for m in std::fs::read_dir("/home/rrojo/models").unwrap().filter_map(Result::ok) {
        let p = m.path();
        if p.extension().is_none_or(|e| e != "gguf") { continue; }
        let entry = settings.entry_for(&p);
        println!("  {}\n    {:?}\n    note: {}",
            p.file_name().unwrap().to_string_lossy(),
            settings.sampling_for(&p),
            entry.and_then(|e| e.note.as_deref()).unwrap_or("(none)"));
    }
}
