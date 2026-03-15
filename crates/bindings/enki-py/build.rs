use std::{env, fs, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=./src/enki.udl");

    uniffi::generate_scaffolding("./src/enki.udl").unwrap();

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR must be set"));
    let scaffolding_path = out_dir.join("enki.uniffi.rs");
    let scaffolding = fs::read_to_string(&scaffolding_path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", scaffolding_path.display()));
    let patched = scaffolding.replace("#[no_mangle]", "#[unsafe(no_mangle)]");

    if patched != scaffolding {
        fs::write(&scaffolding_path, patched).unwrap_or_else(|error| {
            panic!("failed to write {}: {error}", scaffolding_path.display())
        });
    }
}
