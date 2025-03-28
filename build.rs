extern crate bindgen;

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Clone, Copy, Eq, PartialEq)]
enum OS {
    Linux,
    MacOS,
    Windows,
}

impl OS {
    fn get() -> Self {
        let os = env::var("CARGO_CFG_TARGET_OS").expect("Unable to get TARGET_OS");
        match os.as_str() {
            "linux" => Self::Linux,
            "macos" => Self::MacOS,
            "windows" => Self::Windows,
            os => panic!("Unsupported system {os}"),
        }
    }
}

fn get_download_url(os: OS) -> &'static str {
    match os {
        OS::Linux if cfg!(feature = "cpu") && cfg!(target_arch = "x86_64") => {
            "https://github.com/elixir-nx/xla/releases/download/v0.8.0/xla_extension-0.8.0-x86_64-linux-gnu-cpu.tar.gz"
        }
        OS::Linux if cfg!(feature = "cuda") && cfg!(target_arch = "x86_64") => {
            "https://github.com/elixir-nx/xla/releases/download/v0.8.0/xla_extension-0.8.0-x86_64-linux-gnu-cuda12.tar.gz"
        }
        OS::Linux if cfg!(feature = "tpu") && cfg!(target_arch = "x86_64") => {
            "https://github.com/elixir-nx/xla/releases/download/v0.8.0/xla_extension-0.8.0-x86_64-linux-gnu-tpu.tar.gz"
        }
        OS::MacOS if cfg!(feature = "cpu") && cfg!(target_arch = "x86_64") => {
            "https://github.com/elixir-nx/xla/releases/download/v0.8.0/xla_extension-0.8.0-x86_64-darwin-cpu.tar.gz"
        }
        OS::Linux if cfg!(feature = "cpu") && cfg!(target_arch = "aarch64") => {
            "https://github.com/elixir-nx/xla/releases/download/v0.8.0/xla_extension-0.8.0-aarch64-linux-gnu-cpu.tar.gz"
        }
        OS::Linux if cfg!(feature = "cuda") && cfg!(target_arch = "aarch64") => {
            "https://github.com/elixir-nx/xla/releases/download/v0.8.0/xla_extension-0.8.0-aarch64-linux-gnu-cuda12.tar.gz"
        }
        OS::MacOS if cfg!(feature = "cpu") && cfg!(target_arch = "aarch64") => {
            "https://github.com/elixir-nx/xla/releases/download/v0.8.0/xla_extension-0.8.0-aarch64-darwin-cpu.tar.gz"
        }
        _ => panic!("Unsupported OS/architecture combination"),
    }
}

fn make_shared_lib<P: AsRef<Path>>(os: OS, xla_dir: P) {
    println!("cargo:rerun-if-changed=xla_rs/xla_rs.cc");
    println!("cargo:rerun-if-changed=xla_rs/xla_rs.h");
    match os {
        OS::Linux | OS::MacOS => {
            cc::Build::new()
                .cpp(true)
                .pic(true)
                .warnings(false)
                .include(xla_dir.as_ref().join("include"))
                .flag("-std=c++17")
                .flag("-Wno-deprecated-declarations")
                .flag("-DLLVM_ON_UNIX=1")
                .flag("-DLLVM_VERSION_STRING=")
                .file("xla_rs/xla_rs.cc")
                .compile("xla_rs");
        }
        OS::Windows => panic!("does not support windows"),
    };
}

fn env_var_rerun(name: &str) -> Option<String> {
    println!("cargo:rerun-if-env-changed={name}");
    env::var(name).ok()
}

fn main() {
    let os = OS::get();
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    let xla_dir = env_var_rerun("XLA_EXTENSION_DIR")
        .map_or_else(|| out_path.join("xla_extension"), PathBuf::from);

    if !xla_dir.exists() || fs::read_dir(&xla_dir).unwrap().next().is_none() {
        let download_path = out_path.join("xla_extension.tar.gz");
        if !download_path.exists() {
            let download_url = get_download_url(os);

            Command::new("curl")
                .arg("-L")
                .arg("-o")
                .arg(&download_path)
                .arg(download_url)
                .status()
                .expect("Failed to download XLA extension");
        }

        Command::new("mkdir")
            .arg("-p")
            .arg(&xla_dir)
            .status()
            .expect("Failed to create XLA extension directory");

        Command::new("tar")
            .arg("-xzvf")
            .arg(&download_path)
            .arg("-C")
            .arg(&xla_dir)
            .arg("--strip-components=1")
            .status()
            .expect("Failed to extract XLA extension");

        env::set_var("XLA_EXTENSION_DIR", &xla_dir);
    }

    println!("cargo:rerun-if-changed=xla_rs/xla_rs.h");
    println!("cargo:rerun-if-changed=xla_rs/xla_rs.cc");
    let bindings = bindgen::Builder::default()
        .header("xla_rs/xla_rs.h")
        .wrap_unsafe_ops(true)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");
    bindings.write_to_file(out_path.join("c_xla.rs")).expect("Couldn't write bindings!");

    if std::env::var("DOCS_RS").is_ok() {
        return;
    }
    make_shared_lib(os, &xla_dir);

    if os == OS::Linux {
        println!("cargo:rustc-link-arg=-Wl,-lstdc++");
    }
    println!("cargo:rustc-link-lib=dylib=xla_rs");
    let abs_xla_dir = xla_dir.canonicalize().unwrap();
    println!("cargo:rustc-link-search=native={}", abs_xla_dir.join("lib").display());
    if os == OS::MacOS {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", abs_xla_dir.join("lib").display());
    } else {
        println!("cargo:rustc-link-arg=-Wl,-rpath={}", abs_xla_dir.join("lib").display());
    }
    println!("cargo:rustc-link-lib=xla_extension");
}
