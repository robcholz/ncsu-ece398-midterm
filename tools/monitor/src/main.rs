use std::env;
use std::ffi::OsString;
use std::fs;
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::process::{Command, ExitCode};

const DEFAULT_BAUD: &str = "115200";
const PORT_PREFIX: &str = "cu.usbmodem";
const PRODUCT_HINT: &str = "mkboxpro_imu";

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<ExitCode, String> {
    let mut args = env::args_os().skip(1);
    let mut explicit_port: Option<PathBuf> = None;
    let mut print_port = false;
    let mut list_only = false;
    let mut use_screen = false;

    while let Some(arg) = args.next() {
        match arg.to_str() {
            Some("--") => {}
            Some("--port") => {
                let value = args
                    .next()
                    .ok_or_else(|| String::from("missing value after --port"))?;
                explicit_port = Some(PathBuf::from(value));
            }
            Some("--print-port") => print_port = true,
            Some("--list") => list_only = true,
            Some("--screen") => use_screen = true,
            Some("--help") | Some("-h") => {
                print_help();
                return Ok(ExitCode::SUCCESS);
            }
            Some(other) => return Err(format!("unrecognized argument: {other}")),
            None => return Err(String::from("non-utf8 arguments are not supported")),
        }
    }

    let ports = find_ports()?;

    if list_only {
        for port in &ports {
            println!("{}", port.display());
        }
        return Ok(ExitCode::SUCCESS);
    }

    let port = if let Some(port) = explicit_port {
        port
    } else if let Some(port) = env::var_os("MKBOXPRO_PORT") {
        PathBuf::from(port)
    } else {
        choose_port(ports)?
    };

    if print_port {
        println!("{}", port.display());
        return Ok(ExitCode::SUCCESS);
    }

    configure_port(&port)?;

    if use_screen {
        println!("opening {} with screen at {}", port.display(), DEFAULT_BAUD);
        println!("exit screen with Ctrl-A then \\\\");

        let status = Command::new("/usr/bin/screen")
            .arg(&port)
            .arg(DEFAULT_BAUD)
            .status()
            .map_err(|err| format!("failed to launch screen: {err}"))?;

        return Ok(ExitCode::from(status.code().unwrap_or(1) as u8));
    }

    println!("monitoring {} at {}", port.display(), DEFAULT_BAUD);
    println!("press Ctrl-C to stop");

    stream_port(&port)?;

    Ok(ExitCode::SUCCESS)
}

fn configure_port(port: &PathBuf) -> Result<(), String> {
    let status = Command::new("/bin/stty")
        .arg("-f")
        .arg(port)
        .arg(DEFAULT_BAUD)
        .arg("clocal")
        .arg("cread")
        .status()
        .map_err(|err| format!("failed to configure serial port with stty: {err}"))?;

    if !status.success() {
        return Err(format!("stty failed for {}", port.display()));
    }

    Ok(())
}

fn stream_port(port: &PathBuf) -> Result<(), String> {
    let mut serial = File::open(port).map_err(|err| format!("failed to open {}: {err}", port.display()))?;
    let mut stdout = io::stdout().lock();
    let mut buf = [0u8; 1024];

    loop {
        let count = serial
            .read(&mut buf)
            .map_err(|err| format!("serial read failed on {}: {err}", port.display()))?;

        if count == 0 {
            continue;
        }

        stdout
            .write_all(&buf[..count])
            .and_then(|_| stdout.flush())
            .map_err(|err| format!("stdout write failed: {err}"))?;
    }
}

fn print_help() {
    println!("cargo monitor [-- --port /dev/cu.usbmodem...]");
    println!("cargo monitor -- --print-port");
    println!("cargo monitor -- --list");
    println!("cargo monitor -- --screen");
    println!();
    println!("Default mode is read-only and exits with Ctrl-C.");
    println!("Use --screen if you want the old screen-based monitor.");
    println!("If no port is supplied, the tool auto-detects /dev/cu.usbmodem*.");
    println!("You can also set MKBOXPRO_PORT=/dev/cu.usbmodem...");
}

fn find_ports() -> Result<Vec<PathBuf>, String> {
    let mut ports = Vec::new();

    for entry in fs::read_dir("/dev").map_err(|err| format!("failed to read /dev: {err}"))? {
        let entry = entry.map_err(|err| format!("failed to iterate /dev: {err}"))?;
        let name = entry.file_name();
        if is_candidate(&name) {
            ports.push(entry.path());
        }
    }

    ports.sort();
    Ok(ports)
}

fn is_candidate(name: &OsString) -> bool {
    let Some(name) = name.to_str() else {
        return false;
    };

    name.starts_with(PORT_PREFIX)
}

fn choose_port(ports: Vec<PathBuf>) -> Result<PathBuf, String> {
    if ports.is_empty() {
        return Err(String::from(
            "no /dev/cu.usbmodem* device found; connect the board or set MKBOXPRO_PORT",
        ));
    }

    if ports.len() == 1 {
        return Ok(ports.into_iter().next().unwrap());
    }

    let hinted: Vec<_> = ports
        .iter()
        .filter(|path| path.to_string_lossy().contains(PRODUCT_HINT))
        .cloned()
        .collect();

    if hinted.len() == 1 {
        return Ok(hinted.into_iter().next().unwrap());
    }

    let mut message = String::from("multiple USB modem ports found; use --port or MKBOXPRO_PORT:\n");
    for port in ports {
        message.push_str("  ");
        message.push_str(&port.to_string_lossy());
        message.push('\n');
    }
    Err(message)
}
