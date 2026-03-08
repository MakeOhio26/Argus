//! Hardware-in-the-loop tests for the ESP32 rover.
//!
//! Requires ARGUS_ROVER_PORT set in .env or the environment. Run with:
//!   cargo test --test rover_hardware
//!
//! If the ESP32 resets on connect (common with some boards), run serially:
//!   cargo test --test rover_hardware -- --test-threads=1

fn load_port() -> Option<String> {
    dotenvy::dotenv().ok();
    std::env::var("ARGUS_ROVER_PORT").ok()
}

#[test]
fn ping_esp32() {
    let port = match load_port() {
        Some(p) => p,
        None => {
            eprintln!("ARGUS_ROVER_PORT not set — skipping ping_esp32");
            return;
        }
    };
    let mut client = argus::rover::open(&port).expect("failed to open serial port");
    let resp = client.ping().expect("ping failed");
    assert!(
        resp.starts_with("READY|") || resp.starts_with("OK|"),
        "unexpected ping response: {resp}"
    );
    println!("ping: {resp}");
}

#[test]
fn read_air_esp32() {
    let port = match load_port() {
        Some(p) => p,
        None => {
            eprintln!("ARGUS_ROVER_PORT not set — skipping read_air_esp32");
            return;
        }
    };
    let mut client = argus::rover::open(&port).expect("failed to open serial port");
    let resp = client.read_air().expect("read_air failed");
    assert!(
        resp.starts_with("DATA|AIR|"),
        "unexpected air response: {resp}"
    );
    println!("air: {resp}");
}

#[test]
fn read_motion_esp32() {
    let port = match load_port() {
        Some(p) => p,
        None => {
            eprintln!("ARGUS_ROVER_PORT not set — skipping read_motion_esp32");
            return;
        }
    };
    let mut client = argus::rover::open(&port).expect("failed to open serial port");
    let resp = client.read_motion().expect("read_motion failed");
    assert!(
        resp.starts_with("DATA|MOTION|"),
        "unexpected motion response: {resp}"
    );
    println!("motion: {resp}");
}
