//! Hardware-in-the-loop tests for the ESP32 rover.
//!
//! Requires ARGUS_ROVER_PORT set in .env or the environment. Run with:
//!   cargo test --test rover_hardware

fn load_port() -> Option<String> {
    dotenvy::dotenv().ok();
    std::env::var("ARGUS_ROVER_PORT").ok()
}

/// Opens one connection and exercises ping, read_air, and read_motion in sequence.
/// Using a single test avoids parallel port contention and the 2s boot wait happens once.
#[test]
fn esp32_sensors() {
    let port = match load_port() {
        Some(p) => p,
        None => {
            eprintln!("ARGUS_ROVER_PORT not set — skipping esp32_sensors");
            return;
        }
    };

    let mut client = argus::rover::open(&port).expect("failed to open serial port");

    let ping = client.ping().expect("ping failed");
    assert!(
        ping.starts_with("READY|") || ping.starts_with("OK|"),
        "unexpected ping response: {ping}"
    );
    println!("ping:   {ping}");

    let air = client.read_air().expect("read_air failed");
    assert!(air.starts_with("DATA|AIR|"), "unexpected air response: {air}");
    println!("air:    {air}");

    let motion = client.read_motion().expect("read_motion failed");
    assert!(
        motion.starts_with("DATA|MOTION|"),
        "unexpected motion response: {motion}"
    );
    println!("motion: {motion}");
}
