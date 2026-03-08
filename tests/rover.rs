use std::collections::VecDeque;
use argus::rover::{RoverClient, RoverError, SerialTransport};

// ── MockTransport ─────────────────────────────────────────────────────────────

struct MockTransport {
    responses: VecDeque<std::io::Result<Option<String>>>,
    pub written: Vec<String>,
}

impl MockTransport {
    fn new() -> Self {
        Self {
            responses: VecDeque::new(),
            written: Vec::new(),
        }
    }

    fn push_response(&mut self, line: &str) {
        self.responses.push_back(Ok(Some(line.to_string())));
    }

    fn push_io_error(&mut self, msg: &str) {
        self.responses.push_back(Err(std::io::Error::new(
            std::io::ErrorKind::BrokenPipe,
            msg.to_string(),
        )));
    }
}

impl SerialTransport for MockTransport {
    fn write_line(&mut self, line: &str) -> std::io::Result<()> {
        self.written.push(line.to_string());
        Ok(())
    }

    fn read_line_timeout(&mut self) -> std::io::Result<Option<String>> {
        match self.responses.pop_front() {
            Some(result) => result,
            None => Ok(None), // empty queue = timeout slice
        }
    }
}

// ── ping tests ────────────────────────────────────────────────────────────────

#[test]
fn ping_success_ready() {
    let mut t = MockTransport::new();
    t.push_response("READY|");
    let mut client = RoverClient::new(t);

    let resp = client.ping().unwrap();
    assert_eq!(resp, "READY|");
    assert_eq!(client.transport.written, &["CMD|PING"]);
}

#[test]
fn ping_success_ok() {
    let mut t = MockTransport::new();
    t.push_response("OK|");
    let mut client = RoverClient::new(t);

    let resp = client.ping().unwrap();
    assert_eq!(resp, "OK|");
}

#[test]
fn ping_error_response() {
    let mut t = MockTransport::new();
    t.push_response("ERR|NOTREADY");
    let mut client = RoverClient::new(t);

    let err = client.ping().unwrap_err();
    assert!(matches!(err, RoverError::RoverErr(_)));
    if let RoverError::RoverErr(msg) = err {
        assert_eq!(msg, "ERR|NOTREADY");
    }
}

#[test]
fn ping_timeout() {
    // Empty queue — every poll returns Ok(None); should time out.
    let t = MockTransport::new();
    let mut client = RoverClient::new(t);

    assert!(matches!(client.ping().unwrap_err(), RoverError::Timeout));
}

#[test]
fn ping_ignores_unexpected_lines_before_ready() {
    let mut t = MockTransport::new();
    t.push_response("DEBUG|initializing");
    t.push_response("READY|");
    let mut client = RoverClient::new(t);

    let resp = client.ping().unwrap();
    assert_eq!(resp, "READY|");
}

// ── read_air tests ────────────────────────────────────────────────────────────

#[test]
fn read_air_success() {
    let mut t = MockTransport::new();
    t.push_response("DATA|AIR|23.5|45.2");
    let mut client = RoverClient::new(t);

    let resp = client.read_air().unwrap();
    assert!(resp.starts_with("DATA|AIR|"));
    assert_eq!(client.transport.written, &["CMD|SENSOR|AIR"]);
}

#[test]
fn read_air_error_response() {
    let mut t = MockTransport::new();
    t.push_response("ERR|SENSOR_FAIL");
    let mut client = RoverClient::new(t);

    let err = client.read_air().unwrap_err();
    assert!(matches!(err, RoverError::RoverErr(_)));
}

#[test]
fn read_air_timeout() {
    let t = MockTransport::new();
    let mut client = RoverClient::new(t);

    assert!(matches!(client.read_air().unwrap_err(), RoverError::Timeout));
}

#[test]
fn read_air_io_error_propagates() {
    let mut t = MockTransport::new();
    t.push_io_error("broken pipe");
    let mut client = RoverClient::new(t);

    assert!(matches!(client.read_air().unwrap_err(), RoverError::Io(_)));
}

// ── read_motion tests ─────────────────────────────────────────────────────────

#[test]
fn read_motion_success() {
    let mut t = MockTransport::new();
    t.push_response("DATA|MOTION|1");
    let mut client = RoverClient::new(t);

    let resp = client.read_motion().unwrap();
    assert!(resp.starts_with("DATA|MOTION|"));
    assert_eq!(client.transport.written, &["CMD|SENSOR|MOTION"]);
}

#[test]
fn read_motion_error_response() {
    let mut t = MockTransport::new();
    t.push_response("ERR|SENSOR_FAIL");
    let mut client = RoverClient::new(t);

    let err = client.read_motion().unwrap_err();
    assert!(matches!(err, RoverError::RoverErr(_)));
}

#[test]
fn read_motion_timeout() {
    let t = MockTransport::new();
    let mut client = RoverClient::new(t);

    assert!(matches!(client.read_motion().unwrap_err(), RoverError::Timeout));
}

#[test]
fn read_motion_io_error_propagates() {
    let mut t = MockTransport::new();
    t.push_io_error("broken pipe");
    let mut client = RoverClient::new(t);

    assert!(matches!(client.read_motion().unwrap_err(), RoverError::Io(_)));
}
