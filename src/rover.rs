//! Serial client for the ESP32 rover controller.
//!
//! # Protocol
//! Commands are UTF-8 newline-terminated pipe-delimited strings sent at 115200 baud.
//! Responses are newline-terminated lines starting with a status prefix:
//!   `READY|`, `OK|`, `ACK|`, `DONE|`, `DATA|`, `ERR|`
//!
//! # Linux system requirement
//! Requires `libudev-dev` (`apt install libudev-dev`) for the `serialport` crate.

use std::io::{BufRead, Write};
use std::time::Duration;

use thiserror::Error;

// ── Error type ────────────────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum RoverError {
    #[error("serial I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("rover command timed out")]
    Timeout,

    #[error("rover returned error: {0}")]
    RoverErr(String),
}

// ── Transport trait ───────────────────────────────────────────────────────────

/// Abstraction over the serial port, enabling mock implementations for testing.
pub trait SerialTransport: Send {
    /// Write a line terminated with `\n`.
    fn write_line(&mut self, line: &str) -> std::io::Result<()>;

    /// Read one newline-terminated line with a per-call timeout.
    /// Returns `Ok(None)` if the read window expired with no data.
    /// Returns `Ok(Some(line))` with the trailing `\n` stripped.
    /// Returns `Err` for hard I/O failures.
    fn read_line_timeout(&mut self) -> std::io::Result<Option<String>>;
}

// ── Real serial transport ─────────────────────────────────────────────────────

pub struct RealTransport {
    port: Box<dyn serialport::SerialPort>,
    reader: std::io::BufReader<Box<dyn serialport::SerialPort>>,
}

impl RealTransport {
    pub fn new(port_name: &str, baud: u32) -> Result<Self, RoverError> {
        let mut port = serialport::new(port_name, baud)
            .timeout(Duration::from_millis(500))
            .open()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        // The ESP32 resets when DTR is asserted on connect. Wait for it to
        // finish booting (mirrors the Python client's time.sleep(2.0)), then
        // flush any boot-time noise out of the input buffer.
        std::thread::sleep(Duration::from_secs(2));
        port.clear(serialport::ClearBuffer::Input)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        let reader_port = port
            .try_clone()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        Ok(Self {
            port,
            reader: std::io::BufReader::new(reader_port),
        })
    }
}

impl SerialTransport for RealTransport {
    fn write_line(&mut self, line: &str) -> std::io::Result<()> {
        write!(self.port, "{}\n", line)?;
        self.port.flush()
    }

    fn read_line_timeout(&mut self) -> std::io::Result<Option<String>> {
        let mut buf = String::new();
        match self.reader.read_line(&mut buf) {
            Ok(0) => Ok(None),
            Ok(_) => Ok(Some(buf.trim_end_matches(['\n', '\r']).to_string())),
            Err(e)
                if e.kind() == std::io::ErrorKind::TimedOut
                    || e.kind() == std::io::ErrorKind::WouldBlock =>
            {
                Ok(None)
            }
            Err(e) => Err(e),
        }
    }
}

// ── Rover client ──────────────────────────────────────────────────────────────

pub struct RoverClient<T: SerialTransport> {
    /// Exposed for test assertions. Do not use directly in production code.
    pub transport: T,
}

impl<T: SerialTransport> RoverClient<T> {
    pub fn new(transport: T) -> Self {
        Self { transport }
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Poll up to `max_polls` times for a line starting with `prefix`.
    /// `ERR|` lines short-circuit with `RoverErr`. Empty polls (timeout slices) are retried.
    fn wait_for_prefix(&mut self, prefix: &str, max_polls: usize) -> Result<String, RoverError> {
        for _ in 0..max_polls {
            match self.transport.read_line_timeout()? {
                Some(line) if line.starts_with("ERR|") => {
                    return Err(RoverError::RoverErr(line));
                }
                Some(line) if line.starts_with(prefix) => return Ok(line),
                Some(_) => {} // unexpected intermediate line — keep reading
                None => {}   // timeout slice — retry
            }
        }
        Err(RoverError::Timeout)
    }

    /// Write a command and wait for a single response with the given prefix.
    fn send_and_receive(&mut self, cmd: &str, prefix: &str) -> Result<String, RoverError> {
        self.transport.write_line(cmd)?;
        self.wait_for_prefix(prefix, 20)
    }

    /// Write a command and wait for an ACK then a DONE response (two-phase move commands).
    fn send_and_receive_two_phase(
        &mut self,
        cmd: &str,
        ack_prefix: &str,
        done_prefix: &str,
    ) -> Result<String, RoverError> {
        self.transport.write_line(cmd)?;
        self.wait_for_prefix(ack_prefix, 20)?;
        self.wait_for_prefix(done_prefix, 60)
    }

    // ── Public command methods ────────────────────────────────────────────────

    /// Ping the rover — returns `READY|...` or `OK|...`.
    pub fn ping(&mut self) -> Result<String, RoverError> {
        self.transport.write_line("CMD|PING")?;
        for _ in 0..20 {
            match self.transport.read_line_timeout()? {
                Some(line) if line.starts_with("ERR|") => {
                    return Err(RoverError::RoverErr(line));
                }
                Some(line) if line.starts_with("READY|") || line.starts_with("OK|") => {
                    return Ok(line);
                }
                Some(_) => {}
                None => {}
            }
        }
        Err(RoverError::Timeout)
    }

    /// Zero / home position.
    pub fn zero(&mut self) -> Result<String, RoverError> {
        self.send_and_receive("CMD|ZERO", "OK|ZERO")
    }

    /// Stop all movement.
    pub fn stop(&mut self) -> Result<String, RoverError> {
        self.send_and_receive("CMD|STOP", "OK|STOP")
    }

    /// Rotate by `degrees` (positive = clockwise).
    pub fn rotate(&mut self, degrees: f64) -> Result<String, RoverError> {
        let cmd = format!("CMD|MOVE|ROTATE|{:.2}", degrees);
        self.send_and_receive_two_phase(&cmd, "ACK|MOVE|ROTATE|", "DONE|MOVE|ROTATE|")
    }

    /// Move forward/backward by `centimeters` (negative = backward).
    pub fn distance(&mut self, centimeters: f64) -> Result<String, RoverError> {
        let cmd = format!("CMD|MOVE|DISTANCE|{:.2}", centimeters);
        self.send_and_receive_two_phase(&cmd, "ACK|MOVE|DISTANCE|", "DONE|MOVE|DISTANCE|")
    }

    /// Read the air sensor — returns `DATA|AIR|...`.
    pub fn read_air(&mut self) -> Result<String, RoverError> {
        self.send_and_receive("CMD|SENSOR|AIR", "DATA|AIR|")
    }

    /// Read the motion sensor — returns `DATA|MOTION|...`.
    pub fn read_motion(&mut self) -> Result<String, RoverError> {
        self.send_and_receive("CMD|SENSOR|MOTION", "DATA|MOTION|")
    }
}

// ── Convenience alias and constructor ─────────────────────────────────────────

pub type DefaultRoverClient = RoverClient<RealTransport>;

/// Open a serial port at 115200 baud and return a ready `DefaultRoverClient`.
pub fn open(port: &str) -> Result<DefaultRoverClient, RoverError> {
    let transport = RealTransport::new(port, 115_200)?;
    Ok(RoverClient::new(transport))
}
