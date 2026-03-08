use std::collections::VecDeque;

use anyhow::{Result, anyhow};

const DEFAULT_VOC_AVG_WINDOW: usize = 5;
const DEFAULT_VOC_RISING_STEPS: usize = 3;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VocTrendConfig {
    pub avg_window: usize,
    pub rising_steps: usize,
}

impl VocTrendConfig {
    pub fn from_env() -> Result<Self> {
        let avg_window = parse_env_usize("ARGUS_VOC_AVG_WINDOW", DEFAULT_VOC_AVG_WINDOW)?;
        let rising_steps =
            parse_env_usize("ARGUS_VOC_RISING_STEPS", DEFAULT_VOC_RISING_STEPS)?;

        Ok(Self {
            avg_window,
            rising_steps,
        })
    }
}

impl Default for VocTrendConfig {
    fn default() -> Self {
        Self {
            avg_window: DEFAULT_VOC_AVG_WINDOW,
            rising_steps: DEFAULT_VOC_RISING_STEPS,
        }
    }
}

#[derive(Debug)]
pub struct VocTrendTracker {
    config: VocTrendConfig,
    samples: VecDeque<f64>,
    previous_average: Option<f64>,
    consecutive_rises: usize,
}

impl VocTrendTracker {
    pub fn new(config: VocTrendConfig) -> Self {
        Self {
            config,
            samples: VecDeque::with_capacity(config.avg_window),
            previous_average: None,
            consecutive_rises: 0,
        }
    }

    pub fn record(&mut self, sample: f64) -> bool {
        self.samples.push_back(sample);
        if self.samples.len() > self.config.avg_window {
            self.samples.pop_front();
        }

        if self.samples.len() < self.config.avg_window {
            return false;
        }

        let current_average = self.samples.iter().copied().sum::<f64>() / self.samples.len() as f64;
        let is_rising = if let Some(previous_average) = self.previous_average {
            if current_average > previous_average {
                self.consecutive_rises += 1;
            } else {
                self.consecutive_rises = 0;
            }
            self.consecutive_rises >= self.config.rising_steps
        } else {
            false
        };
        self.previous_average = Some(current_average);
        is_rising
    }
}

fn parse_env_usize(key: &str, default: usize) -> Result<usize> {
    let raw = std::env::var(key).unwrap_or_else(|_| default.to_string());
    let parsed = raw
        .parse::<usize>()
        .map_err(|e| anyhow!("invalid {key} `{raw}`: {e}"))?;
    if parsed == 0 {
        return Err(anyhow!("{key} must be greater than zero"));
    }
    Ok(parsed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex as StdMutex, OnceLock};

    fn env_lock() -> &'static StdMutex<()> {
        static LOCK: OnceLock<StdMutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| StdMutex::new(()))
    }

    fn with_env<T>(vars: &[(&str, Option<&str>)], f: impl FnOnce() -> T) -> T {
        let _guard = env_lock().lock().expect("env lock poisoned");
        let previous: Vec<_> = vars
            .iter()
            .map(|(key, _)| ((*key).to_string(), std::env::var_os(key)))
            .collect();

        for (key, value) in vars {
            match value {
                Some(value) => unsafe {
                    std::env::set_var(key, value);
                },
                None => unsafe {
                    std::env::remove_var(key);
                },
            }
        }

        let result = f();

        for (key, value) in previous {
            match value {
                Some(value) => unsafe {
                    std::env::set_var(&key, value);
                },
                None => unsafe {
                    std::env::remove_var(&key);
                },
            }
        }

        result
    }

    fn assert_trend_outputs(config: VocTrendConfig, samples: &[f64], expected: &[bool]) {
        assert_eq!(
            samples.len(),
            expected.len(),
            "samples and expected outputs must be the same length"
        );

        let mut tracker = VocTrendTracker::new(config);
        let actual: Vec<bool> = samples.iter().map(|sample| tracker.record(*sample)).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn default_voc_config_matches_expected_values() {
        with_env(
            &[
                ("ARGUS_VOC_AVG_WINDOW", None),
                ("ARGUS_VOC_RISING_STEPS", None),
            ],
            || {
                assert_eq!(VocTrendConfig::from_env().unwrap(), VocTrendConfig::default());
            },
        );
    }

    #[test]
    fn insufficient_history_is_not_increasing() {
        let mut tracker = VocTrendTracker::new(VocTrendConfig {
            avg_window: 3,
            rising_steps: 2,
        });

        assert!(!tracker.record(1.0));
        assert!(!tracker.record(2.0));
        assert!(!tracker.record(3.0));
    }

    #[test]
    fn increasing_average_requires_consecutive_windows() {
        let mut tracker = VocTrendTracker::new(VocTrendConfig {
            avg_window: 3,
            rising_steps: 2,
        });

        for sample in [1.0, 2.0, 3.0, 4.0] {
            assert!(!tracker.record(sample));
        }
        assert!(tracker.record(5.0));
    }

    #[test]
    fn non_rising_window_resets_streak() {
        let mut tracker = VocTrendTracker::new(VocTrendConfig {
            avg_window: 2,
            rising_steps: 2,
        });

        assert!(!tracker.record(1.0));
        assert!(!tracker.record(2.0));
        assert!(!tracker.record(3.0));
        assert!(!tracker.record(2.0));
        assert!(!tracker.record(4.0));
        assert!(tracker.record(5.0));
    }

    #[test]
    fn flat_sensor_data_never_counts_as_increasing() {
        assert_trend_outputs(
            VocTrendConfig {
                avg_window: 3,
                rising_steps: 2,
            },
            &[10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            &[false, false, false, false, false, false],
        );
    }

    #[test]
    fn decreasing_sensor_data_never_counts_as_increasing() {
        assert_trend_outputs(
            VocTrendConfig {
                avg_window: 3,
                rising_steps: 2,
            },
            &[8.0, 7.0, 6.0, 5.0, 4.0, 3.0],
            &[false, false, false, false, false, false],
        );
    }

    #[test]
    fn noisy_sensor_data_without_sustained_rise_never_triggers() {
        assert_trend_outputs(
            VocTrendConfig {
                avg_window: 3,
                rising_steps: 2,
            },
            &[10.0, 11.0, 10.0, 11.0, 10.0, 11.0, 10.0],
            &[false, false, false, false, false, false, false],
        );
    }

    #[test]
    fn sustained_rise_after_reset_triggers_again() {
        assert_trend_outputs(
            VocTrendConfig {
                avg_window: 2,
                rising_steps: 2,
            },
            &[1.0, 2.0, 3.0, 2.0, 4.0, 5.0],
            &[false, false, false, false, false, true],
        );
    }

    #[test]
    fn invalid_env_values_are_rejected() {
        with_env(
            &[
                ("ARGUS_VOC_AVG_WINDOW", Some("0")),
                ("ARGUS_VOC_RISING_STEPS", Some("abc")),
            ],
            || {
                let error = VocTrendConfig::from_env().unwrap_err().to_string();
                assert!(error.contains("ARGUS_VOC_AVG_WINDOW"));
            },
        );
    }
}
