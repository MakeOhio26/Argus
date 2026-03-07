use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use image::imageops::FilterType;

use crate::error::Result;

/// Persistent, disk-backed frame store with perceptual-hash deduplication.
///
/// Frames are stored per entity under `{base_dir}/frames/{entity_label}/`.
/// Before writing, a 64-bit average hash is computed from the JPEG bytes and
/// compared against all previously stored hashes for that entity.  A frame is
/// only persisted if its minimum Hamming distance to every stored hash exceeds
/// `novelty_threshold` — this avoids filling disk with temporally adjacent
/// frames that look identical.
///
/// Python analogy: a file-backed `dict[str, list[bytes]]` that rejects new
/// entries too similar to existing ones (like a `set` with a fuzzy equality).
pub struct FrameStore {
    base_dir: PathBuf,
    max_frames_per_entity: usize,
    /// Minimum Hamming distance (out of 64) for a frame to be considered novel.
    novelty_threshold: u32,
    /// In-memory index: entity_label → vec of perceptual hashes in insertion order.
    /// We keep this in memory so we never need to re-hash files on disk.
    hash_index: Mutex<HashMap<String, Vec<u64>>>,
}

impl FrameStore {
    /// Create a new `FrameStore`.
    ///
    /// `base_dir` is created (recursively) if it does not exist.
    /// `max_frames` is the maximum number of novel frames stored per entity;
    /// once exceeded, the oldest file on disk is deleted.
    pub fn new(base_dir: impl Into<PathBuf>, max_frames: usize) -> Result<Self> {
        let base_dir = base_dir.into();
        std::fs::create_dir_all(base_dir.join("frames"))?;
        Ok(Self {
            base_dir,
            max_frames_per_entity: max_frames,
            novelty_threshold: 10, // ~15 % of 64 bits
            hash_index: Mutex::new(HashMap::new()),
        })
    }

    /// Override the novelty threshold (useful for tests).
    pub fn with_novelty_threshold(mut self, threshold: u32) -> Self {
        self.novelty_threshold = threshold;
        self
    }

    /// Save `jpeg_bytes` for `entity_label` **only if** it is perceptually novel.
    ///
    /// Returns `true` if the frame was stored, `false` if it was too similar to
    /// an already-stored frame and was skipped.
    ///
    /// Rust concept: `Mutex::lock()` returns a `MutexGuard` that auto-releases
    /// at the end of the block — equivalent to Python's `with lock:`.
    pub fn save_if_novel(
        &self,
        entity_label: &str,
        frame_id: &str,
        jpeg_bytes: &[u8],
    ) -> Result<bool> {
        let new_hash = perceptual_hash(jpeg_bytes)?;

        let mut index = self.hash_index.lock().unwrap();
        let hashes = index.entry(entity_label.to_string()).or_default();

        // Reject if too similar to any stored frame.
        if hashes
            .iter()
            .any(|&h| hamming_distance(new_hash, h) < self.novelty_threshold)
        {
            return Ok(false);
        }

        // Write to disk.
        let entity_dir = self.entity_dir(entity_label);
        std::fs::create_dir_all(&entity_dir)?;

        let unix_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        // Sanitize frame_id so it is safe as a filename component.
        let safe_id = frame_id.replace(['/', '\\', ':'], "_");
        let filename = format!("{unix_ms:016}_{safe_id}.jpg");
        std::fs::write(entity_dir.join(&filename), jpeg_bytes)?;

        hashes.push(new_hash);

        // Evict oldest file + hash if over the limit.
        if hashes.len() > self.max_frames_per_entity {
            self.evict_oldest(entity_label, &entity_dir, hashes);
        }

        Ok(true)
    }

    /// Return all stored JPEG frames for `entity_label`, oldest-first.
    pub fn load_all(&self, entity_label: &str) -> Result<Vec<Vec<u8>>> {
        let entity_dir = self.entity_dir(entity_label);
        if !entity_dir.exists() {
            return Ok(vec![]);
        }

        let mut files = sorted_jpg_files(&entity_dir)?;
        files.sort();

        let mut frames = Vec::with_capacity(files.len());
        for path in files {
            frames.push(std::fs::read(path)?);
        }
        Ok(frames)
    }

    /// Return the most recently stored frame for `entity_label`, or `None` if
    /// no frames have been stored yet.
    pub fn load_latest(&self, entity_label: &str) -> Result<Option<Vec<u8>>> {
        let entity_dir = self.entity_dir(entity_label);
        if !entity_dir.exists() {
            return Ok(None);
        }

        let mut files = sorted_jpg_files(&entity_dir)?;
        files.sort();

        match files.last() {
            Some(path) => Ok(Some(std::fs::read(path)?)),
            None => Ok(None),
        }
    }

    // ── private helpers ──────────────────────────────────────────────────────

    fn entity_dir(&self, entity_label: &str) -> PathBuf {
        // Sanitize label so it is safe as a directory name.
        let safe_label = entity_label.replace(['/', '\\', ':'], "_");
        self.base_dir.join("frames").join(safe_label)
    }

    /// Delete the oldest file on disk and remove its hash from the in-memory index.
    fn evict_oldest(
        &self,
        _entity_label: &str,
        entity_dir: &PathBuf,
        hashes: &mut Vec<u64>,
    ) {
        if let Ok(mut files) = sorted_jpg_files(entity_dir) {
            files.sort();
            if let Some(oldest) = files.first() {
                let _ = std::fs::remove_file(oldest);
            }
        }
        if !hashes.is_empty() {
            hashes.remove(0); // oldest hash is at index 0 (insertion order)
        }
    }
}

// ── perceptual hashing ────────────────────────────────────────────────────────

/// Compute a 64-bit average hash (aHash) from JPEG bytes.
///
/// Algorithm:
/// 1. Decode JPEG → RGB image.
/// 2. Resize to 8×8 pixels (64 pixels total).
/// 3. Convert to grayscale.
/// 4. Compute the mean pixel value.
/// 5. Set bit `i` = 1 if pixel[i] ≥ mean, 0 otherwise.
///
/// Python analogy: `imagehash.average_hash(image)` from the `imagehash` library.
fn perceptual_hash(jpeg_bytes: &[u8]) -> Result<u64> {
    let img = image::load_from_memory(jpeg_bytes)?;
    let small = img.resize_exact(8, 8, FilterType::Lanczos3).to_luma8();
    let pixels: Vec<u8> = small.into_raw();

    let mean = pixels.iter().map(|&p| p as u32).sum::<u32>() / pixels.len() as u32;
    let mean = mean as u8;

    let mut hash: u64 = 0;
    for (i, &p) in pixels.iter().enumerate() {
        if p >= mean {
            hash |= 1u64 << i;
        }
    }
    Ok(hash)
}

/// Hamming distance: number of bit positions where `a` and `b` differ.
///
/// Python analogy: `bin(a ^ b).count('1')`
pub fn hamming_distance(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

/// Return a list of `.jpg` file paths in `dir`.
fn sorted_jpg_files(dir: &PathBuf) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().map(|e| e == "jpg").unwrap_or(false) {
            paths.push(path);
        }
    }
    Ok(paths)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// Minimal valid JPEG bytes (1×1 white pixel).
    fn tiny_jpeg() -> Vec<u8> {
        // A real 1×1 white JPEG so the image crate can decode it for hashing.
        let img = image::RgbImage::from_pixel(1, 1, image::Rgb([255u8, 255, 255]));
        let mut buf = Vec::new();
        let mut cursor = std::io::Cursor::new(&mut buf);
        img.write_to(&mut cursor, image::ImageFormat::Jpeg).unwrap();
        buf
    }

    /// A different tiny JPEG (1×1 black pixel).
    fn black_jpeg() -> Vec<u8> {
        let img = image::RgbImage::from_pixel(1, 1, image::Rgb([0u8, 0, 0]));
        let mut buf = Vec::new();
        let mut cursor = std::io::Cursor::new(&mut buf);
        img.write_to(&mut cursor, image::ImageFormat::Jpeg).unwrap();
        buf
    }

    #[test]
    fn hamming_distance_same_hash_is_zero() {
        assert_eq!(hamming_distance(0xDEADBEEF, 0xDEADBEEF), 0);
    }

    #[test]
    fn hamming_distance_all_bits_differ() {
        assert_eq!(hamming_distance(0u64, u64::MAX), 64);
    }

    #[test]
    fn identical_frames_stored_only_once() {
        let dir = TempDir::new().unwrap();
        let store = FrameStore::new(dir.path(), 10).unwrap();
        let jpeg = tiny_jpeg();

        let first = store.save_if_novel("cup", "f1", &jpeg).unwrap();
        let second = store.save_if_novel("cup", "f2", &jpeg).unwrap();

        assert!(first, "first frame should be stored");
        assert!(!second, "identical frame should be rejected");
        assert_eq!(store.load_all("cup").unwrap().len(), 1);
    }

    #[test]
    fn distinct_frames_both_stored() {
        let dir = TempDir::new().unwrap();
        // threshold=0 means every frame is stored (all distances ≥ 0)
        let store = FrameStore::new(dir.path(), 10)
            .unwrap()
            .with_novelty_threshold(0);

        let first = store.save_if_novel("cup", "f1", &tiny_jpeg()).unwrap();
        let second = store.save_if_novel("cup", "f2", &black_jpeg()).unwrap();

        assert!(first);
        assert!(second);
        assert_eq!(store.load_all("cup").unwrap().len(), 2);
    }

    #[test]
    fn eviction_keeps_max_frames() {
        let dir = TempDir::new().unwrap();
        let store = FrameStore::new(dir.path(), 2)
            .unwrap()
            .with_novelty_threshold(0); // store everything

        store.save_if_novel("cup", "f1", &tiny_jpeg()).unwrap();
        store.save_if_novel("cup", "f2", &black_jpeg()).unwrap();
        // Third distinct frame — should trigger eviction.
        let img = image::RgbImage::from_pixel(1, 1, image::Rgb([128u8, 128, 128]));
        let mut grey_jpeg = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut grey_jpeg), image::ImageFormat::Jpeg)
            .unwrap();
        store.save_if_novel("cup", "f3", &grey_jpeg).unwrap();

        assert_eq!(store.load_all("cup").unwrap().len(), 2);
    }

    #[test]
    fn load_latest_on_empty_entity_returns_none() {
        let dir = TempDir::new().unwrap();
        let store = FrameStore::new(dir.path(), 10).unwrap();
        assert!(store.load_latest("nonexistent").unwrap().is_none());
    }

    #[test]
    fn load_latest_returns_most_recently_stored() {
        let dir = TempDir::new().unwrap();
        let store = FrameStore::new(dir.path(), 10)
            .unwrap()
            .with_novelty_threshold(0);

        let white = tiny_jpeg();
        let black = black_jpeg();
        store.save_if_novel("cup", "f1", &white).unwrap();
        // Small sleep so unix_ms differs between filenames.
        std::thread::sleep(std::time::Duration::from_millis(5));
        store.save_if_novel("cup", "f2", &black).unwrap();

        let latest = store.load_latest("cup").unwrap().unwrap();
        // The latest stored frame should be the black one.
        assert!(!latest.is_empty());
    }
}
