//! DuplexBudget: token-bucket bandwidth governor for NVMe reads and writes.
//!
//! Splits the measured NVMe bandwidth between the prefetch-read path and the
//! write-back path so that a burst of optimizer write-backs can never push a
//! prefetch read past its EDF deadline.
//!
//! # Architecture
//! Two independent [`AtomicI64`] token counters (one per direction) are refilled
//! proportionally to elapsed wall-clock time. Before submitting any SQE, the
//! relevant subsystem calls [`DuplexBudget::take_read`] or
//! [`DuplexBudget::take_write`].  On exhaustion, the caller defers the operation
//! and retries after the next [`DuplexBudget::refill`] call.
//!
//! # Duplex split
//! Default: 60 % reads / 40 % writes ([`DEFAULT_READ_FRACTION`]).
//! Reads are biased because the prefetch pipeline is latency-critical; write-backs
//! are best-effort and tolerate deferral.
//!
//! # Ordering guarantee
//! [`take_read`] and [`take_write`] use AcqRel CAS loops to atomically
//! check-and-decrement without a Mutex.  No tokens are consumed on failure.
//!
//! # Integration with the in-flight count cap (A4)
//! The existing `max_inflight_writes` cap in [`crate::writeback::WritebackScheduler`]
//! remains as a concurrency guard.  **Both** the token-bucket bandwidth guarantee
//! AND the count cap must pass before a `write_async` SQE is submitted.
//!
//! [`take_read`]: DuplexBudget::take_read
//! [`take_write`]: DuplexBudget::take_write

use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default fraction of NVMe bandwidth allocated to prefetch reads.
///
/// 60 % reads / 40 % writes: biases toward reads since prefetch latency is
/// critical; write-backs tolerate deferral.
pub const DEFAULT_READ_FRACTION: f64 = 0.6;

// ---------------------------------------------------------------------------
// BandwidthExhausted
// ---------------------------------------------------------------------------

/// Returned when a token-bucket direction is exhausted.
///
/// The caller must defer the I/O operation and retry after the next call to
/// [`DuplexBudget::refill`].  No tokens are consumed when this error is returned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BandwidthExhausted;

impl std::fmt::Display for BandwidthExhausted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NVMe bandwidth token bucket exhausted; defer and retry after refill")
    }
}

// ---------------------------------------------------------------------------
// DuplexBudget
// ---------------------------------------------------------------------------

/// Dual-channel token bucket for NVMe read/write bandwidth separation.
///
/// Construct with [`DuplexBudget::new`].  Wire into the prefetch state machine
/// via [`crate::state_machine::PrefetchStateMachine::set_duplex_budget`] and
/// into the write-back scheduler via
/// [`crate::writeback::WritebackScheduler::set_duplex_budget`].
pub struct DuplexBudget {
    /// Available bytes for prefetch-read SQEs.  May transiently dip negative
    /// if a concurrent CAS loop races before the cap is applied; `refill` always
    /// clamps back to `read_cap`.
    read_tokens: AtomicI64,
    /// Available bytes for write-back SQEs.
    write_tokens: AtomicI64,
    /// Ceiling for read tokens: 1 second × read bandwidth.
    read_cap: i64,
    /// Ceiling for write tokens: 1 second × write bandwidth.
    write_cap: i64,
    /// Read-bucket refill rate: bytes per microsecond.
    refill_read_bpu: f64,
    /// Write-bucket refill rate: bytes per microsecond.
    refill_write_bpu: f64,
    /// Timestamp of the last [`refill`] call, guarded by a Mutex so that
    /// [`refill`] can take `&self` while updating mutable state.
    ///
    /// [`refill`]: DuplexBudget::refill
    last_refill: Mutex<Instant>,
}

impl DuplexBudget {
    /// Create a duplex token bucket from the measured NVMe bandwidth.
    ///
    /// `nvme_bandwidth_gbs` — `HardwareProfile::nvme_bandwidth_gbs` (GB/s).
    /// `read_fraction` — fraction of bandwidth reserved for reads; must be in
    ///   `(0.0, 1.0)`. Values are clamped to `[0.01, 0.99]`.
    ///   Use [`DEFAULT_READ_FRACTION`] (0.6) for the recommended default.
    /// `initial_fill_fraction` — how full to pre-fill both buckets (`1.0` = full,
    ///   `0.0` = empty). Full is the right choice for production; empty for tests
    ///   that want to verify deferral behaviour immediately.
    pub fn new(nvme_bandwidth_gbs: f32, read_fraction: f64, initial_fill_fraction: f64) -> Self {
        let rf = read_fraction.clamp(0.01, 0.99);
        // 1 GB/s = 1000 bytes/μs (1e9 bytes/s ÷ 1e6 μs/s).
        let total_bpu = nvme_bandwidth_gbs as f64 * 1_000.0;
        let read_bpu = total_bpu * rf;
        let write_bpu = total_bpu * (1.0 - rf);

        // Token caps: 1 second of allocated bandwidth.  Prevents runaway token
        // accumulation after long idle periods.
        let read_cap = (read_bpu * 1_000_000.0) as i64;
        let write_cap = (write_bpu * 1_000_000.0) as i64;
        let fill = initial_fill_fraction.clamp(0.0, 1.0);

        Self {
            read_tokens: AtomicI64::new((read_cap as f64 * fill) as i64),
            write_tokens: AtomicI64::new((write_cap as f64 * fill) as i64),
            read_cap,
            write_cap,
            refill_read_bpu: read_bpu,
            refill_write_bpu: write_bpu,
            last_refill: Mutex::new(Instant::now()),
        }
    }

    /// Credit both buckets for the wall-clock time elapsed since the previous call.
    ///
    /// Must be called at the start of every `on_layer_start` so that tokens
    /// accumulate at the configured NVMe bandwidth rate.
    pub fn refill(&self) {
        let elapsed_us = {
            let mut guard = self.last_refill.lock().unwrap_or_else(|p| p.into_inner());
            let us = guard.elapsed().as_micros() as f64;
            *guard = Instant::now();
            us
        };
        self.refill_by_elapsed_us(elapsed_us);
    }

    /// Credit both buckets for exactly `elapsed_us` microseconds.
    ///
    /// Separated from [`refill`] so unit tests can inject a precise elapsed
    /// duration without sleeping.
    ///
    /// [`refill`]: DuplexBudget::refill
    pub fn refill_by_elapsed_us(&self, elapsed_us: f64) {
        let read_add = (elapsed_us * self.refill_read_bpu) as i64;
        let write_add = (elapsed_us * self.refill_write_bpu) as i64;
        let read_cap = self.read_cap;
        let write_cap = self.write_cap;
        let _ = self.read_tokens.fetch_update(
            Ordering::AcqRel,
            Ordering::Acquire,
            |cur| Some((cur + read_add).min(read_cap)),
        );
        let _ = self.write_tokens.fetch_update(
            Ordering::AcqRel,
            Ordering::Acquire,
            |cur| Some((cur + write_add).min(write_cap)),
        );
    }

    /// Consume `bytes` from the read-token bucket.
    ///
    /// Uses an AcqRel CAS loop: atomically checks that at least `bytes` tokens
    /// are available, then decrements.  If the bucket is exhausted no tokens
    /// are consumed and [`BandwidthExhausted`] is returned.
    ///
    /// # Errors
    /// [`BandwidthExhausted`] — fewer than `bytes` read tokens remain.
    /// The caller must defer the SQE and retry after the next [`refill`] call.
    ///
    /// [`refill`]: DuplexBudget::refill
    pub fn take_read(&self, bytes: u64) -> Result<(), BandwidthExhausted> {
        let need = bytes as i64;
        loop {
            let cur = self.read_tokens.load(Ordering::Acquire);
            if cur < need {
                return Err(BandwidthExhausted);
            }
            if self.read_tokens
                .compare_exchange(cur, cur - need, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return Ok(());
            }
        }
    }

    /// Consume `bytes` from the write-token bucket.
    ///
    /// Uses an AcqRel CAS loop: atomically checks that at least `bytes` tokens
    /// are available, then decrements.  If the bucket is exhausted no tokens
    /// are consumed and [`BandwidthExhausted`] is returned.
    ///
    /// # Errors
    /// [`BandwidthExhausted`] — fewer than `bytes` write tokens remain.
    /// The caller must defer the write-back and retry after the next [`refill`] call.
    ///
    /// [`refill`]: DuplexBudget::refill
    pub fn take_write(&self, bytes: u64) -> Result<(), BandwidthExhausted> {
        let need = bytes as i64;
        loop {
            let cur = self.write_tokens.load(Ordering::Acquire);
            if cur < need {
                return Err(BandwidthExhausted);
            }
            if self.write_tokens
                .compare_exchange(cur, cur - need, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return Ok(());
            }
        }
    }

    /// Current read-token balance (point-in-time snapshot).
    pub fn read_tokens(&self) -> i64 {
        self.read_tokens.load(Ordering::Acquire)
    }

    /// Current write-token balance (point-in-time snapshot).
    pub fn write_tokens(&self) -> i64 {
        self.write_tokens.load(Ordering::Acquire)
    }

    /// Read-token ceiling (1 second × read bandwidth in bytes).
    pub fn read_cap(&self) -> i64 {
        self.read_cap
    }

    /// Write-token ceiling (1 second × write bandwidth in bytes).
    pub fn write_cap(&self) -> i64 {
        self.write_cap
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// 3 GB/s NVMe, 60 % reads, fully pre-filled.
    fn full_budget() -> DuplexBudget {
        DuplexBudget::new(3.0, DEFAULT_READ_FRACTION, 1.0)
    }

    /// 3 GB/s NVMe, 60 % reads, completely empty (for deferral tests).
    fn empty_budget() -> DuplexBudget {
        DuplexBudget::new(3.0, DEFAULT_READ_FRACTION, 0.0)
    }

    // -----------------------------------------------------------------------
    // Refill math
    // -----------------------------------------------------------------------

    /// After 1 ms (1 000 μs) of elapsed time the read bucket should gain exactly
    /// `read_bpu × 1000` bytes, and the write bucket `write_bpu × 1000` bytes.
    ///
    /// With 3 GB/s total, 50/50 split:
    ///   bpu = 1 500 bytes/μs → 1 ms → 1 500 000 bytes each.
    #[test]
    fn refill_math_1ms_elapsed_symmetric() {
        let b = DuplexBudget::new(3.0, 0.5, 0.0);
        b.refill_by_elapsed_us(1_000.0);
        let expected = (1_000.0_f64 * 3.0 * 1_000.0 * 0.5) as i64; // 1_500_000
        assert_eq!(b.read_tokens(), expected.min(b.read_cap()));
        assert_eq!(b.write_tokens(), expected.min(b.write_cap()));
    }

    /// With an asymmetric 60/40 split the two buckets accumulate at different rates.
    #[test]
    fn refill_math_asymmetric_split() {
        let b = DuplexBudget::new(3.0, 0.6, 0.0);
        b.refill_by_elapsed_us(1_000.0);
        // read: 3000 × 0.6 × 1000 = 1_800_000
        // write: 3000 × 0.4 × 1000 = 1_200_000
        let read_expected = (1_000.0 * 3.0 * 1_000.0 * 0.6) as i64;
        let write_expected = (1_000.0 * 3.0 * 1_000.0 * 0.4) as i64;
        assert_eq!(b.read_tokens(), read_expected.min(b.read_cap()));
        assert_eq!(b.write_tokens(), write_expected.min(b.write_cap()));
    }

    /// Repeated refills must not push tokens above the cap.
    #[test]
    fn refill_capped_at_max() {
        let b = DuplexBudget::new(3.0, DEFAULT_READ_FRACTION, 0.0);
        // Simulate a very long idle period.
        b.refill_by_elapsed_us(10_000_000.0); // 10 seconds
        assert_eq!(b.read_tokens(), b.read_cap(), "read tokens must not exceed cap");
        assert_eq!(b.write_tokens(), b.write_cap(), "write tokens must not exceed cap");
    }

    // -----------------------------------------------------------------------
    // take_read / take_write: basic correctness
    // -----------------------------------------------------------------------

    #[test]
    fn take_read_succeeds_when_tokens_available() {
        let b = full_budget();
        assert!(b.take_read(1024).is_ok());
    }

    #[test]
    fn take_write_succeeds_when_tokens_available() {
        let b = full_budget();
        assert!(b.take_write(1024).is_ok());
    }

    #[test]
    fn take_read_fails_when_empty() {
        let b = empty_budget();
        assert_eq!(b.take_read(1), Err(BandwidthExhausted));
    }

    #[test]
    fn take_write_fails_when_empty() {
        let b = empty_budget();
        assert_eq!(b.take_write(1), Err(BandwidthExhausted));
    }

    /// `take_read` must not consume any tokens when it fails.
    #[test]
    fn take_read_no_tokens_consumed_on_failure() {
        let b = DuplexBudget::new(3.0, 0.5, 0.0);
        b.refill_by_elapsed_us(100.0); // small refill
        let before = b.read_tokens();
        // Request more than available.
        let _ = b.take_read((before + 1) as u64);
        assert_eq!(b.read_tokens(), before, "tokens must be unchanged after failed take_read");
    }

    /// `take_write` must not consume any tokens when it fails.
    #[test]
    fn take_write_no_tokens_consumed_on_failure() {
        let b = DuplexBudget::new(3.0, 0.5, 0.0);
        b.refill_by_elapsed_us(100.0);
        let before = b.write_tokens();
        let _ = b.take_write((before + 1) as u64);
        assert_eq!(b.write_tokens(), before, "tokens must be unchanged after failed take_write");
    }

    // -----------------------------------------------------------------------
    // Read starvation prevention
    // -----------------------------------------------------------------------

    /// Write tokens exhausted → read tokens are unaffected (buckets are independent).
    ///
    /// This is the key duplex guarantee: large write bursts cannot starve reads.
    #[test]
    fn write_exhaustion_does_not_affect_read_tokens() {
        let b = DuplexBudget::new(3.0, DEFAULT_READ_FRACTION, 1.0);
        let read_before = b.read_tokens();

        // Drain the entire write budget.
        let write_cap = b.write_cap();
        let result = b.take_write(write_cap as u64);
        assert!(result.is_ok(), "draining full write cap should succeed");
        assert_eq!(b.write_tokens(), 0);

        // Read budget must be completely unaffected.
        assert_eq!(
            b.read_tokens(),
            read_before,
            "read tokens must not decrease when write tokens are exhausted"
        );

        // Further reads still work.
        assert!(b.take_read(1024).is_ok(), "reads must succeed even when writes are exhausted");
    }

    /// Read tokens exhausted → write tokens are unaffected.
    #[test]
    fn read_exhaustion_does_not_affect_write_tokens() {
        let b = DuplexBudget::new(3.0, DEFAULT_READ_FRACTION, 1.0);
        let write_before = b.write_tokens();

        let read_cap = b.read_cap();
        b.take_read(read_cap as u64).unwrap();
        assert_eq!(b.read_tokens(), 0);
        assert_eq!(b.write_tokens(), write_before);
        assert!(b.take_write(1024).is_ok());
    }

    // -----------------------------------------------------------------------
    // Write deferral lifecycle
    // -----------------------------------------------------------------------

    /// Exhausting write tokens then refilling re-enables take_write.
    ///
    /// This models the "write deferred → next on_layer_start refills → write fires"
    /// lifecycle tested end-to-end in writeback.rs integration tests.
    #[test]
    fn write_deferred_fires_after_refill() {
        let b = empty_budget();

        // Write bucket empty → deferral required.
        assert_eq!(b.take_write(1024), Err(BandwidthExhausted));

        // Simulate one on_layer_start interval (e.g., 500 μs at 3 GB/s → 600 KiB/s).
        b.refill_by_elapsed_us(500.0);

        // After refill, write tokens should be available.
        assert!(b.take_write(1024).is_ok(), "write must succeed after refill");
    }

    /// Read bucket is still usable after a write-deferral cycle.
    #[test]
    fn reads_succeed_during_write_deferral_cycle() {
        let b = DuplexBudget::new(3.0, DEFAULT_READ_FRACTION, 1.0);

        // Drain write bucket.
        b.take_write(b.write_cap() as u64).unwrap();
        // Write is deferred.
        assert_eq!(b.take_write(1024), Err(BandwidthExhausted));

        // Reads must still work (A3 prefetch is not blocked by write saturation).
        assert!(b.take_read(256 * 1024 * 1024).is_ok(), "prefetch reads must not be blocked");
    }

    // -----------------------------------------------------------------------
    // Monotonic token balance
    // -----------------------------------------------------------------------

    /// Sequential takes reduce the balance monotonically until exhaustion.
    #[test]
    fn sequential_takes_reduce_balance_monotonically() {
        let b = DuplexBudget::new(3.0, 0.5, 1.0);
        let chunk = 10_000_000_i64;
        let mut prev = b.read_tokens();
        while b.take_read(chunk as u64).is_ok() {
            let cur = b.read_tokens();
            assert!(cur < prev, "balance must decrease after each successful take");
            prev = cur;
        }
        assert!(b.read_tokens() < chunk, "must exhaust before chunk boundary");
    }
}
