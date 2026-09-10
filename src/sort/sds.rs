use super::predictor::Predictor;
use std::time::{Duration, Instant};

const BUCKET_SLACK_SHIFT: usize = 4;
const PREFLIGHT_MAX_POINTS: usize = 1_024;

#[derive(Debug, Clone, Copy)]
pub struct SdsConfig {
    pub sample_size_floor: usize,
    pub near_sorted_threshold: f64,
    pub fallback_threshold: f64,
}

impl Default for SdsConfig {
    fn default() -> Self {
        Self {
            sample_size_floor: 1_000,
            near_sorted_threshold: 0.05,
            fallback_threshold: 0.20,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct PhaseTimes {
    pub build_model: Duration,
    pub diffusion: Duration,
    pub verification: Duration,
    pub correction: Duration,
}

impl PhaseTimes {
    pub fn total(self) -> Duration {
        self.build_model + self.diffusion + self.verification + self.correction
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SdsMetrics {
    pub sample_size: usize,
    pub inversions: usize,
    pub error_rate: f64,
    pub phases: PhaseTimes,
}

pub fn sds_sort(arr: &mut [f64]) -> SdsMetrics {
    sds_sort_with_config(arr, SdsConfig::default())
}

pub fn sds_sort_with_config(arr: &mut [f64], config: SdsConfig) -> SdsMetrics {
    let n = arr.len();
    if n < 2 {
        return SdsMetrics {
            sample_size: n,
            ..SdsMetrics::default()
        };
    }

    if arr.iter().any(|value| !value.is_finite()) {
        let started = Instant::now();
        arr.sort_by(|a, b| a.total_cmp(b));
        return SdsMetrics {
            phases: PhaseTimes {
                correction: started.elapsed(),
                ..PhaseTimes::default()
            },
            ..SdsMetrics::default()
        };
    }

    if let Some(metrics) = simple_preflight(arr) {
        return metrics;
    }

    let mut phases = PhaseTimes::default();
    let (initial_inversions, initial_error_rate) = probe_initial_error(arr);
    if initial_error_rate <= config.near_sorted_threshold {
        let started = Instant::now();
        arr.sort_by(|a, b| a.total_cmp(b));
        phases.correction = started.elapsed();
        return SdsMetrics {
            sample_size: 0,
            inversions: initial_inversions,
            error_rate: initial_error_rate,
            phases,
        };
    }

    let started = Instant::now();
    let model_size = select_sample_size(n, config.sample_size_floor);
    let mut model = build_model(arr, model_size);
    model.sort_unstable_by(|a, b| a.total_cmp(b));
    phases.build_model = started.elapsed();

    let started = Instant::now();
    let bucket_count = select_bucket_count(n);
    let inv_widths = inverse_widths(&model);
    let predictor = Predictor::new(&model, &inv_widths, bucket_count);

    let (_, probe_error_rate) = probe_prediction_error(arr, &predictor, 4_096);
    if probe_error_rate > config.fallback_threshold {
        phases.diffusion = started.elapsed();
        let correction_started = Instant::now();
        arr.sort_unstable_by(|a, b| a.total_cmp(b));
        phases.correction = correction_started.elapsed();
        return SdsMetrics {
            sample_size: model_size,
            inversions: ((probe_error_rate * n as f64).round() as usize).min(n - 1),
            error_rate: probe_error_rate,
            phases,
        };
    }

    let mut histogram = vec![0usize; bucket_count];
    for &value in arr.iter() {
        histogram[predictor.predict(value)] += 1;
    }

    let mut offsets = histogram;
    let mut running = 0usize;
    for slot in &mut offsets {
        let count = *slot;
        *slot = running;
        running += count;
    }

    let mut mapped = vec![0.0; n];
    for &value in arr.iter() {
        let bucket = predictor.predict(value);
        let position = offsets[bucket];
        mapped[position] = value;
        offsets[bucket] = position + 1;
    }
    phases.diffusion = started.elapsed();

    let started = Instant::now();
    let inversions = mapped.windows(2).filter(|pair| pair[0] > pair[1]).count();
    let error_rate = inversions as f64 / n as f64;
    phases.verification = started.elapsed();

    let started = Instant::now();
    mapped.sort_by(|a, b| a.total_cmp(b));
    arr.copy_from_slice(&mapped);
    phases.correction = started.elapsed();

    SdsMetrics {
        sample_size: model_size,
        inversions,
        error_rate,
        phases,
    }
}

/// SDS variant that avoids allocating the full `mapped: Vec<f64>` buffer.
/// It still uses O(bucket_count) auxiliary memory for bucket metadata, and
/// `bucket_count` scales linearly with the input size.
pub fn sds_sort_inplace(arr: &mut [f64]) -> SdsMetrics {
    sds_sort_inplace_with_config(arr, SdsConfig::default())
}

pub fn sds_sort_inplace_with_config(arr: &mut [f64], config: SdsConfig) -> SdsMetrics {
    let n = arr.len();
    if n < 2 {
        return SdsMetrics {
            sample_size: n,
            ..SdsMetrics::default()
        };
    }

    if arr.iter().any(|value| !value.is_finite()) {
        let started = Instant::now();
        arr.sort_by(|a, b| a.total_cmp(b));
        return SdsMetrics {
            phases: PhaseTimes {
                correction: started.elapsed(),
                ..PhaseTimes::default()
            },
            ..SdsMetrics::default()
        };
    }

    if let Some(metrics) = simple_preflight(arr) {
        return metrics;
    }

    let mut phases = PhaseTimes::default();
    let (initial_inversions, initial_error_rate) = probe_initial_error(arr);
    if initial_error_rate <= config.near_sorted_threshold {
        let started = Instant::now();
        arr.sort_by(|a, b| a.total_cmp(b));
        phases.correction = started.elapsed();
        return SdsMetrics {
            sample_size: 0,
            inversions: initial_inversions,
            error_rate: initial_error_rate,
            phases,
        };
    }

    let started = Instant::now();
    let model_size = select_sample_size(n, config.sample_size_floor);
    let mut model = build_model(arr, model_size);
    model.sort_unstable_by(|a, b| a.total_cmp(b));
    phases.build_model = started.elapsed();

    let started = Instant::now();
    let bucket_count = select_bucket_count(n);
    let inv_widths = inverse_widths(&model);
    let predictor = Predictor::new(&model, &inv_widths, bucket_count);

    let (_, probe_error_rate) = probe_prediction_error(arr, &predictor, 4_096);
    if probe_error_rate > config.fallback_threshold {
        phases.diffusion = started.elapsed();
        let correction_started = Instant::now();
        arr.sort_unstable_by(|a, b| a.total_cmp(b));
        phases.correction = correction_started.elapsed();
        return SdsMetrics {
            sample_size: model_size,
            inversions: ((probe_error_rate * n as f64).round() as usize).min(n - 1),
            error_rate: probe_error_rate,
            phases,
        };
    }

    let mut histogram = vec![0usize; bucket_count];
    for &value in arr.iter() {
        histogram[predictor.predict(value)] += 1;
    }

    let mut bucket_starts = vec![0usize; bucket_count];
    let mut running = 0usize;
    for bucket in 0..bucket_count {
        bucket_starts[bucket] = running;
        running += histogram[bucket];
    }

    let mut cursors = bucket_starts.clone();
    let mut bucket_ends = vec![0usize; bucket_count];
    for bucket in 0..bucket_count {
        bucket_ends[bucket] = bucket_starts[bucket] + histogram[bucket];
    }

    for bucket in 0..bucket_count {
        while cursors[bucket] < bucket_ends[bucket] {
            let target_bucket = predictor.predict(arr[cursors[bucket]]);
            if target_bucket == bucket {
                cursors[bucket] += 1;
            } else {
                let target_position = cursors[target_bucket];
                arr.swap(cursors[bucket], target_position);
                cursors[target_bucket] += 1;
            }
        }
    }
    phases.diffusion = started.elapsed();

    let started = Instant::now();
    let inversions = arr.windows(2).filter(|pair| pair[0] > pair[1]).count();
    let error_rate = inversions as f64 / n as f64;
    phases.verification = started.elapsed();

    let started = Instant::now();
    arr.sort_by(|a, b| a.total_cmp(b));
    phases.correction = started.elapsed();

    SdsMetrics {
        sample_size: model_size,
        inversions,
        error_rate,
        phases,
    }
}

fn simple_preflight(arr: &mut [f64]) -> Option<SdsMetrics> {
    let started = Instant::now();
    let is_sorted = arr.windows(2).all(|pair| pair[0] <= pair[1]);
    if is_sorted {
        return Some(SdsMetrics {
            phases: PhaseTimes {
                correction: started.elapsed(),
                ..PhaseTimes::default()
            },
            ..SdsMetrics::default()
        });
    }

    let is_reverse = arr.windows(2).all(|pair| pair[0] >= pair[1]);
    if is_reverse {
        arr.reverse();
        return Some(SdsMetrics {
            inversions: arr.len().saturating_sub(1),
            error_rate: 1.0,
            phases: PhaseTimes {
                correction: started.elapsed(),
                ..PhaseTimes::default()
            },
            ..SdsMetrics::default()
        });
    }

    None
}

fn inverse_widths(model: &[f64]) -> Vec<f64> {
    model
        .windows(2)
        .map(|pair| {
            let width = pair[1] - pair[0];
            if width > 0.0 { 1.0 / width } else { 0.0 }
        })
        .collect()
}

#[inline]
pub fn select_sample_size(n: usize, floor: usize) -> usize {
    ((n as f64).sqrt().round() as usize).max(floor).min(n)
}

#[inline]
pub fn select_bucket_count(n: usize) -> usize {
    n.saturating_add((n >> BUCKET_SLACK_SHIFT).max(1))
}

#[inline]
pub fn select_preflight_points(n: usize) -> usize {
    if n <= 2 {
        return n;
    }
    n.min(PREFLIGHT_MAX_POINTS).max(2)
}

#[inline]
pub fn probe_initial_error(arr: &[f64]) -> (usize, f64) {
    if arr.len() < 2 {
        return (0, 0.0);
    }

    let points = select_preflight_points(arr.len());
    let step = ((arr.len() - 1) / (points - 1)).max(1);
    let mut inversions = 0usize;
    let mut previous = robust_sample(arr, 0);
    let mut used = 1usize;
    let mut index = step;

    while used < points {
        let current_index = index.min(arr.len() - 1);
        let current = robust_sample(arr, current_index);
        inversions += (previous > current) as usize;
        previous = current;
        index = index.saturating_add(step);
        used += 1;
    }

    let pairs = (points - 1).max(1);
    (inversions, inversions as f64 / pairs as f64)
}

#[inline(always)]
pub fn robust_sample(arr: &[f64], index: usize) -> f64 {
    let left = arr[index.saturating_sub(1)];
    let middle = arr[index];
    let right = arr[(index + 1).min(arr.len() - 1)];
    median3(left, middle, right)
}

#[inline(always)]
fn median3(a: f64, b: f64, c: f64) -> f64 {
    if a < b {
        if b < c {
            b
        } else if a < c {
            c
        } else {
            a
        }
    } else if a < c {
        a
    } else if b < c {
        c
    } else {
        b
    }
}

#[inline]
pub fn probe_prediction_error(
    arr: &[f64],
    predictor: &Predictor<'_>,
    max_points: usize,
) -> (usize, f64) {
    if arr.len() < 2 {
        return (0, 0.0);
    }

    let points = arr.len().min(max_points).max(2);
    let step = ((arr.len() - 1) / (points - 1)).max(1);
    let mut used = 1usize;
    let mut index = step;
    let mut inversions = 0usize;
    let mut previous = predictor.predict(arr[0]);

    while used < points {
        let current = predictor.predict(arr[index.min(arr.len() - 1)]);
        inversions += (previous > current) as usize;
        previous = current;
        index = index.saturating_add(step);
        used += 1;
    }

    let pairs = (points - 1).max(1);
    (inversions, inversions as f64 / pairs as f64)
}

pub fn build_model(arr: &[f64], sample_size: usize) -> Vec<f64> {
    if sample_size == arr.len() {
        return arr.to_vec();
    }

    let mut samples = Vec::with_capacity(sample_size);
    let step = arr.len() as f64 / sample_size as f64;
    let mut cursor = 0.0;
    for _ in 0..sample_size {
        let index = (cursor as usize).min(arr.len() - 1);
        samples.push(arr[index]);
        cursor += step;
    }
    samples
}
