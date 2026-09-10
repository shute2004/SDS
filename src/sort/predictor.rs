#[derive(Clone, Copy)]
pub struct Predictor<'a> {
    samples: &'a [f64],
    inv_widths: &'a [f64],
    sample_last: usize,
    bucket_last: usize,
    bucket_scale: f64,
    min_sample: f64,
    max_sample: f64,
    inv_span: f64,
}

impl<'a> Predictor<'a> {
    #[inline]
    pub fn new(samples: &'a [f64], inv_widths: &'a [f64], bucket_count: usize) -> Self {
        assert!(!samples.is_empty(), "predictor requires at least one sample");
        assert!(bucket_count > 0, "predictor requires at least one bucket");

        let sample_last = samples.len() - 1;
        let bucket_last = bucket_count - 1;
        let min_sample = samples[0];
        let max_sample = samples[sample_last];
        let span = max_sample - min_sample;
        let inv_span = if span > 0.0 { 1.0 / span } else { 0.0 };
        let bucket_scale = if sample_last > 0 {
            bucket_last as f64 / sample_last as f64
        } else {
            0.0
        };

        Self {
            samples,
            inv_widths,
            sample_last,
            bucket_last,
            bucket_scale,
            min_sample,
            max_sample,
            inv_span,
        }
    }

    #[inline(always)]
    pub fn predict(&self, x: f64) -> usize {
        // SDS itself falls back before constructing a predictor for non-finite
        // input. Keep this guard as a second safety boundary because Predictor
        // is a public module and uses unchecked indexing in its hot path.
        if x.is_nan() {
            return 0;
        }
        if self.bucket_last == 0 || self.sample_last == 0 {
            return 0;
        }
        if x <= self.min_sample {
            return 0;
        }
        if x >= self.max_sample {
            return self.bucket_last;
        }

        let upper = self.upper_bound(x);
        if upper == 0 {
            return 0;
        }
        let lo_idx = upper - 1;
        let lo = unsafe { *self.samples.get_unchecked(lo_idx) };
        let mut model_pos = lo_idx as f64;
        let inv = unsafe { *self.inv_widths.get_unchecked(lo_idx) };
        if inv > 0.0 {
            model_pos += (x - lo) * inv;
        }

        ((model_pos * self.bucket_scale) as usize).min(self.bucket_last)
    }

    #[inline(always)]
    fn upper_bound(&self, x: f64) -> usize {
        debug_assert!(!x.is_nan());
        let estimate = ((x - self.min_sample) * self.inv_span * self.sample_last as f64) as usize;
        let probe = estimate.min(self.sample_last);
        let probe_value = unsafe { *self.samples.get_unchecked(probe) };
        let (mut left, mut right) = if probe_value <= x {
            (probe + 1, self.samples.len())
        } else {
            (0, probe)
        };

        while left < right {
            let mid = (left + right) >> 1;
            let value = unsafe { *self.samples.get_unchecked(mid) };
            if value <= x {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        left
    }
}
