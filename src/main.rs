use sds_sort::sort::sds_sort;

fn main() {
    let mut values = vec![9.0, 1.5, 7.2, 2.1, 0.3, 6.8, 3.7, 8.4, 5.5, 4.6];
    let metrics = sds_sort(&mut values);

    println!("SDS sorted: {values:?}");
    println!(
        "phase_ms: build={:.3}, diffusion={:.3}, verify={:.3}, correction={:.3}",
        metrics.phases.build_model.as_secs_f64() * 1_000.0,
        metrics.phases.diffusion.as_secs_f64() * 1_000.0,
        metrics.phases.verification.as_secs_f64() * 1_000.0,
        metrics.phases.correction.as_secs_f64() * 1_000.0,
    );
    println!("error_rate={:.6}, sample_size={}, inversions={}", metrics.error_rate, metrics.sample_size, metrics.inversions);
}
