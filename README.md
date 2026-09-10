# SDS — Stochastic Diffusion Sort

SDS is an experimental sorting pipeline written in Rust. It uses a sampled value model to predict coarse destination regions, diffuses values into those regions, measures remaining disorder, and applies a final corrective sort.

The implementation also includes an in-place variant that avoids the full auxiliary mapped array.

## Structure

- `src/sort/predictor.rs` — crate-internal sampled interpolation predictor.
- `src/sort/sds.rs` — model building, diffusion, verification, fallback, and in-place variant.
- `src/main.rs` — minimal executable example.
- `tests/sds.rs` — correctness checks against Rust's total-order sort.

The predictor is not a public construction API. Its constructor establishes the length, ordering, and finite-value invariants required by the interpolation model, and the prediction path uses bounds-checked slice access. Non-finite user input is handled by the public SDS sorting entry points through total-order fallback sorting.

## Build and Test

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets
cargo run --release
```

The same format, lint, and test checks are defined in GitHub Actions.

SDS is an experimental algorithm project, not a claim that it universally outperforms standard library sorting. Performance depends on input distribution, size, fallback behavior, and hardware.

## License

Source-visible, all rights reserved. See [LICENSE](LICENSE).
