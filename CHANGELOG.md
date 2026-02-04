# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-02-04

### Changed
- **Major Performance Optimization**: Optimized SGT transform for billion-row scale with O(1) time weight lookups via cumulative product prefix arrays.
- **Speed Improvements**: Optimized Rust implementation with fast `exp`/`pow` approximations, pre-allocated buffers, and elimination of post-sort overhead.
- **Enhanced `sgt_transform_df`**: 
    - Returns a single merged wide-format DataFrame by default.
    - Automatically prefixes feature names with group values (e.g., `sgt_buy_login`).
    - Uses efficient reduce-join for merging multi-group analysis.
    - Full support for Polar's LazyFrame and streaming engine.

### Fixed
- **Time Weight Correctness**: Fixed weight calculation for `kappa > 2` to correctly accumulate time penalties across *all* individual transitions in an n-gram.
- **Numerical Stability**: Implemented periodic renormalization and zero-trap protection for weighted products to prevent underflow in very long sequences.

### Added
- Comprehensive README documentation with spotlights on high-level APIs, scalability, and grouped analysis usage.


## [0.2.5] - 2026-02-04

### Added
- `use_tqdm` parameter to `sgt_transform_df` to control progress bar visibility.
- `keep_original_name` parameter to `sgt_transform_df` to optionally restore original sequence ID names.
- Support for multiple columns in `sequence_id_col` in `sgt_transform_df` (automatically concatenates and splits).

### Fixed
- `sgt_transform_df` now correctly handles `group_cols=None` by processing the entire DataFrame.
- `sgt_transform_df` now correctly filters subsets dynamically based on unique values of `group_cols` instead of hardcoded columns.

## [0.2.0] - 2026-02-02

### Added
- Parallel processing support with `rayon` for SGT transform.
- Support for custom output struct field names via `sequence_id_name` and `state_name` parameters.

### Changed
- **Major Performance Optimization**: Rewrote SGT transform to use O(n) group-based indexing instead of O(n*m) scanning. Throughput increased to ~1.4M+ records/second.
- **Struct Field Rename (BREAKING)**: Renamed `ngram_values` field in the output struct to `value` for consistency with current Polars version and parameter names.

### Fixed
- Performance bottleneck on large datasets (10M+ records).
