# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
