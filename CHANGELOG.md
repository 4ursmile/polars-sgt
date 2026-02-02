# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-02-02

### Added
- Parallel processing support with `rayon` for SGT transform.
- Support for custom output struct field names via `sequence_id_name` and `state_name` parameters.

### Changed
- **Major Performance Optimization**: Rewrote SGT transform to use O(n) group-based indexing instead of O(n*m) scanning. Throughput increased to ~1.4M+ records/second.
- **Struct Field Rename (BREAKING)**: Renamed `ngram_values` field in the output struct to `value` for consistency with current Polars version and parameter names.

### Fixed
- Performance bottleneck on large datasets (10M+ records).
