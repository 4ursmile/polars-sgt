// High-performance SGT implementation optimized for 100M+ records
// Uses group-based indexing (O(n)) and parallel processing with Rayon
use polars::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;

/// Time penalty modes for SGT
#[derive(Debug, Clone, Copy)]
pub enum TimePenalty {
    Inverse,
    Exponential,
    Linear,
    Power,
    None,
}

impl TimePenalty {
    pub fn from_str(s: &str) -> PolarsResult<Self> {
        match s {
            "inverse" => Ok(TimePenalty::Inverse),
            "exponential" => Ok(TimePenalty::Exponential),
            "linear" => Ok(TimePenalty::Linear),
            "power" => Ok(TimePenalty::Power),
            "none" => Ok(TimePenalty::None),
            _ => polars_bail!(InvalidOperation: "Unknown time_penalty: {}", s),
        }
    }

    #[inline(always)]
    pub fn apply(&self, time_diff: f64, alpha: f64, beta: f64) -> f64 {
        if time_diff == 0.0 {
            return 1.0;
        }
        match self {
            TimePenalty::Inverse => alpha / time_diff,
            TimePenalty::Exponential => (-alpha * time_diff).exp(),
            TimePenalty::Linear => (1.0 - alpha * time_diff).max(0.0),
            TimePenalty::Power => 1.0 / time_diff.powf(beta),
            TimePenalty::None => 1.0,
        }
    }
}

/// Normalization modes for SGT
#[derive(Debug, Clone, Copy)]
pub enum NormMode {
    L1,
    L2,
    None,
}

impl NormMode {
    pub fn from_str(s: &str) -> PolarsResult<Self> {
        match s {
            "l1" => Ok(NormMode::L1),
            "l2" => Ok(NormMode::L2),
            "none" => Ok(NormMode::None),
            _ => polars_bail!(InvalidOperation: "Unknown mode: {}", s),
        }
    }

    #[inline(always)]
    pub fn normalize(&self, weights: &mut Vec<f64>) {
        match self {
            NormMode::L1 => {
                let sum: f64 = weights.iter().sum();
                if sum > 0.0 {
                    for weight in weights.iter_mut() {
                        *weight /= sum;
                    }
                }
            }
            NormMode::L2 => {
                let sum_sq: f64 = weights.iter().map(|w| w * w).sum();
                if sum_sq > 0.0 {
                    let norm = sum_sq.sqrt();
                    for weight in weights.iter_mut() {
                        *weight /= norm;
                    }
                }
            }
            NormMode::None => {}
        }
    }
}

/// Convert deltatime string to seconds multiplier
#[inline(always)]
fn deltatime_to_seconds(deltatime: Option<&str>) -> PolarsResult<f64> {
    match deltatime {
        None => Ok(1.0),
        Some("s") => Ok(1.0),
        Some("m") => Ok(60.0),
        Some("h") => Ok(3600.0),
        Some("d") => Ok(86400.0),
        Some("w") => Ok(604800.0),
        Some("month") => Ok(2629800.0),
        Some("q") => Ok(7889400.0),
        Some("y") => Ok(31557600.0),
        Some(other) => polars_bail!(InvalidOperation: "Unknown deltatime: {}", other),
    }
}

/// Extract time values as f64 for a batch of indices
#[inline]
fn extract_time_values(
    series: &Series,
    indices: &[usize],
    deltatime: Option<&str>,
) -> PolarsResult<Vec<Option<f64>>> {
    let divisor = deltatime_to_seconds(deltatime)?;
    
    match series.dtype() {
        DataType::Datetime(time_unit, _) => {
            let ca = series.datetime()?;
            let time_unit_divisor = match time_unit {
                TimeUnit::Nanoseconds => 1_000_000_000.0,
                TimeUnit::Microseconds => 1_000_000.0,
                TimeUnit::Milliseconds => 1_000.0,
            };
            Ok(indices
                .iter()
                .map(|&i| unsafe { ca.phys.get_unchecked(i) }.map(|v| v as f64 / time_unit_divisor / divisor))
                .collect())
        }
        DataType::Date => {
            let ca = series.date()?;
            let date_divisor = divisor / 86400.0;
            Ok(indices
                .iter()
                .map(|&i| unsafe { ca.phys.get_unchecked(i) }.map(|v| v as f64 / date_divisor))
                .collect())
        }
        DataType::Duration(time_unit) => {
            let ca = series.duration()?;
            let time_unit_divisor = match time_unit {
                TimeUnit::Nanoseconds => 1_000_000_000.0,
                TimeUnit::Microseconds => 1_000_000.0,
                TimeUnit::Milliseconds => 1_000.0,
            };
            Ok(indices
                .iter()
                .map(|&i| unsafe { ca.phys.get_unchecked(i) }.map(|v| v as f64 / time_unit_divisor / divisor))
                .collect())
        }
        _ => {
            let ca = series.cast(&DataType::Float64)?;
            let f64_ca = ca.f64()?;
            Ok(indices.iter().map(|&i| f64_ca.get(i)).collect())
        }
    }
}

/// Result for a single sequence
struct SequenceResult {
    seq_id: String,
    ngram_keys: Vec<String>,
    ngram_values: Vec<f64>,
}

/// Generate n-grams with weights from a sequence (optimized version)
#[inline]
fn generate_ngrams_fast(
    states: &[&str],
    time_values: &[Option<f64>],
    kappa: usize,
    time_penalty: TimePenalty,
    alpha: f64,
    beta: f64,
) -> (Vec<String>, Vec<f64>) {
    if states.is_empty() {
        return (Vec::new(), Vec::new());
    }

    // Estimate capacity: n-grams up to kappa for sequence of length L
    // Total n-grams ≈ L + (L-1) + ... + (L-kappa+1)
    let estimated_capacity = states.len() * kappa.min(states.len());
    let mut ngram_weights: HashMap<String, f64> = HashMap::with_capacity(estimated_capacity);

    // Generate n-grams up to kappa size
    let max_n = kappa.min(states.len());
    for n in 1..=max_n {
        for i in 0..=(states.len() - n) {
            // Build n-gram key efficiently
            let ngram_key = if n == 1 {
                states[i].to_string()
            } else {
                states[i..i + n].join(" -> ")
            };

            // Calculate weight based on time difference
            let weight = if n > 1 && i + n - 1 < time_values.len() {
                if let (Some(curr_time), Some(prev_time)) = 
                    (time_values[i + n - 1], time_values[i + n - 2])
                {
                    let time_diff = (curr_time - prev_time).abs();
                    time_penalty.apply(time_diff, alpha, beta)
                } else {
                    1.0
                }
            } else {
                1.0
            };

            *ngram_weights.entry(ngram_key).or_insert(0.0) += weight;
        }
    }

    // Convert to sorted vectors
    let mut keys: Vec<String> = ngram_weights.keys().cloned().collect();
    keys.sort_unstable();
    let values: Vec<f64> = keys.iter().map(|k| ngram_weights[k]).collect();
    
    (keys, values)
}

/// Process a single sequence group
#[inline]
fn process_sequence(
    seq_id: &str,
    indices: &[usize],
    states_ca: &StringChunked,
    time_series: Option<&Series>,
    kappa: usize,
    length_sensitive: bool,
    time_penalty: TimePenalty,
    norm_mode: NormMode,
    alpha: f64,
    beta: f64,
    deltatime: Option<&str>,
) -> PolarsResult<Option<SequenceResult>> {
    // Extract states for this sequence using direct index access
    let states: Vec<&str> = indices
        .iter()
        .filter_map(|&i| states_ca.get(i))
        .collect();

    if states.is_empty() {
        return Ok(None);
    }

    // Extract time values
    let time_values = if let Some(ts) = time_series {
        extract_time_values(ts, indices, deltatime)?
    } else {
        // Use index positions as time
        indices.iter().map(|&i| Some(i as f64)).collect()
    };

    // Generate n-grams with weights
    let (keys, mut values) = generate_ngrams_fast(
        &states,
        &time_values,
        kappa,
        time_penalty,
        alpha,
        beta,
    );

    // Apply length normalization if requested
    if length_sensitive && states.len() > 1 {
        let seq_len = states.len() as f64;
        for weight in values.iter_mut() {
            *weight /= seq_len;
        }
    }

    // Apply normalization mode
    norm_mode.normalize(&mut values);

    Ok(Some(SequenceResult {
        seq_id: seq_id.to_string(),
        ngram_keys: keys,
        ngram_values: values,
    }))
}

/// High-performance SGT implementation using group-based indexing and parallel processing
#[allow(clippy::too_many_arguments)]
pub fn impl_sgt_transform(
    inputs: &[Series],
    kappa: i64,
    length_sensitive: bool,
    mode: &str,
    time_penalty: &str,
    alpha: f64,
    beta: f64,
    deltatime: Option<&str>,
    sequence_id_name: Option<&str>,
    state_name: Option<&str>,
) -> PolarsResult<Series> {
    if inputs.len() < 2 {
        polars_bail!(InvalidOperation: "sgt_transform requires at least sequence_id and state columns");
    }

    let sequence_ids = inputs[0].cast(&DataType::String)?;
    let states_series = &inputs[1];
    let time_series = if inputs.len() > 2 {
        Some(&inputs[2])
    } else {
        None
    };

    let kappa = kappa as usize;
    let time_penalty_mode = TimePenalty::from_str(time_penalty)?;
    let norm_mode = NormMode::from_str(mode)?;

    let seq_ids_ca = sequence_ids.str()?;
    let states_ca = states_series.str()?;

    // OPTIMIZATION 1: Build group index in O(n) - single pass
    // This replaces the O(n*m) nested loop
    let mut groups: HashMap<&str, Vec<usize>> = HashMap::new();
    for (idx, seq_id) in seq_ids_ca.iter().enumerate() {
        if let Some(id) = seq_id {
            groups.entry(id).or_default().push(idx);
        }
    }

    // OPTIMIZATION 2: Process groups in parallel with Rayon
    let results: Vec<PolarsResult<Option<SequenceResult>>> = groups
        .par_iter()
        .map(|(seq_id, indices)| {
            process_sequence(
                seq_id,
                indices,
                states_ca,
                time_series,
                kappa,
                length_sensitive,
                time_penalty_mode,
                norm_mode,
                alpha,
                beta,
                deltatime,
            )
        })
        .collect();

    // Collect successful results
    let mut result_seq_ids: Vec<String> = Vec::with_capacity(groups.len());
    let mut result_ngram_keys_list: Vec<Series> = Vec::with_capacity(groups.len());
    let mut result_ngram_values_list: Vec<Series> = Vec::with_capacity(groups.len());

    for result in results {
        if let Some(seq_result) = result? {
            result_seq_ids.push(seq_result.seq_id);
            result_ngram_keys_list.push(
                StringChunked::from_iter(seq_result.ngram_keys.iter().map(|s| Some(s.as_str())))
                    .into_series(),
            );
            result_ngram_values_list.push(
                Float64Chunked::from_vec(PlSmallStr::EMPTY, seq_result.ngram_values).into_series(),
            );
        }
    }

    // Sort by sequence ID for deterministic output
    let mut indexed: Vec<(usize, &String)> = result_seq_ids.iter().enumerate().collect();
    indexed.sort_by(|a, b| a.1.cmp(b.1));
    
    let sorted_seq_ids: Vec<String> = indexed.iter().map(|(i, _)| result_seq_ids[*i].clone()).collect();
    let sorted_keys: Vec<Series> = indexed.iter().map(|(i, _)| result_ngram_keys_list[*i].clone()).collect();
    let sorted_values: Vec<Series> = indexed.iter().map(|(i, _)| result_ngram_values_list[*i].clone()).collect();

    // Use parameter names for struct fields (fallback to defaults)
    let seq_field_name = sequence_id_name.unwrap_or("sequence_id");
    let _state_field_name = state_name.unwrap_or("state"); // Reserved for future use

    // Build result struct
    let mut seq_id_ca = StringChunked::from_iter(sorted_seq_ids.iter().map(|s| Some(s.as_str())));
    seq_id_ca.rename(PlSmallStr::from_str(seq_field_name));
    let seq_id_series = seq_id_ca.into_series();

    // Convert to list series
    let ngram_keys_dtype = DataType::List(Box::new(DataType::String));
    let ngram_keys_series = Series::new(PlSmallStr::from_str("ngram_keys"), sorted_keys)
        .cast(&ngram_keys_dtype)?;

    // Renamed from ngram_values to value
    let ngram_values_dtype = DataType::List(Box::new(DataType::Float64));
    let ngram_values_series = Series::new(PlSmallStr::from_str("value"), sorted_values)
        .cast(&ngram_values_dtype)?;

    // Create struct
    let struct_fields = [seq_id_series, ngram_keys_series, ngram_values_series];
    Ok(StructChunked::from_series(
        PlSmallStr::from_str("sgt_result"),
        sorted_seq_ids.len(),
        struct_fields.iter(),
    )?
    .into_series())
}
