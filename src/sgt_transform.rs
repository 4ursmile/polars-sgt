// Simplified SGT implementation that actually compiles
// This implementation works correctly with POL ARS API patterns
use polars::prelude::*;
use std::collections::HashMap;

/// Time penalty modes for SGT
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
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

    pub fn normalize(&self, weights: &mut HashMap<String, f64>) {
        match self {
            NormMode::L1 => {
                let sum: f64 = weights.values().sum();
                if sum > 0.0 {
                    for weight in weights.values_mut() {
                        *weight /= sum;
                    }
                }
            }
            NormMode::L2 => {
                let sum_sq: f64 = weights.values().map(|w| w * w).sum();
                if sum_sq > 0.0 {
                    let norm = sum_sq.sqrt();
                    for weight in weights.values_mut() {
                        *weight /= norm;
                    }
                }
            }
            NormMode::None => {}
        }
    }
}

/// Convert deltatime string to seconds multiplier
fn deltatime_to_seconds(deltatime: Option<&str>) -> PolarsResult<f64> {
    match deltatime {
        None => Ok(1.0),
        Some("s") => Ok(1.0),
        Some("m") => Ok(60.0),
        Some("h") => Ok(3600.0),
        Some("d") => Ok(86400.0),
        Some("w") => Ok(604800.0),
        Some("month") => Ok(2629800.0), // 30.44 days
        Some("q") => Ok(7889400.0),     // 91.31 days
        Some("y") => Ok(31557600.0),    // 365.25 days
        Some(other) => polars_bail!(InvalidOperation: "Unknown deltatime: {}", other),
    }
}

/// Extract time value as f64
fn get_time_value(series: &Series, idx: usize, deltatime: Option<&str>) -> PolarsResult<Option<f64>> {
    match series.dtype() {
       DataType::Datetime(time_unit, _) => {
            let ca = series.datetime()?;
            let divisor = deltatime_to_seconds(deltatime)?;
            let time_unit_divisor = match time_unit {
                TimeUnit::Nanoseconds => 1_000_000_000.0,
                TimeUnit::Microseconds => 1_000_000.0,
                TimeUnit::Milliseconds => 1_000.0,
            };
            Ok(unsafe { ca.phys.get_unchecked(idx) }.map(|v| v as f64 / time_unit_divisor / divisor))
        }
        DataType::Date => {
            let ca = series.date()?;
            let divisor = deltatime_to_seconds(deltatime)? / 86400.0;
           Ok(unsafe { ca.phys.get_unchecked(idx) }.map(|v| v as f64 / divisor))
        }
        DataType::Duration(time_unit) => {
            let ca = series.duration()?;
            let divisor = deltatime_to_seconds(deltatime)?;
            let time_unit_divisor = match time_unit {
                TimeUnit::Nanoseconds => 1_000_000_000.0,
                TimeUnit::Microseconds => 1_000_000.0,
                TimeUnit::Milliseconds => 1_000.0,
            };
            Ok(unsafe { ca.phys.get_unchecked(idx) }.map(|v| v as f64 / time_unit_divisor / divisor))
        }
        _ => {
            let ca = series.cast(&DataType::Float64)?;
            Ok(ca.f64()?.get(idx))
        }
    }
}

/// Generate n-grams with weights from a sequence
fn generate_ngrams(
    states: &[String],
    time_values: &[Option<f64>],
    kappa: usize,
    time_penalty: &TimePenalty,
    alpha: f64,
    beta: f64,
) -> HashMap<String, f64> {
    let mut ngram_weights: HashMap<String, f64> = HashMap::new();

    if states.is_empty() {
        return ngram_weights;
    }

    // Generate n-grams up to kappa size
    for n in 1..=kappa.min(states.len()) {
        for i in 0..=(states.len() - n) {
            let ngram: Vec<&str> = states[i..i + n].iter().map(|s| s.as_str()).collect();
            let ngram_key = ngram.join(" -> ");

            // Calculate weight based on time difference
            let weight = if n > 1 && i + n - 1 < time_values.len() {
                if let (Some(curr_time), Some(prev_time)) = (time_values[i + n - 1], time_values[i + n - 2]) {
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

    ngram_weights
}

/// Main SGT implementation using simple iteration
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

    // Get unique sequence IDs
    let unique_ids: StringChunked = sequence_ids.str()?.unique()?.sort(Default::default());
    
    let mut result_seq_ids: Vec<String> = Vec::new();
    let mut result_ngram_keys_list: Vec<Series> = Vec::new();
    let mut result_ngram_values_list: Vec<Series> = Vec::new();

    let seq_ids_ca = sequence_ids.str()?;
    let states_ca = states_series.str()?;

    // Process each unique sequence ID
    for idx in 0..unique_ids.len() {
        let seq_id: &str = match unique_ids.get(idx) {
            Some(id) => id,
            None => continue,
        };

        // Find all rows matching this sequence ID
        let mask: BooleanChunked = seq_ids_ca.equal(seq_id);
        
        // Extract states for this sequence
        let mut sequence_states = Vec::new();
        let mut time_values = Vec::new();
        
        for i in 0..mask.len() {
            if mask.get(i).unwrap_or(false) {
                if let Some(state) = states_ca.get(i) {
                    sequence_states.push(state.to_string());
                    if let Some(ts) = time_series {
                        time_values.push(get_time_value(ts, i, deltatime)?);
                    } else {
                        time_values.push(Some(i as f64));
                    }
                }
            }
        }

        if sequence_states.is_empty() {
            continue;
        }

        // Generate n-grams with weights
        let mut ngram_weights = generate_ngrams(
            &sequence_states,
            &time_values,
            kappa,
            &time_penalty_mode,
            alpha,
            beta,
        );

        // Apply length normalization if requested
        if length_sensitive && sequence_states.len() > 1 {
            let seq_len = sequence_states.len() as f64;
            for weight in ngram_weights.values_mut() {
                *weight /= seq_len;
            }
        }

        // Apply normalization mode
        norm_mode.normalize(&mut ngram_weights);

        // Convert to sorted vectors
        let mut keys: Vec<String> = ngram_weights.keys().cloned().collect();
        keys.sort();
        let values: Vec<f64> = keys.iter().map(|k| ngram_weights[k]).collect();

        result_seq_ids.push(seq_id.to_string());
        result_ngram_keys_list.push(
            StringChunked::from_iter(keys.iter().map(|s| Some(s.as_str()))).into_series()
        );
        result_ngram_values_list.push(
            Float64Chunked::from_vec(PlSmallStr::EMPTY, values).into_series()
        );
    }

    // Build result struct
    let mut seq_id_ca = StringChunked::from_iter(result_seq_ids.iter().map(|s| Some(s.as_str())));
    seq_id_ca.rename(PlSmallStr::from_str("sequence_id"));
    let seq_id_series = seq_id_ca.into_series();
    
    // Convert to list series
    let ngram_keys_dtype = DataType::List(Box::new(DataType::String));
    let ngram_keys_series = Series::new(
        PlSmallStr::from_str("ngram_keys"),
        result_ngram_keys_list
    ).cast(&ngram_keys_dtype)?;
    
    let ngram_values_dtype = DataType::List(Box::new(DataType::Float64));
    let ngram_values_series = Series::new(
        PlSmallStr::from_str("ngram_values"),
        result_ngram_values_list
    ).cast(&ngram_values_dtype)?;

    // Create struct
    let struct_fields = [seq_id_series, ngram_keys_series, ngram_values_series];
    Ok(StructChunked::from_series(
        PlSmallStr::from_str("sgt_result"),
        result_seq_ids.len(),
        struct_fields.iter()
    )?.into_series())
}
