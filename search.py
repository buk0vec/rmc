import numpy as np

PRED_MAP = {
    None: 0,
    "quarter": 1,
    "half": 2,
    "bar": 3
}

INV_PRED_MAP = {
    0: None,
    1: "quarter",
    2: "half",
    3: "bar"
}

<<<<<<< Updated upstream
def get_best_region(input_block, coding_params, buffer, threshold=0.8):
    """
    Find the best rhythmic prediction region for the input block.
    
    Parameters:
    -----------
    input_block : np.ndarray
        2048 samples to encode
    coding_params : object
        Contains tempo, sample_rate, etc.
    buffer : np.ndarray
        Previous audio history (1 bar + padding)
    padding : int
        Search range padding in samples (default: 200)
    threshold : float
        MSE threshold ratio for rejecting prediction (default: 0.8)
    
    Returns:
=======
def pred_type_to_samples(pred_type, coding_params):
    """Map prediction range type string to sample offset."""
    return {'quarter': coding_params.numSamplesQuarterNote,
            'half':    coding_params.numSamplesHalfBar,
            'bar':     coding_params.numSamplesBar}[pred_type]


def update_search_buffer(buf, new_data, halfN, N):
    """Shift search buffer by halfN, overlap-add new_data (length N), clip to [-1, 1]."""
    buf[:-halfN] = buf[halfN:]
    buf[-halfN:] = 0
    buf[-N:] += new_data
    buf[-N:] = np.clip(buf[-N:], -1, 1)


def quantize_gain(alpha_star):
    """Quantize gain to nearest entry in GAIN_TABLE. Returns (index, quantized_value)."""
    idx = int(np.argmin(np.abs(GAIN_TABLE - alpha_star)))
    return idx, GAIN_TABLE[idx]

def get_search_offsets(time_signature):
    """
    Generate meter related search offsets based on meter.
    
    Parameters:
    -----------
    time_signature : tuple
        (beats_per_bar, beat_unit), e.g., (4, 4), (3, 4), (12, 8), (7, 4)
    
    Returns:
    --------
    dict: {range_type: num_beat_units}
        Mapping from range type name to number of beat units to look backward
    
    Examples:
    ---------
    (4, 4) → {'quarter': 1, 'half': 2, 'bar': 4}
    (3, 4) → {'quarter': 1, 'half': None, 'bar': 3}  # no half-bar in 3/4
    (12, 8) → {'quarter': 3, 'half': 6, 'bar': 12}  # compound meter
    (7, 4) → {'quarter': 1, 'half': 3, 'bar': 7}    # asymmetric
    """
    beats_per_bar, beat_unit = time_signature
    offsets = {}
    
    # COMPOUND METERS (e.g., 6/8, 9/8, 12/8)
    if beat_unit == 8 and beats_per_bar % 3 == 0:
        compound_beats = beats_per_bar // 3
        
        # 1 compound beat (3 eighth notes)
        offsets['quarter'] = 3
        
        # Half bar (if even number of compound beats)
        if compound_beats % 2 == 0 and compound_beats > 2:
            offsets['half'] = beats_per_bar // 2
        else:
            offsets['half'] = None
        
        # Full bar
        offsets['bar'] = beats_per_bar
    
    # SIMPLE METERS (e.g., 2/4, 3/4, 4/4, 5/4, 7/4)
    else:
        # 1 beat
        offsets['quarter'] = 1
        
        # Half bar (only if even beats_per_bar or >3)
        if beats_per_bar % 2 == 0:
            offsets['half'] = beats_per_bar // 2
        elif beats_per_bar > 3:
            offsets['half'] = beats_per_bar // 2  # e.g., 5/4 → 2 beats, 7/4 → 3 beats
        else:
            offsets['half'] = None  # no half-bar in 3/4
        
        # Full bar
        offsets['bar'] = beats_per_bar
    
    return offsets

def get_best_region(mdct_X, input_pcm, coding_params, buffer, block_type=LONG):
    """
    Find the best rhythmic prediction region using normalised cross-correlation
    in the time domain (one dot product per lag candidate, one MDCT per range type).
    Now meter-aware: adapts search ranges based on time signature.

    Returns
>>>>>>> Stashed changes
    --------
    tuple: (range_type, data_to_encode, relative_offset)
        range_type: str - "quarter", "half", "bar", or None
        data_to_encode: np.ndarray - residual if prediction used, otherwise input_block
        relative_offset: int - offset from center (0 if no prediction used)
    """
<<<<<<< Updated upstream
    
    # Calculate baseline energy
    original_energy = np.var(input_block)
    
    block_size = len(input_block)
    
    # Initialize results storage
    results = {}
    
    #search range 
    padding = coding_params.search_range
=======
    halfN = coding_params.nMDCTLines
    N = 2 * halfN
    N_short = 2 * coding_params.nMDCTLines_short
    window = WindowForBlockType(block_type, N, N_short)
    windowed_x = window * input_pcm

    search_range = coding_params.search_range
    
    # Get time signature from codingParams (default to 4/4 if not set)
    time_signature = getattr(coding_params, 'time_signature', (4, 4))
    
    # Get meter-aware search offsets
    meter_offsets = get_search_offsets(time_signature)
    
    # Calculate samples per beat unit (e.g., quarter note in 4/4, eighth note in 6/8)
    tempo = coding_params.tempo
    sample_rate = coding_params.sampleRate
    samples_per_quarter_note = int((60.0 / tempo) * sample_rate)
    
    # Adjust for beat_unit (4 = quarter note, 8 = eighth note, etc.)
    beats_per_bar, beat_unit = time_signature
    samples_per_beat_unit = int(samples_per_quarter_note * (4.0 / beat_unit))
    if beat_unit == 8 and beats_per_bar % 3 == 0:
        # Compound meter: tempo refers to dotted quarter notes (3 eighths)
        # So one "beat unit" (eighth note) = (1/3) of the tempo-marked beat
        samples_per_beat_unit = int(samples_per_quarter_note * (4.0 / 8))  # eighth note duration
        # But tempo is marked in compound beats (dotted quarters)
        # So we need to treat the tempo as "compound beats per minute"
        # 1 compound beat = 3 eighth notes = 1.5 quarter notes
        # If tempo = 107 compound beats/min, then quarter note tempo = 107 * 1.5 = 160.5
        # But we're given tempo as compound beats, so:
        samples_per_compound_beat = int((60.0 / tempo) * sample_rate)
        samples_per_beat_unit = samples_per_compound_beat // 3  # each eighth note
    else:
        # Simple meter: tempo refers to the beat_unit
        samples_per_beat_unit = int(samples_per_quarter_note * (4.0 / beat_unit))

    # For each rhythmic lag defined by meter, search ±search_range samples around
    # the center offset using normalized cross-correlation (one dot product per candidate).
    # Only the best candidate per range type gets an MDCT (avoids N_search MDCTs per type).
    results = {}
    for range_type in ('quarter', 'half', 'bar'):
        # Skip this range type if meter doesn't support it (e.g., no half-bar in 3/4)
        beat_units = meter_offsets.get(range_type)
        if beat_units is None:
            continue
        
        # Calculate center offset in samples
        qn_multiplier = beat_units * samples_per_beat_unit
        center_offset = len(buffer) - qn_multiplier
        search_start = center_offset - search_range
        search_end = center_offset + search_range
>>>>>>> Stashed changes

    # Search each range type
    range_configs = {
        'quarter': coding_params.numSamplesQuarterNote ,
        'half': coding_params.numSamplesHalfBar,
        'bar': coding_params.numSamplesBar
    }


    for range_type, qn_multiplier in range_configs.items():
        
        # Determine center offset
        center_offset = qn_multiplier
        
        # Define search window using negative indices from end of buffer
        # -center_offset is the nominal prediction point (e.g. exactly 1 quarter note ago)
        search_start = -center_offset - padding
        search_end = -center_offset + padding

        # Ensure we don't go out of buffer bounds
        if search_start < -len(buffer):
            continue  # Skip this range if not enough history in buffer

        # Sliding correlation search
        best_correlation = -np.inf
        best_sample_offset = None

        for sample_offset in range(search_start, search_end + 1):
            # Extract candidate region from buffer
            candidate_region = buffer[sample_offset : sample_offset + block_size]

            # Calculate normalized correlation
            correlation = np.corrcoef(input_block, candidate_region)[0, 1]
            # Track best match
            if correlation > best_correlation:
                best_correlation = correlation
                best_sample_offset = sample_offset

        if best_sample_offset is None:
            best_sample_offset = -center_offset

        # Calculate residual and MSE
        predicted_block = buffer[best_sample_offset : best_sample_offset + block_size]
        residual = input_block - predicted_block
        mse = np.mean(residual ** 2)

        # Calculate relative offset from nominal center for block header encoding.
        # Decoder reconstructs as buffer[-start_offset + relative_offset], so
        # relative_offset = best_sample_offset + center_offset gives the same index.
        relative_offset = best_sample_offset + center_offset
        
        # Store results
        results[range_type] = {
            'sample_offset': best_sample_offset,
            'relative_offset': relative_offset,
            'residual': residual,
            'mse': mse,
            'correlation': best_correlation
        }
    
    # Check if we found any valid results
    if not results:
        # No valid search performed, encode input directly
        return (None, input_block, 0)
    
    # Find best range across all types
    best_range = min(results.keys(), key=lambda x: results[x]['mse'])
<<<<<<< Updated upstream
    best_mse = results[best_range]['mse']
    best_residual = results[best_range]['residual']
    best_relative_offset = results[best_range]['relative_offset']
    
    # Evaluate against threshold
    if best_mse < threshold * original_energy:
        # Prediction is helpful - encode residual
        return (best_range, best_residual, best_relative_offset)
    else:
        # Prediction doesn't help - encode original input
        return (None, input_block, 0)
=======
    best = results[best_range]

    # Always return the best candidate; per-band enables in WriteDataBlock
    # guarantee prediction is only applied where it actually reduces energy,
    # so no global threshold is needed here.

    return (
        best_range, best['pcm_residual'], best['relative_offset'],
        best['mdct_P'], best['alpha_idx'], best['alpha_q']
    )
>>>>>>> Stashed changes
