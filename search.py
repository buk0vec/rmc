import numpy as np
from mdct import MDCT
from blockswitching import WindowForBlockType, LONG

PRED_MAP = {
    None: 0,
    "quarter": 1,
    "half": 2,
    "bar": 3
}

# AAC LTP gain table: 8 values, transmitted as 3-bit index
# https://github.com/FFmpeg/FFmpeg/blob/master/libavcodec/aactab.c lines 110-113
GAIN_TABLE = np.array([0.570829, 0.696616, 0.813004, 0.911304, 0.984900, 1.067894, 1.194601, 1.369533])

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
    --------
    (range_type, pcm_residual, relative_offset, mdct_P, alpha_idx, alpha_q)
        range_type    : "quarter" / "half" / "bar" or None
        pcm_residual  : time-domain residual (or input_pcm if no prediction)
        relative_offset: int sample offset from center (0 if no prediction)
        mdct_P        : prediction MDCT coefficients (or None if no prediction)
        alpha_idx     : 3-bit gain table index (0 if no prediction)
        alpha_q       : quantized gain scalar (1.0 if no prediction)
    """
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
    
    if beat_unit == 8 and beats_per_bar % 3 == 0:
        # Compound meter: tempo refers to dotted quarter notes (3 eighths)
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

        if search_start < 0 or search_end + N > len(buffer):
            continue

        # Search via normalised cross-correlation (no MDCTs in the loop)
        best_score = -np.inf
        best_sample_offset = center_offset

        for sample_offset in range(search_start, search_end + 1):
            candidate = buffer[sample_offset : sample_offset + N]
            windowed_p = window * candidate
            p_energy = np.dot(windowed_p, windowed_p)
            if p_energy == 0:
                continue
            xcorr = np.dot(windowed_x, windowed_p)
            score = (xcorr ** 2) / p_energy
            if score > best_score:
                best_score = score
                best_sample_offset = sample_offset

        # One MDCT for the winner
        best_candidate = buffer[best_sample_offset : best_sample_offset + N]
        mdct_P_best = MDCT(window * best_candidate, halfN, halfN)[:halfN]
        relative_offset = best_sample_offset - center_offset

        # Optimal block gain (least-squares scalar)
        p_energy_mdct = np.dot(mdct_P_best, mdct_P_best)
        if p_energy_mdct > 0:
            alpha_star = np.dot(mdct_X, mdct_P_best) / p_energy_mdct
            alpha_idx, alpha_q = quantize_gain(alpha_star)
        else:
            alpha_idx, alpha_q = 5, GAIN_TABLE[5]  # ~1.0 fallback

        residual_mdct = mdct_X - alpha_q * mdct_P_best
        mse = np.mean(residual_mdct ** 2)

        results[range_type] = {
            'relative_offset': relative_offset,
            'pcm_residual': input_pcm - alpha_q * best_candidate,
            'mdct_P': mdct_P_best,
            'mse': mse,
            'alpha_idx': alpha_idx,
            'alpha_q': alpha_q,
        }

    if not results:
        return (None, input_pcm, 0, None, 0, 1.0)

    best_range = min(results.keys(), key=lambda x: results[x]['mse'])
    best = results[best_range]

    # Always return the best candidate; per-band enables in WriteDataBlock
    # guarantee prediction is only applied where it actually reduces energy,
    # so no global threshold is needed here.

    return (
        best_range, best['pcm_residual'], best['relative_offset'],
        best['mdct_P'], best['alpha_idx'], best['alpha_q']
    )
