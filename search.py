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
GAIN_TABLE = np.array([0.5, 0.5674, 0.6424, 0.7267, 0.8224, 0.9305, 1.0526, 1.1608])

def quantize_gain(alpha_star):
    """Quantize gain to nearest entry in GAIN_TABLE. Returns (index, quantized_value)."""
    idx = int(np.argmin(np.abs(GAIN_TABLE - alpha_star)))
    return idx, GAIN_TABLE[idx]

def get_best_region(mdct_X, input_pcm, coding_params, buffer):
    """
    Find the best rhythmic prediction region using normalised cross-correlation
    in the time domain (one dot product per lag candidate, one MDCT per range type).

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
    window = WindowForBlockType(LONG, N, N_short)
    windowed_x = window * input_pcm

    search_range = coding_params.search_range

    range_configs = {
        'quarter': coding_params.numSamplesQuarterNote,
        'half':    coding_params.numSamplesHalfBar,
        'bar':     coding_params.numSamplesBar,
    }

    results = {}
    for range_type, qn_multiplier in range_configs.items():
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

    return (None, input_pcm, 0, None, 0, 1.0)
    return (
        best_range, best['pcm_residual'], best['relative_offset'],
        best['mdct_P'], best['alpha_idx'], best['alpha_q']
    )
