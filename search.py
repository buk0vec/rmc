import numpy as np
from scipy.signal import correlate as _fft_correlate

from blockswitching import LONG, WindowForBlockType
from features import (
    COMPLEX_PREDICTION,
    PRED_EXTENDED_RANGE,
)
from mdct import MDCT

PRED_MAP = {
    None: 0,
    "quarter": 1,
    "half": 2,
    "bar": 3,
    "2bar": 4,
    "4bar": 5,
}

# TODO: Investigate whether our gain tables should actually be going this low - certainly
# low gains mean that we shouldn't waste time with prediction, right?

# Gain table, 13 values, 4-bit index. Tuned to cover low-magnitude wins and
# extend headroom for high-band prediction while increasing resolution through
# the 0.55–1.3 corridor where most optimal gains land.
GAIN_TABLE = np.array(
    [0.06, 0.09, 0.13, 0.18, 0.24, 0.32, 0.42, 0.55, 0.75, 1.02, 1.36, 1.85, 2.50]
)

# 3-bit (8-entry) gain table for COMPLEX_PREDICTION — log-spaced over same range.
# Saves 1 bit/enabled band vs the 13-entry table; precision sweep shows +1.1 dB
# quantization penalty vs optimal, down from +0.75 dB for 4b, acceptable trade.
COMPLEX_GAIN_TABLE = np.exp(np.linspace(np.log(0.06), np.log(2.50), 8))


def pred_type_to_samples(pred_type, coding_params):
    """Map prediction range type string to sample offset."""

    def _require(attr_name: str) -> int:
        if not hasattr(coding_params, attr_name):
            raise AttributeError(
                f"coding_params missing required attribute '{attr_name}'"
            )
        return getattr(coding_params, attr_name)

    mapping = {
        "quarter": _require("numSamplesQuarterNote"),
        "half": _require("numSamplesHalfBar"),
        "bar": _require("numSamplesBar"),
    }
    if hasattr(coding_params, "numSamples2Bar"):
        mapping["2bar"] = getattr(coding_params, "numSamples2Bar")
    if hasattr(coding_params, "numSamples4Bar"):
        mapping["4bar"] = getattr(coding_params, "numSamples4Bar")
    try:
        return mapping[pred_type]
    except KeyError as exc:
        raise KeyError(f"Unknown prediction range type: {pred_type}") from exc


def update_search_buffer(buf, new_data, halfN, N):
    """Shift search buffer by halfN, overlap-add new_data (length N), clip to [-1, 1]."""
    buf[:-halfN] = buf[halfN:]
    buf[-halfN:] = 0
    buf[-N:] += new_data
    buf[-N:] = np.clip(buf[-N:], -1, 1)


def phase_idx_to_radians(idx):
    """Map 4-bit phase index (0-15) back to angle in [-π, π)."""
    return (idx * np.pi / 8.0) - np.pi


FRACTIONAL_CENTER_IDX = 2


def get_best_region(mdct_X, input_pcm, coding_params, buffer, block_type=LONG):
    """Select the best prediction region, optionally refining with fractional delay."""
    halfN = coding_params.nMDCTLines
    N = 2 * halfN
    N_short = 2 * coding_params.nMDCTLines_short
    window = WindowForBlockType(block_type, N, N_short)
    windowed_x = window * input_pcm

    search_range = coding_params.search_range
    sfBands = coding_params.sfBands
    nBands = sfBands.nBands

    lo_arr = sfBands.lowerLine
    nLines_arr = sfBands.nLines

    def evaluate_candidate(candidate_time: np.ndarray):
        mdct_P_cplx = MDCT(window * candidate_time, halfN, halfN, return_complex=True)[
            :halfN
        ]
        mdct_P = mdct_P_cplx.real
        if COMPLEX_PREDICTION:
            mdst_P = mdct_P_cplx.imag
            # Vectorized per-band sums via reduceat (bands tile [0, halfN))
            PP = mdct_P * mdct_P
            DD = mdst_P * mdst_P
            XP = mdct_X * mdct_P
            XD = mdct_X * mdst_P
            PD = mdct_P * mdst_P
            sum_PP = np.add.reduceat(PP, lo_arr)
            sum_DD = np.add.reduceat(DD, lo_arr)
            sum_XP = np.add.reduceat(XP, lo_arr)
            sum_XD = np.add.reduceat(XD, lo_arr)
            sum_PD = np.add.reduceat(PD, lo_arr)
            # Exact per-band LS: solve [sum_PP,-sum_PD;-sum_PD,sum_DD][a;b]=[sum_XP;-sum_XD]
            det = sum_PP * sum_DD - sum_PD**2
            valid_zz = det > 1e-12
            safe_det = np.where(valid_zz, det, 1.0)
            a_opt = np.where(
                valid_zz, (sum_DD * sum_XP - sum_PD * sum_XD) / safe_det, 0.0
            )
            b_opt = np.where(
                valid_zz, (sum_PD * sum_XP - sum_PP * sum_XD) / safe_det, 0.0
            )
            mag_v = np.where(
                valid_zz, np.sqrt(a_opt**2 + b_opt**2), COMPLEX_GAIN_TABLE[4]
            )
            phase_v = np.where(valid_zz, np.arctan2(b_opt, a_opt), 0.0)
            # Vectorized gain quantization (3-bit COMPLEX_GAIN_TABLE)
            alpha_idxs = np.argmin(
                np.abs(COMPLEX_GAIN_TABLE[np.newaxis, :] - mag_v[:, np.newaxis]), axis=1
            ).astype(np.int32)
            alpha_qs = COMPLEX_GAIN_TABLE[alpha_idxs]
            # Vectorized phase quantization
            phase_norm = np.angle(np.exp(1j * phase_v))
            phase_idxs = (
                np.round((phase_norm + np.pi) * 8.0 / np.pi).astype(np.int32)
            ) % 16
            phase_qs = phase_idxs * (np.pi / 8.0) - np.pi

            a_line = np.repeat(alpha_qs * np.cos(phase_qs), nLines_arr)
            b_line = np.repeat(alpha_qs * np.sin(phase_qs), nLines_arr)
            residual_mdct = mdct_X - a_line * mdct_P + b_line * mdst_P
            pcm_residual = input_pcm
        else:
            p_energy_mdct = np.dot(mdct_P, mdct_P)
            if p_energy_mdct > 0:
                alpha_star = np.dot(mdct_X, mdct_P) / p_energy_mdct
                base_idx = int(np.argmin(np.abs(GAIN_TABLE - alpha_star)))
                base_gain = float(GAIN_TABLE[base_idx])
            else:
                base_idx, base_gain = 5, float(GAIN_TABLE[5])
            phase_idxs = 128

            alpha_idxs, alpha_qs = base_idx, base_gain
            residual_mdct = mdct_X - alpha_qs * mdct_P
            pcm_residual = input_pcm - alpha_qs * candidate_time

        # Vectorized scoring via reduceat
        XX = np.add.reduceat(mdct_X * mdct_X, lo_arr)
        RR = np.add.reduceat(residual_mdct * residual_mdct, lo_arr)
        orig_rms_v = np.sqrt(XX / nLines_arr)
        res_rms_v = np.sqrt(RR / nLines_arr)
        valid_v = orig_rms_v > 0
        ratio_v = np.where(valid_v, res_rms_v / np.where(valid_v, orig_rms_v, 1.0), 1.0)
        improvement_v = np.maximum(0.0, 1.0 - ratio_v)
        weight_sum_v = float((nLines_arr * valid_v).sum())
        mse = (
            float(np.dot(ratio_v * valid_v, nLines_arr)) / weight_sum_v
            if weight_sum_v > 0
            else 1.0
        )
        improvement_score = float(np.dot(improvement_v, nLines_arr))

        return {
            "pcm_residual": pcm_residual,
            "mdct_P": mdct_P,
            "mdst_P": mdst_P if COMPLEX_PREDICTION else None,
            "alpha_idxs": alpha_idxs,
            "alpha_qs": alpha_qs,
            "phase_idxs": phase_idxs,
            "mse": mse,
            "score": improvement_score,
        }

    window_sq = window**2
    f_xcorr = windowed_x * window

    # Compute how many samples of real audio are in the buffer (zeros at start).
    # Prediction for range X starts as soon as X samples have been accumulated.
    buffer_fill = getattr(coding_params, "buffer_fill", len(buffer))
    valid_from = max(0, len(buffer) - buffer_fill)

    range_types = (
        ("quarter", "half", "bar", "2bar", "4bar")
        if PRED_EXTENDED_RANGE
        else ("quarter", "half", "bar")
    )
    results = {}
    for range_type in range_types:
        if range_type == "bar" and coding_params.numSamplesBar == 0:
            continue
        if range_type == "2bar" and getattr(coding_params, "numSamples2Bar", 0) == 0:
            continue
        if range_type == "4bar" and getattr(coding_params, "numSamples4Bar", 0) == 0:
            continue
        qn_multiplier = pred_type_to_samples(range_type, coding_params)
        center_offset = len(buffer) - halfN - qn_multiplier
        # Clamp search_start to the oldest valid (non-zero) sample in the buffer.
        search_start = max(center_offset - search_range, valid_from)
        search_end = center_offset + search_range
        if search_start > search_end or search_end + N > len(buffer):
            continue

        s0 = max(search_start, 0)
        s1 = min(search_end + N, len(buffer))
        buf_slice = buffer[s0:s1]

        xcorr = _fft_correlate(buf_slice, f_xcorr, mode="valid", method="fft")
        power_xcorr = _fft_correlate(
            buf_slice**2, window_sq, mode="valid", method="fft"
        )

        # Slice-relative offset k → absolute buffer offset k + s0.
        offsets = np.arange(search_start, search_end + 1)
        valid_m = (offsets >= 0) & (offsets + N <= len(buffer))
        offsets = offsets[valid_m]
        if offsets.size == 0:
            continue
        k = offsets - s0

        xc = xcorr[k]
        pe = power_xcorr[k]
        valid_pe = pe > 0
        safe_pe = np.where(valid_pe, pe, 1.0)
        scores = np.where(valid_pe, xc**2 / safe_pe, -np.inf)

        if not np.any(np.isfinite(scores)):
            continue
        primary_offset = int(offsets[int(np.argmax(scores))])

        def evaluate_offset(sample_offset: int):
            k_off = sample_offset - s0
            candidate_block = buf_slice[k_off : k_off + N].copy()
            evaluation_local = evaluate_candidate(candidate_block)
            evaluation_local["relative_offset"] = sample_offset - center_offset
            return evaluation_local

        results[range_type] = evaluate_offset(primary_offset)

    if not results:
        return (None, input_pcm, 0, None, None, 0, 1.0, 128)

    best_range = max(
        results.keys(), key=lambda x: (results[x].get("score", 0.0), -results[x]["mse"])
    )
    best = results[best_range]
    return (
        best_range,
        best["pcm_residual"],
        best["relative_offset"],
        best["mdct_P"],
        best.get("mdst_P"),
        best["alpha_idxs"],
        best["alpha_qs"],
        best["phase_idxs"],
    )
