import numpy as np
import scipy.fft as _scipy_fft
from scipy.fft import irfft as _irfft, next_fast_len as _next_fast_len, rfft as _rfft

from blockswitching import LONG, START, STOP, MEDIUM, WindowForBlockType
from mdct import MDCT

PRED_MAP = {
    None: 0,
    "quarter": 1,
    "half": 2,
    "bar": 3,
    "2bar": 4,
    "4bar": 5,
}

# 3-bit (8-entry) gain table for complex prediction — log-spaced over [0.06, 2.50].
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
    # Clip everything except the new incomplete right tail. With variable block
    # sizes (block switching), a previous block may have left a larger incomplete
    # tail; clipping buf[:-halfN] retroactively cleans those up too.
    # The new tail (buf[-halfN:]) is left unclipped — it hasn't received the
    # next block's left-half OLA contribution yet.
    buf[:-halfN] = np.clip(buf[:-halfN], -1, 1)


def phase_idx_to_radians(idx):
    """Map 4-bit phase index (0-15) back to angle in [-π, π)."""
    return (idx * np.pi / 8.0) - np.pi


FRACTIONAL_CENTER_IDX = 2


def get_best_region(
    mdct_X, input_pcm, coding_params, buffer, block_type=LONG,
    cascade_a=None, cascade_b=None, sfBands=None
):
    """Select the best prediction region using per-band complex (MDCT+MDST) gains."""
    halfN = coding_params.nMDCTLines
    N = 2 * halfN
    N_short = 2 * coding_params.nMDCTLines_short

    # Derive block-specific dimensions from cascade kwargs
    if cascade_a is not None and cascade_b is not None:
        ca, cb = cascade_a, cascade_b
        N_t = ca + cb
        halfN_t = N_t // 2
    else:
        ca = cb = halfN
        N_t = N
        halfN_t = halfN

    # Use per-block sfBands so scoring, LS, and band loops all work natively at halfN_t.
    if sfBands is None:
        sfBands = coding_params.sfBands

    window = WindowForBlockType(
        block_type, N, N_short,
        cascade_a=ca if block_type in (STOP, MEDIUM) else None,
        cascade_b=cb if block_type in (START, MEDIUM) else None,
    )
    windowed_x = window * input_pcm

    _mdct_X_raw = mdct_X.real if np.iscomplexobj(mdct_X) else mdct_X

    search_range = coding_params.search_range
    nBands = sfBands.nBands

    lo_arr = sfBands.lowerLine
    nLines_arr = sfBands.nLines

    def evaluate_candidate(candidate_time: np.ndarray):
        mdct_P_cplx = MDCT(window * candidate_time, ca, cb, return_complex=True)[:halfN_t]
        mdct_P = mdct_P_cplx.real
        mdst_P = mdct_P_cplx.imag
        mdct_X_real = _mdct_X_raw
        # Vectorized per-band sums via reduceat (bands tile [0, halfN))
        PP = mdct_P * mdct_P
        DD = mdst_P * mdst_P
        XP = mdct_X_real * mdct_P
        XD = mdct_X_real * mdst_P
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
        residual_mdct = mdct_X_real - a_line * mdct_P + b_line * mdst_P
        pcm_residual = input_pcm

        # Vectorized scoring via reduceat
        residual_real = (
            residual_mdct.real if np.iscomplexobj(residual_mdct) else residual_mdct
        )
        XX = np.add.reduceat(mdct_X_real * mdct_X_real, lo_arr)
        RR = np.add.reduceat(residual_real * residual_real, lo_arr)
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
            "mdct_P": mdct_P_cplx.real,
            "mdst_P": mdct_P_cplx.imag,
            "alpha_idxs": alpha_idxs,
            "alpha_qs": alpha_qs,
            "phase_idxs": phase_idxs,
            "mse": mse,
            "score": improvement_score,
        }

    window_sq = window**2
    f_xcorr = windowed_x * window

    # Pre-compute FFTs of the fixed filter signals once; reused across all range types.
    _filter_len = len(f_xcorr)
    _fft_n = _next_fast_len(2 * search_range + N_t + _filter_len - 1)
    _F_xcorr = _rfft(f_xcorr,   n=_fft_n)
    _F_wsq   = _rfft(window_sq, n=_fft_n)

    # Compute how many samples of real audio are in the buffer (zeros at start).
    buffer_fill = getattr(coding_params, "buffer_fill", len(buffer))
    valid_from = max(0, len(buffer) - buffer_fill)

    range_types = ("quarter", "half", "bar", "2bar", "4bar")
    results = {}
    for range_type in range_types:
        if range_type == "bar" and coding_params.numSamplesBar == 0:
            continue
        if range_type == "2bar" and getattr(coding_params, "numSamples2Bar", 0) == 0:
            continue
        if range_type == "4bar" and getattr(coding_params, "numSamples4Bar", 0) == 0:
            continue
        qn_multiplier = pred_type_to_samples(range_type, coding_params)
        center_offset = len(buffer) - halfN_t - qn_multiplier

        # Clamp search_start to the oldest valid (non-zero) sample in the buffer.
        search_start = max(center_offset - search_range, valid_from)
        search_end = center_offset + search_range
        if search_start > search_end or search_end + N_t > len(buffer):
            continue

        s0 = max(search_start, 0)
        s1 = min(search_end + N_t, len(buffer))
        buf_slice = buffer[s0:s1]

        _vlen = len(buf_slice) - _filter_len + 1
        xcorr       = _irfft(_rfft(buf_slice,      n=_fft_n) * np.conj(_F_xcorr), n=_fft_n)[:_vlen]
        power_xcorr = _irfft(_rfft(buf_slice**2,   n=_fft_n) * np.conj(_F_wsq),   n=_fft_n)[:_vlen]

        # Slice-relative offset k → absolute buffer offset k + s0.
        offsets = np.arange(search_start, search_end + 1)
        valid_m = (offsets >= 0) & (offsets + N_t <= len(buffer))
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
            candidate_block = buf_slice[k_off : k_off + N_t].copy()
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
