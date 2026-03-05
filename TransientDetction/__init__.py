"""
TransientDetction package - global transient detection for the RMC codec.

Exposes detect_transient_blocks() which analyzes an entire audio signal
and returns a dict mapping block_index -> k_attack for use by the encoder.
"""

import numpy as np
from TransientDetction.src.TD import envelopeFollower, extractTransient, CWT_detect_transients_onset


def _group_consecutive(block_list):
    """Return only the first block of each consecutive run."""
    if len(block_list) == 0:
        return []
    groups = [[block_list[0]]]
    for block in block_list[1:]:
        if block - groups[-1][-1] == 1:
            groups[-1].append(block)
        else:
            groups.append([block])
    return [group[0] for group in groups]


def detect_transient_blocks(audioData, sr, block_size,
                            time_threshold_factor=0.25,
                            cwt_onset_threshold=0.3,
                            cwt_min_distance_ms=50):
    """
    Run transient detection over the entire audio signal and return a dict
    mapping block_index -> k_attack (sub-block index 0..N_SHORT_BLOCKS-1).

    Parameters
    ----------
    audioData : numpy array, shape (nChannels, nSamples) or (nSamples,)
    sr        : sample rate in Hz
    block_size: encoder block size (= nMDCTLines, half the MDCT block)

    Returns
    -------
    dict : {block_index (int): k_attack (int)}
        k_attack is the sub-block index (0-7) where the transient falls,
        used directly by SelectBlockType() in blockswitching.py.
    """
    from blockswitching import N_SHORT_BLOCKS

    if audioData.ndim == 1:
        audioData = np.vstack([audioData, audioData])
    elif audioData.shape[0] > 2:
        audioData = audioData[:2]

    sub_size = block_size // N_SHORT_BLOCKS

    # ------------------------------------------------------------------ #
    # 1. Time-domain envelope detection
    # ------------------------------------------------------------------ #
    fast_env = envelopeFollower(audioData, sr, attack_ms=0.5, release_ms=5.0, mode='peak')
    slow_env = envelopeFollower(audioData, sr, attack_ms=10.0, release_ms=50.0, mode='peak')
    env_diff = fast_env - slow_env

    env_diff_mono = np.maximum(env_diff[0], env_diff[1])
    threshold_time = np.max(env_diff_mono) * time_threshold_factor

    transient_sig = extractTransient(audioData, env_diff, threshold=threshold_time)
    transient_mono = np.maximum(np.abs(transient_sig[0]), np.abs(transient_sig[1]))

    n_blocks = int(np.ceil(len(transient_mono) / block_size))
    blocks_time_multi = [
        i for i in range(n_blocks)
        if np.any(transient_mono[i * block_size: min((i + 1) * block_size, len(transient_mono))] > 0.001)
    ]
    blocks_time = _group_consecutive(blocks_time_multi)

    # ------------------------------------------------------------------ #
    # 2. CWT-based onset detection (optional - falls back if pycwt missing)
    # ------------------------------------------------------------------ #
    blocks_cwt_with_peak = {}  # block_idx -> first peak sample in block
    try:
        import pycwt
        import pycwt.helpers
        # compatibility shim for older pycwt versions
        pycwt.helpers.fft_kwargs = lambda signal: {
            'n': int(2 ** np.ceil(np.log2(len(signal))))
        }
        if not hasattr(np, 'int'):
            np.int = int
            np.float = float
            np.bool = bool
            np.complex = complex

        dt = 1.0 / sr
        dj = 1.0 / 8
        f_min = 50
        f_max = sr / 2
        s0 = 2 * dt
        J = int(np.ceil(np.log2(f_max / f_min) / dj))
        wavelet = pycwt.Morlet(6)

        coeff_l, _, _, _, _, _ = pycwt.cwt(audioData[0], dt, dj, s0, J, wavelet)
        coeff_r, _, _, _, _, _ = pycwt.cwt(audioData[1], dt, dj, s0, J, wavelet)

        cwt_results = CWT_detect_transients_onset(
            coeff_l, coeff_r, sr,
            onset_threshold_factor=cwt_onset_threshold,
            min_distance_ms=cwt_min_distance_ms,
        )
        for peak_sample in cwt_results['peaks']:
            b = int(peak_sample) // block_size
            if b not in blocks_cwt_with_peak:
                blocks_cwt_with_peak[b] = int(peak_sample)
    except Exception:
        pass  # CWT unavailable or failed; rely on time-domain only

    # ------------------------------------------------------------------ #
    # 3. Combine and build {block_idx: k_attack} map
    # ------------------------------------------------------------------ #
    result = {}

    # Time-domain blocks: only know the block, not the exact sample.
    # Use k_attack = 0 (transient assumed at start of block).
    for b in blocks_time:
        if b not in result:
            result[b] = 0

    # CWT blocks: have the peak sample, so compute k_attack precisely.
    for b, peak_sample in blocks_cwt_with_peak.items():
        k = (peak_sample % block_size) // sub_size
        k = min(k, N_SHORT_BLOCKS - 1)
        if b not in result:
            result[b] = k
        else:
            result[b] = min(result[b], k)  # keep earliest sub-block

    return result
