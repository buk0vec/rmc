"""
TransientDetction package - global transient detection for the RMC codec.

Exposes detect_transient_blocks() which analyzes an entire audio signal
and returns a dict mapping block_index -> k_attack for use by the encoder.
"""

import numpy as np
from TransientDetction.src.TD import envelopeFollower, extractTransient


def detect_transient_blocks(audioData, sr, block_size,
                            time_threshold_factor=0.25):
    """
    Run transient detection over the entire audio signal and return a dict
    mapping block_index -> k_attack (sub-block index 0..N_SHORT_BLOCKS-1).

    Uses extractTransient (envelope follower difference) to find transient
    samples, then locates the peak within each block to compute k_attack
    precisely.

    Parameters
    ----------
    audioData          : numpy array, shape (nChannels, nSamples) or (nSamples,)
    sr                 : sample rate in Hz
    block_size         : encoder block size (= nMDCTLines, half the MDCT block)
    time_threshold_factor : fraction of peak env_diff to use as threshold

    Returns
    -------
    dict : {block_index (int): k_attack (int)}
    """
    from blockswitching import N_SHORT_BLOCKS

    if audioData.ndim == 1:
        audioData = np.vstack([audioData, audioData])
    elif audioData.shape[0] > 2:
        audioData = audioData[:2]

    sub_size = block_size // N_SHORT_BLOCKS

    fast_env = envelopeFollower(audioData, sr, attack_ms=0.5, release_ms=5.0, mode='peak')
    slow_env = envelopeFollower(audioData, sr, attack_ms=10.0, release_ms=50.0, mode='peak')
    env_diff = fast_env - slow_env

    env_diff_mono = np.maximum(env_diff[0], env_diff[1])
    threshold_time = np.max(env_diff_mono) * time_threshold_factor

    transient_sig = extractTransient(audioData, env_diff, threshold=threshold_time)
    transient_mono = np.maximum(np.abs(transient_sig[0]), np.abs(transient_sig[1]))

    n_blocks = int(np.ceil(len(transient_mono) / block_size))
    result = {}
    for b in range(n_blocks):
        block_slice = transient_mono[b * block_size : min((b + 1) * block_size, len(transient_mono))]
        if np.any(block_slice > 0):
            # compute k_attack from the peak transient sample's position
            peak_in_block = int(np.argmax(block_slice))
            k = min(peak_in_block // sub_size, N_SHORT_BLOCKS - 1)
            result[b] = k

    return result
