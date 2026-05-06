"""
Sub-block Peak Energy (SPE) transient detection.

Implements the SPE method from:
  Fan et al., "Transient Detection Methods for Audio Coding,"
  AES 155th Convention, New York, 2023.

Algorithm summary (Section 3.4 + Layer 5 extension):
  - Buffer: N samples (left N/2 = previous block, right N/2 = current block)
  - High-pass filter at 2 kHz (lowered from paper's 8 kHz for broader coverage)
  - Five layers of sub-block sizes: N/2, N/4, N/8, N/16, N/32
  - For each layer j, compare peak of each right-half sub-block to the
    immediately preceding sub-block (cascading within the right half).
  - Flag condition: curr_peak * T[j] > prev_peak  AND  curr_peak > zero_threshold
  - A transient is declared if ANY layer fires.

Threshold selection (empirically validated on castanets, glockenspiel, harpsichord):
  - L1 (N/2  = 1024 samples): T=0.40  → ratio must exceed 2.5×
  - L2 (N/4  =  512 samples): T=0.40  → ratio must exceed 2.5×
  - L3 (N/8  =  256 samples): T=0.07  → ratio must exceed 14.3×
  - L4 (N/16 =  128 samples): T=0.07  → ratio must exceed 14.3×
  - L5 (N/32 =   64 samples): T=0.07  → ratio must exceed 14.3×
  Layer 1 dominates for these signals (all transients visible at coarse scale).
  Layers 2-5 add coverage for brief, localised spikes that Layer 1 misses.
"""

import numpy as np
import scipy.io.wavfile
import scipy.signal

# Per-layer thresholds T[j].  A transient fires when: curr_peak * T[j] > prev_peak
# L1 (N/2  = 1024 samples): ratio > 2.5×   — coarse, catches large block-level jumps
# L2 (N/4  =  512 samples): ratio > 2.5×   — catches attacks mid-block
# L3 (N/8  =  256 samples): ratio > 14.3×  — catches sub-block spikes (one SHORT-block)
# L4 (N/16 =  128 samples): ratio > 14.3×  — 128-sample resolution
# L5 (N/32 =   64 samples): ratio > 14.3×  — finest resolution, matches SHORT half-window
# Validated empirically on castanets / glockenspiel / harpsichord (EBU-SQAM).
THRESHOLDS = (0.4, 0.4, 0.07, 0.07, 0.07)

# Paper uses 1500/32768 ≈ 0.046 at 8 kHz HPF cutoff. Lowering cutoff to 2 kHz
# passes more attack energy from low-frequency instruments (harpsichord, plucked
# strings); zero_threshold of 750/32768 avoids noise false-positives at that cutoff.
# Validated on castanets (88), glockenspiel (29), harpsichord (10).
ZERO_THRESHOLD = 750.0 / 32768.0    # ≈ 0.0229
HPF_CUTOFF = 2000.0                 # Hz; lower than paper's 8 kHz


def _design_highpass(sr: int, cutoff: float = HPF_CUTOFF, order: int = 4):
    """4th-order Butterworth high-pass SOS filter."""
    nyq = sr / 2.0
    wn = min(cutoff / nyq, 0.9999)
    return scipy.signal.butter(order, wn, btype='high', output='sos')


def _sub_peak(buf: np.ndarray, start: int, length: int) -> float:
    """Peak absolute amplitude in buf[start : start+length]."""
    return float(np.max(np.abs(buf[start: start + length])))


def spe_block(hp_buf: np.ndarray,
              N: int,
              thresholds: tuple = THRESHOLDS,
              zero_threshold: float = ZERO_THRESHOLD):
    """
    Run SPE on a single high-pass-filtered N-sample buffer.

    Layout: buf[0 : N//2] = previous half-block
            buf[N//2 : N] = current half-block

    Runs all 5 layers coarse-to-fine. Each firing layer overwrites the previous
    best_offset, so the finest (highest-j) layer that fires sets the final position.

    Returns
    -------
    fired : bool
    sample_offset : int
        Byte offset of the first firing sub-block within the current half,
        at the finest resolution layer that fired. 0 if not fired.
        Resolution: 64 samples (L5), 128 (L4), 256 (L3), 512 (L2), 1024 (L1).
    """
    half = N >> 1
    fired = False
    best_offset = 0

    for j in range(1, 6):
        sub = N >> j            # sub-block size: N/2, N/4, N/8, N/16, N/32
        n_curr = 1 << (j - 1)  # sub-blocks in current half: 1, 2, 4, 8, 16
        T = thresholds[j - 1]
        prev_peak = _sub_peak(hp_buf, half - sub, sub)
        for k in range(n_curr):
            curr_peak = _sub_peak(hp_buf, half + k * sub, sub)
            if curr_peak > zero_threshold and curr_peak * T > prev_peak:
                best_offset = k * sub   # finer layers always overwrite coarser
                fired = True
                break                  # first firing sub-block per layer
            prev_peak = curr_peak

    return fired, best_offset


def spe_block_details(hp_buf: np.ndarray,
                      N: int,
                      thresholds: tuple = THRESHOLDS,
                      zero_threshold: float = ZERO_THRESHOLD):
    """
    Like spe_block but also returns per-layer diagnostics for plotting.

    Returns
    -------
    flagged : bool
    ratios : list of float   -- max curr/prev peak ratio per layer
    layer_flags : list of bool -- whether each layer fired
    """
    half = N >> 1
    flagged = False
    ratios = []
    layer_flags = []
    for j in range(1, 6):
        sub = N >> j
        n_curr = 1 << (j - 1)
        T = thresholds[j - 1]
        prev_peak = _sub_peak(hp_buf, half - sub, sub)
        max_ratio = 0.0
        layer_flag = False
        for k in range(n_curr):
            curr_peak = _sub_peak(hp_buf, half + k * sub, sub)
            if prev_peak > 1e-10:
                ratio = curr_peak / prev_peak
            else:
                ratio = curr_peak / zero_threshold if curr_peak > 0 else 0.0
            if ratio > max_ratio:
                max_ratio = ratio
            if curr_peak > zero_threshold and curr_peak * T > prev_peak:
                layer_flag = True
                flagged = True
            prev_peak = curr_peak
        ratios.append(max_ratio)
        layer_flags.append(layer_flag)
    return flagged, ratios, layer_flags


def detectTransientsSPE(audioPath: str,
                        nMDCTLines: int = 1024,
                        cutoff: float = HPF_CUTOFF,
                        thresholds: tuple = THRESHOLDS,
                        zero_threshold: float = ZERO_THRESHOLD,
                        verbose: bool = False) -> np.ndarray:
    """
    Detect transients using SPE. Returns sorted block indices.

    Each codec block advances `nMDCTLines` samples; the SPE buffer is
    2*nMDCTLines samples (previous half + current half).
    """
    sr, raw = scipy.io.wavfile.read(audioPath)

    if raw.dtype == np.int16:
        audio = raw.astype(np.float64) / 32768.0
    elif raw.dtype == np.int32:
        audio = raw.astype(np.float64) / 2147483648.0
    else:
        audio = raw.astype(np.float64)

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    sos = _design_highpass(sr, cutoff)
    zi = scipy.signal.sosfilt_zi(sos) * audio[0]

    N = nMDCTLines * 2
    half = nMDCTLines

    n_samples = len(audio)
    n_blocks = int(np.ceil(n_samples / half))

    prev_hp = np.zeros(half, dtype=np.float64)
    transient_blocks = []

    for block_idx in range(n_blocks):
        start = block_idx * half
        end = min(start + half, n_samples)
        chunk = audio[start:end]
        if len(chunk) < half:
            chunk = np.pad(chunk, (0, half - len(chunk)))

        curr_hp, zi = scipy.signal.sosfilt(sos, chunk, zi=zi)

        buf = np.concatenate([prev_hp, curr_hp])
        fired, _ = spe_block(buf, N, thresholds, zero_threshold)
        if fired:
            transient_blocks.append(block_idx)

        prev_hp = curr_hp

    if verbose:
        print(f"  SPE blocks ({len(transient_blocks)}): {transient_blocks}")

    return np.array(transient_blocks, dtype=int)


def detectTransientsSPESamples(audioPath: str,
                                nMDCTLines: int = 1024,
                                cutoff: float = HPF_CUTOFF,
                                thresholds: tuple = THRESHOLDS,
                                zero_threshold: float = ZERO_THRESHOLD,
                                verbose: bool = False) -> list:
    """
    Detect transients using SPE. Returns event dicts with sample positions.

    Uses the finest-resolution firing sub-block (L5 = 64-sample resolution)
    to compute an exact-ish sample position within each detected block.
    This matches the event dict format expected by xrmc.py's AC2A path.

    Returns
    -------
    list of {"sample_index": int, "block": int}
        sample_index : absolute sample position of the onset
        block        : block index (sample_index // nMDCTLines)
    """
    sr, raw = scipy.io.wavfile.read(audioPath)

    if raw.dtype == np.int16:
        audio = raw.astype(np.float64) / 32768.0
    elif raw.dtype == np.int32:
        audio = raw.astype(np.float64) / 2147483648.0
    else:
        audio = raw.astype(np.float64)

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    sos = _design_highpass(sr, cutoff)
    zi = scipy.signal.sosfilt_zi(sos) * audio[0]

    N = nMDCTLines * 2
    half = nMDCTLines

    n_samples = len(audio)
    n_blocks = int(np.ceil(n_samples / half))

    prev_hp = np.zeros(half, dtype=np.float64)
    events = []

    for block_idx in range(n_blocks):
        start = block_idx * half
        end = min(start + half, n_samples)
        chunk = audio[start:end]
        if len(chunk) < half:
            chunk = np.pad(chunk, (0, half - len(chunk)))

        curr_hp, zi = scipy.signal.sosfilt(sos, chunk, zi=zi)

        buf = np.concatenate([prev_hp, curr_hp])
        fired, sample_offset = spe_block(buf, N, thresholds, zero_threshold)
        if fired:
            sample_index = block_idx * half + sample_offset
            events.append({
                "sample_index": sample_index,
                "block": block_idx,
            })

        prev_hp = curr_hp

    if verbose:
        print(f"  SPE events ({len(events)}): {[e['sample_index'] for e in events]}")

    return events
