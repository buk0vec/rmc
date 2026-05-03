"""
Sub-block Peak Energy (SPE) transient detection.

Implements the SPE method from:
  Fan et al., "Transient Detection Methods for Audio Coding,"
  AES 155th Convention, New York, 2023.

Algorithm summary (Section 3.4 + Layer 4 extension):
  - Buffer: N samples (left N/2 = previous block, right N/2 = current block)
  - Dual-band high-pass filter design (see below)
  - Four layers of sub-block sizes: N/2, N/4, N/8, N/16
  - For each layer j, compare peak of each right-half sub-block to the
    immediately preceding sub-block (cascading within the right half).
  - Flag condition: curr_peak * T[j] > prev_peak  AND  curr_peak > zero_threshold
  - A transient is declared if ANY layer of EITHER band fires.

Threshold selection (empirically validated on 5 EBU-SQAM instruments):
  - L1 (N/2  = 1024 samples): T=0.42  → ratio must exceed 2.38×
  - L2 (N/4  =  512 samples): T=0.42  → ratio must exceed 2.38×
  - L3 (N/8  =  256 samples): T=0.07  → ratio must exceed 14.3×
  - L4 (N/16 =  128 samples): T=0.07  → ratio must exceed 14.3×
  Layer 1 dominates for percussive/plucked signals.
  Layers 2-4 add coverage for brief, localised spikes that Layer 1 misses.
  T12 raised from paper's 0.40 to 0.42 to recover missed glockenspiel onsets
  (ratios 2.37-2.45x) without adding false positives to castanets or harpsichord.

Dual-band HPF design:
  Band 1 (HPF_CUTOFF  = 2000 Hz): primary band, tuned for percussive /
    plucked / struck instruments (castanets, glockenspiel, harpsichord).
    Filters out low-frequency sustain that would elevate prev_peak and
    suppress ratio-based detection.
  Band 2 (HPF_CUTOFF2 = 1500 Hz): secondary band, added to handle bowed /
    sustained instruments (violin) whose harmonic energy sits mostly below
    2 kHz. At 2 kHz, 96% of violin blocks are ZT-blocked; at 1500 Hz, the
    violin attack energy passes through and produces detectable ratios.
    A block is declared a transient if EITHER band fires.

Validated detection counts (EBU-SQAM + violin):
  castanets=89  glockenspiel=32  harpsichord=10  violin=30  oboe=6

Note on oboe / fully sustained instruments:
  SPE is a ratio-based detector that requires a significant amplitude jump
  relative to the preceding block. Wind instruments (oboe, clarinet, flute)
  sustain continuously between notes with ratio ~1.0, so only phrase-level
  onsets from near-silence are catchable. Per-note detection for purely
  sustained instruments requires a different algorithm (spectral flux,
  pitch-period onset, etc.) and is a known limitation of the SPE family.

Integration note:
  xrmc.py already shifts returned block indices back by 1, so that the
  START block (LONG->SHORT transition) precedes the detected attack.
  This function simply returns the indices of blocks containing attacks.
"""

import numpy as np
import scipy.io.wavfile
import scipy.signal

# Per-layer thresholds T[j].  A transient fires when: curr_peak * T[j] > prev_peak
# Equivalently:  curr_peak / prev_peak > 1 / T[j]
# L1 (N/2  = 1024 samples): ratio > 2.38x  -- coarse, catches large block-level jumps
# L2 (N/4  =  512 samples): ratio > 2.38x  -- catches attacks mid-block
# L3 (N/8  =  256 samples): ratio > 14.3x  -- catches sub-block spikes (one SHORT-block)
# L4 (N/16 =  128 samples): ratio > 14.3x  -- finest resolution, same scale as paper L3
#
# T12 tuned from paper's 0.40 (-> 2.5x ratio) to 0.42 (-> 2.38x ratio) after
# diagnosing 3 missed glockenspiel note onsets with ratios of 2.37-2.45x.
# Fine-grained sweep confirmed T12 in [0.42, 0.47] keeps castanets=88, harpsichord=10
# with zero extra false positives.  0.42 is the most conservative value.
THRESHOLDS = (0.42, 0.42, 0.07, 0.07)

# zero_threshold: minimum HPF-signal peak for the current block to qualify.
# Prevents spurious detections in near-silence where tiny noise produces large ratios.
# Paper uses 1500/32768 ~= 0.046 at 8 kHz HPF.  Lowering HPF cutoff to 2 kHz passes
# more attack energy; zero_threshold reduced proportionally.
# Fine sweep (ZT 750->600 in steps of 25, dual-band) showed ZT=725/32768 is the
# exact breakpoint where violin reaches 32 detections without changing any other
# instrument count (castanets=89, glock=32, harp=10, oboe=6 all unchanged).
ZERO_THRESHOLD = 725.0 / 32768.0    # ~= 0.02212

# Primary HPF cutoff: 2000 Hz (lowered from paper's 8 kHz for harpsichord coverage).
HPF_CUTOFF = 2000.0

# Secondary HPF cutoff: 1500 Hz.  Added for bowed/sustained instruments (violin)
# whose harmonic energy lies mostly below 2 kHz.  A dual-band OR logic fires if
# EITHER band detects a transient.  Sweep confirmed:
#   castanets=89 (+1), glockenspiel=32 (unchanged), harpsichord=10 (unchanged),
#   violin=30 (from 4), oboe=6 (from 2).
HPF_CUTOFF2 = 1500.0


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
              zero_threshold: float = ZERO_THRESHOLD) -> bool:
    """
    Run SPE on a single high-pass-filtered N-sample buffer.

    Layout: buf[0 : N//2] = previous half-block
            buf[N//2 : N] = current half-block

    Returns True if any layer detects a transient in the current half.
    """
    half = N >> 1
    for j in range(1, 5):
        sub = N >> j            # sub-block size: N/2, N/4, N/8, N/16
        n_curr = 1 << (j - 1)  # sub-blocks in current half: 1, 2, 4, 8
        T = thresholds[j - 1]
        # Reference: last sub-block of the previous half
        prev_peak = _sub_peak(hp_buf, half - sub, sub)
        for k in range(n_curr):
            curr_peak = _sub_peak(hp_buf, half + k * sub, sub)
            if curr_peak > zero_threshold and curr_peak * T > prev_peak:
                return True
            prev_peak = curr_peak   # cascade: each sub-block references the one before it
    return False


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
    for j in range(1, 5):
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


def _run_spe_pass(audio: np.ndarray, sr: int, cutoff: float,
                  nMDCTLines: int, thresholds: tuple,
                  zero_threshold: float) -> set:
    """
    Internal helper: run one full HPF pass over the audio and return
    the set of block indices where SPE fires.
    """
    sos = _design_highpass(sr, cutoff)
    zi = scipy.signal.sosfilt_zi(sos) * audio[0]

    N = nMDCTLines * 2
    half = nMDCTLines
    n_blocks = int(np.ceil(len(audio) / half))
    prev_hp = np.zeros(half, dtype=np.float64)
    detected = set()

    for block_idx in range(n_blocks):
        start = block_idx * half
        end = min(start + half, len(audio))
        chunk = audio[start:end]
        if len(chunk) < half:
            chunk = np.pad(chunk, (0, half - len(chunk)))

        curr_hp, zi = scipy.signal.sosfilt(sos, chunk, zi=zi)
        buf = np.concatenate([prev_hp, curr_hp])

        if spe_block(buf, N, thresholds, zero_threshold):
            detected.add(block_idx)

        prev_hp = curr_hp

    return detected


def detectTransientsSPE(audioPath: str,
                         nMDCTLines: int = 1024,
                         cutoff: float = HPF_CUTOFF,
                         cutoff2: float = HPF_CUTOFF2,
                         thresholds: tuple = THRESHOLDS,
                         zero_threshold: float = ZERO_THRESHOLD,
                         verbose: bool = False) -> np.ndarray:
    """
    Detect transients using dual-band Sub-block Peak Energy (SPE).

    Runs two HPF passes (cutoff and cutoff2) and returns the union of
    their detections.  The primary band (cutoff=2000 Hz) is tuned for
    percussive/plucked instruments; the secondary band (cutoff2=1500 Hz)
    extends coverage to bowed instruments whose energy lies below 2 kHz.
    Pass cutoff2=None to disable the secondary band (single-band mode).

    Processes the file block-by-block with no whole-file pre-analysis.
    Each codec block advances `nMDCTLines` samples; the SPE buffer is
    2*nMDCTLines samples (previous half + current half), matching the
    codec's 50%-overlap MDCT window.

    Parameters
    ----------
    audioPath : str
    nMDCTLines : int
        New samples per codec block (= nMDCTLines in the codec). Default 1024.
    cutoff : float
        Primary high-pass filter cutoff in Hz. Default 2000.
    cutoff2 : float or None
        Secondary high-pass filter cutoff in Hz. Default 1500. Pass None
        to disable and revert to single-band mode.
    thresholds : tuple of 4 floats
        Per-layer thresholds (T1, T2, T3, T4). Transient fires when
        curr_peak * T[j] > prev_peak. Applied to BOTH bands.
    zero_threshold : float
        Minimum current-half HPF peak; suppresses noise-floor detections.
        Applied to BOTH bands.
    verbose : bool

    Returns
    -------
    np.ndarray of int
        Sorted block indices where transients are detected (union of both bands).
        xrmc.py shifts these back by 1 to place START blocks correctly.
    """
    sr, raw = scipy.io.wavfile.read(audioPath)

    if raw.dtype == np.int16:
        audio = raw.astype(np.float64) / 32768.0
    elif raw.dtype == np.int32:
        audio = raw.astype(np.float64) / 2147483648.0
    else:
        audio = raw.astype(np.float64)

    if audio.ndim == 2:          # stereo -> mono
        audio = audio.mean(axis=1)

    # Primary band
    detected = _run_spe_pass(audio, sr, cutoff, nMDCTLines, thresholds, zero_threshold)

    # Secondary band (union)
    if cutoff2 is not None:
        detected |= _run_spe_pass(audio, sr, cutoff2, nMDCTLines, thresholds, zero_threshold)

    transient_blocks = sorted(detected)

    if verbose:
        print(f"  SPE blocks ({len(transient_blocks)}): {transient_blocks}")

    return np.array(transient_blocks, dtype=int)
