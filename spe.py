"""
Sub-block Peak Energy (SPE) transient detection.

Implements the SPE method from:
  Fan et al., "Transient Detection Methods for Audio Coding,"
  AES 155th Convention, New York, 2023.

Algorithm summary (Section 3.4 + Layer 5 extension):
  - Buffer: N samples (left N/2 = previous block, right N/2 = current block)
  - Tri-band high-pass filter (see HPF_CUTOFF / HPF_CUTOFF2 / HPF_CUTOFF3)
  - Five layers of sub-block sizes: N/2, N/4, N/8, N/16, N/32
  - For each layer j, compare peak of each right-half sub-block to the
    immediately preceding sub-block (cascading within the right half).
  - Flag condition: curr_peak * T[j] > prev_peak  AND  curr_peak > zero_threshold
  - A transient is declared if ANY layer of ANY band fires.

Threshold selection (empirically validated on castanets, glockenspiel, harpsichord):
  - L1 (N/2  = 1024 samples): T=0.40  → ratio must exceed 2.5×
  - L2 (N/4  =  512 samples): T=0.40  → ratio must exceed 2.5×
  - L3 (N/8  =  256 samples): T=0.07  → ratio must exceed 14.3×
  - L4 (N/16 =  128 samples): T=0.07  → ratio must exceed 14.3×
  - L5 (N/32 =   64 samples): T=0.05  → ratio must exceed 20×
  Layer 1 dominates for these signals (all transients visible at coarse scale).
  Layers 2-5 add coverage for brief, localised spikes that Layer 1 misses.

Tri-band HPF design:
  Band 1 (HPF_CUTOFF  = 2500 Hz): primary — percussive/plucked instruments
    (castanets, harpsichord). Filters sustain that would elevate prev_peak.
  Band 2 (HPF_CUTOFF2 = 1500 Hz): secondary — bowed/sustained instruments
    (violin) whose harmonic energy lies mostly below 2 kHz.
  Band 3 (HPF_CUTOFF3 = 5000 Hz): tertiary — metallic resonators (glockenspiel).
    Attack energy above 5 kHz decays much faster than lower harmonics, so
    the attack/decay ratio is clean here even within ringing clusters.
    Without this band, glockenspiel notes inside a decay tail are masked
    because prev_peak stays elevated in the lower bands.

Validated detection counts (tri-band SPE, EBU-SQAM):
  castanets=86  glockenspiel=44  harpsichord=10  Van_124=91
"""

import numpy as np
import scipy.io.wavfile
import scipy.signal

from pcmfile import PCMFile

# Per-layer thresholds T[j].  A transient fires when: curr_peak * T[j] > prev_peak
THRESHOLDS = (0.40, 0.40, 0.07, 0.07, 0.05)

ZERO_THRESHOLD = 750.0 / 32768.0    # ≈ 0.0229

HPF_CUTOFF  = 2500.0   # Hz — primary band
HPF_CUTOFF2 = 1500.0   # Hz — secondary band (violin / bowed instruments)
HPF_CUTOFF3 = 5000.0   # Hz — tertiary band (glockenspiel / metallic resonators)


def _design_highpass(sr: int, cutoff: float = HPF_CUTOFF, order: int = 4):
    """4th-order Butterworth high-pass SOS filter."""
    nyq = sr / 2.0
    wn = min(cutoff / nyq, 0.9999)
    return scipy.signal.butter(order, wn, btype='high', output='sos')


def _sub_peak(buf: np.ndarray, start: int, length: int) -> float:
    """Peak absolute amplitude in buf[start : start+length]."""
    return float(np.max(np.abs(buf[start: start + length])))


def _refine_offset(hp_buf: np.ndarray, start: int, sub: int, target: int = 64,
                   zero_threshold: float = ZERO_THRESHOLD) -> int:
    """
    Binary bisection within a detected sub-block down to `target`-sample resolution.
    Picks the higher-peak half at each step, then backs up one slot if the preceding
    slot carries >= 25% of the detection slot's energy (onset is there, not at peak).
    Returns absolute start index of the chosen target-size sub-block.
    No-op if sub == target already.
    """
    half = len(hp_buf) // 2
    while sub > target:
        sub //= 2
        e_lo = _sub_peak(hp_buf, start, sub)
        e_hi = _sub_peak(hp_buf, start + sub, sub)
        if e_hi > e_lo:
            start += sub
    if start - target >= half:
        e_prev = _sub_peak(hp_buf, start - target, target)
        e_curr = _sub_peak(hp_buf, start, target)
        if e_prev >= 0.25 * e_curr:
            start -= target
    return start


def spe_block(hp_buf: np.ndarray,
              N: int,
              thresholds: tuple = THRESHOLDS,
              zero_threshold: float = ZERO_THRESHOLD):
    """
    Run SPE on a single high-pass-filtered N-sample buffer.

    Layout: buf[0 : N//2] = previous half-block
            buf[N//2 : N] = current half-block

    Runs L1–L5 coarse-to-fine. Each firing layer overwrites best_offset.
    After the loop, if the finest firing layer is coarser than 64 samples,
    _refine_offset bisects it down to 64-sample resolution.

    Returns
    -------
    fired : bool
    sample_offset : int
        Onset offset within the current half at 64-sample resolution. 0 if not fired.
    """
    half = N >> 1
    fired = False
    best_offset = 0
    best_sub = 64

    for j in range(1, 6):
        sub = N >> j
        n_curr = 1 << (j - 1)
        T = thresholds[j - 1]
        prev_peak = _sub_peak(hp_buf, half - sub, sub)
        for k in range(n_curr):
            curr_peak = _sub_peak(hp_buf, half + k * sub, sub)
            if curr_peak > zero_threshold and curr_peak * T > prev_peak:
                best_offset = k * sub
                best_sub = sub
                fired = True
                break
            prev_peak = curr_peak

    if fired and best_sub > 64:
        best_offset = _refine_offset(hp_buf, half + best_offset, best_sub,
                                     zero_threshold=zero_threshold) - half

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


def _run_spe_pass(audio: np.ndarray, sr: int, cutoff: float,
                  nMDCTLines: int, thresholds: tuple,
                  zero_threshold: float) -> dict:
    """
    Run one full HPF pass over the audio.
    Returns {block_idx: sample_offset} for every block where SPE fires.
    sample_offset is at 64-sample resolution within the current half-block.
    """
    sos = _design_highpass(sr, cutoff)
    zi = scipy.signal.sosfilt_zi(sos) * audio[0]

    N = nMDCTLines * 2
    half = nMDCTLines
    n_blocks = int(np.ceil(len(audio) / half))
    prev_hp = np.zeros(half, dtype=np.float64)
    detected = {}

    for block_idx in range(n_blocks):
        start = block_idx * half
        end = min(start + half, len(audio))
        chunk = audio[start:end]
        if len(chunk) < half:
            chunk = np.pad(chunk, (0, half - len(chunk)))

        curr_hp, zi = scipy.signal.sosfilt(sos, chunk, zi=zi)
        buf = np.concatenate([prev_hp, curr_hp])

        fired, offset = spe_block(buf, N, thresholds, zero_threshold)
        if fired:
            detected[block_idx] = offset

        prev_hp = curr_hp

    return detected


def detectTransientsSPE(audioPath: str,
                        nMDCTLines: int = 1024,
                        cutoff: float = HPF_CUTOFF,
                        cutoff2: float = HPF_CUTOFF2,
                        cutoff3: float = HPF_CUTOFF3,
                        thresholds: tuple = THRESHOLDS,
                        zero_threshold: float = ZERO_THRESHOLD,
                        verbose: bool = False) -> np.ndarray:
    """
    Detect transients using tri-band SPE. Returns sorted block indices.
    Union of all three band detections, with adjacent-block dedup.
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

    detected = set()
    for cf in (cutoff, cutoff2, cutoff3):
        if cf is not None:
            detected |= set(_run_spe_pass(audio, sr, cf, nMDCTLines,
                                          thresholds, zero_threshold))
    combined = sorted(detected)

    # Adjacent-block dedup: straddled-boundary artifacts fire on consecutive
    # blocks for the same attack. Drop the later of any (N, N+1) pair.
    transient_blocks = []
    skip_next = False
    for i, b in enumerate(combined):
        if skip_next:
            skip_next = False
            continue
        transient_blocks.append(b)
        if i + 1 < len(combined) and combined[i + 1] == b + 1:
            skip_next = True

    if verbose:
        print(f"  SPE blocks ({len(transient_blocks)}): {transient_blocks}")
    return np.array(transient_blocks, dtype=int)


def detectTransientsSPESamples(audioPath: str,
                                nMDCTLines: int = 1024,
                                cutoff: float = HPF_CUTOFF,
                                cutoff2: float = HPF_CUTOFF2,
                                cutoff3: float = HPF_CUTOFF3,
                                thresholds: tuple = THRESHOLDS,
                                zero_threshold: float = ZERO_THRESHOLD,
                                verbose: bool = False) -> list:
    """
    Detect transients using tri-band SPE. Returns event dicts with exact sample positions.

    Runs three independent HPF passes and unions their detections.
    For blocks detected by multiple bands, uses the earliest (minimum) offset
    so the transient marker lands as close to the true onset as possible.
    Adjacent-block duplicates from boundary-straddling are suppressed.

    Returns
    -------
    list of {"sample_index": int, "block": int}
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

    half = nMDCTLines

    # Run all enabled bands; union block detections, keep earliest offset per block
    block_offsets = {}
    for cf in (cutoff, cutoff2, cutoff3):
        if cf is None:
            continue
        for block_idx, offset in _run_spe_pass(audio, sr, cf, nMDCTLines,
                                               thresholds, zero_threshold).items():
            if block_idx not in block_offsets:
                block_offsets[block_idx] = offset
            else:
                block_offsets[block_idx] = min(block_offsets[block_idx], offset)

    combined = sorted(block_offsets)

    # Adjacent-block dedup (same logic as detectTransientsSPE)
    deduped = []
    skip_next = False
    for i, b in enumerate(combined):
        if skip_next:
            skip_next = False
            continue
        deduped.append(b)
        if i + 1 < len(combined) and combined[i + 1] == b + 1:
            skip_next = True

    events = [{"sample_index": b * half + block_offsets[b], "block": b}
              for b in deduped]

    if verbose:
        print(f"  SPE events ({len(events)}): {[e['sample_index'] for e in events]}")
    return events


class TransientDetector:
    """
    Streaming SPE transient detector with its own PCM read pointer.

    Maintains filter state and reads ahead of the encoder by one block.
    Call poll(encoder_pos) once per free-running LONG/STOP block to get the
    absolute sample position of the next transient, or None if none found.
    """

    def __init__(self, inFilename, nMDCTLines, sampleRate):
        self._halfN = nMDCTLines
        self._read_pos = 0

        self._filters = [_design_highpass(sampleRate, cf)
                         for cf in (HPF_CUTOFF, HPF_CUTOFF2, HPF_CUTOFF3)
                         if cf is not None]

        self._pcm = PCMFile(inFilename)
        self._cp = self._pcm.OpenForReading()
        self._cp.nSamplesPerBlock = nMDCTLines

        first_chunk = self._read_chunk()
        if first_chunk is not None:
            self._filter_states = [scipy.signal.sosfilt_zi(f) * first_chunk[0]
                                   for f in self._filters]
            self._prev_filtered = [np.zeros(nMDCTLines) for _ in self._filters]
            self._advance_filters(first_chunk)
        else:
            self._filter_states = [scipy.signal.sosfilt_zi(f) for f in self._filters]
            self._prev_filtered = [np.zeros(nMDCTLines) for _ in self._filters]

    def _read_chunk(self):
        data = self._pcm.ReadDataBlock(self._cp)
        if not data:
            return None
        chunk = data[0] if len(data) == 1 else np.mean(data, axis=0)
        return chunk.astype(np.float64)

    def _read_n(self, n):
        """Read exactly n samples, mono-mixed. Returns None at EOF."""
        self._cp.nSamplesPerBlock = n
        data = self._pcm.ReadDataBlock(self._cp)
        self._cp.nSamplesPerBlock = self._halfN
        if not data:
            return None
        chunk = data[0] if len(data) == 1 else np.mean(data, axis=0)
        return chunk.astype(np.float64)

    def _advance_filters(self, chunk):
        for i, (filt, state) in enumerate(zip(self._filters, self._filter_states)):
            curr, new_state = scipy.signal.sosfilt(filt, chunk, zi=state)
            self._filter_states[i] = new_state
            self._prev_filtered[i] = curr
        self._read_pos += self._halfN

    def poll(self, encoder_pos):
        """
        Call once per free-running LONG/STOP block.

        Checks territory [encoder_pos+halfN, encoder_pos+2*halfN) for a transient.
        Uses variable-size catch-up to handle non-halfN-aligned encoder positions
        that occur after cascades (whose total size is not a multiple of halfN).
        Returns absolute hit sample position, or None.
        """
        halfN = self._halfN

        # If behind encoder_pos, catch up with one variable-size read (filter state only,
        # no _prev_filtered update — the prev half will be read explicitly next).
        if self._read_pos < encoder_pos:
            gap = encoder_pos - self._read_pos
            chunk = self._read_n(gap)
            if chunk is None:
                return None
            for i, (filt, state) in enumerate(zip(self._filters, self._filter_states)):
                _, new_state = scipy.signal.sosfilt(filt, chunk, zi=state)
                self._filter_states[i] = new_state
            self._read_pos = encoder_pos

        # If now exactly at encoder_pos (fresh arrival after catch-up, or start of file),
        # read [encoder_pos, encoder_pos+halfN) explicitly as the prev half.
        if self._read_pos == encoder_pos:
            prev_chunk = self._read_n(halfN)
            if prev_chunk is None:
                return None
            for i, (filt, state) in enumerate(zip(self._filters, self._filter_states)):
                curr, new_state = scipy.signal.sosfilt(filt, prev_chunk, zi=state)
                self._filter_states[i] = new_state
                self._prev_filtered[i] = curr
            self._read_pos = encoder_pos + halfN

        # read_pos == encoder_pos + halfN; _prev_filtered covers [encoder_pos, encoder_pos+halfN).
        # Read curr half [encoder_pos+halfN, encoder_pos+2*halfN) and run SPE.
        curr_chunk = self._read_n(halfN)
        if curr_chunk is None:
            return None

        earliest = None
        N = 2 * halfN
        for i, (filt, state) in enumerate(zip(self._filters, self._filter_states)):
            curr, new_state = scipy.signal.sosfilt(filt, curr_chunk, zi=state)
            buf = np.concatenate([self._prev_filtered[i], curr])
            fired, offset = spe_block(buf, N, THRESHOLDS, ZERO_THRESHOLD)
            if fired:
                earliest = offset if earliest is None else min(earliest, offset)
            self._filter_states[i] = new_state
            self._prev_filtered[i] = curr
        self._read_pos += halfN

        return (encoder_pos + halfN + earliest) if earliest is not None else None

    def close(self):
        self._pcm.Close(self._cp)
