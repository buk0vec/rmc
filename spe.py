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


class RingBuffer:
    """
    Single-handle ring buffer for encoder data + streaming SPE transient detection.

    Maintains a 4096-sample ring fed from inFile. The encoder reads via step(n),
    which also advances the tri-band HPF by n samples each block. spe_peek()
    runs SPE on the current lookahead window without consuming it.

    Pointer invariant (absolute sample indices):
        encoder_pos  ≤  filter_pos  =  encoder_pos + halfN
        filter_pos   ≤  write_pos   =  encoder_pos + 2*halfN
    """

    _C = 4096  # ring capacity; power of 2 so masking works

    def __init__(self, inFile, codingParams, halfN):
        self._halfN = halfN
        self._nCh = codingParams.nChannels
        self._inFile = inFile
        self._cp = codingParams
        self._total = codingParams.numSamples
        self._buf = np.zeros((self._nCh, self._C), dtype=np.float64)

        self._encoder_pos = 0
        self._filter_pos = 0
        self._write_pos = 0
        self._eof = False

        self._filters = [_design_highpass(codingParams.sampleRate, cf)
                         for cf in (HPF_CUTOFF, HPF_CUTOFF2, HPF_CUTOFF3)]
        self._filter_states = None
        self._rolling_prev = np.zeros((len(self._filters), halfN))

        self._prefill()

    def _write_raw(self, data):
        n = len(data[0])
        i = self._write_pos & (self._C - 1)
        tail = self._C - i
        if n <= tail:
            for ch, arr in enumerate(data):
                self._buf[ch, i:i + n] = arr
        else:
            for ch, arr in enumerate(data):
                self._buf[ch, i:] = arr[:tail]
                self._buf[ch, :n - tail] = arr[tail:]
        self._write_pos += n

    def _read_ring(self, start, n):
        i = start & (self._C - 1)
        if i + n <= self._C:
            return self._buf[:, i:i + n].copy()
        return np.concatenate([self._buf[:, i:], self._buf[:, :n - (self._C - i)]], axis=1)

    def _file_read(self, n):
        """Read n samples from file into ring; writes zeros when past EOF."""
        if self._eof:
            self._write_raw([np.zeros(n) for _ in range(self._nCh)])
            return
        saved = self._cp.nSamplesPerBlock
        self._cp.nSamplesPerBlock = n
        data = self._inFile.ReadDataBlock(self._cp)
        self._cp.nSamplesPerBlock = saved
        if data is None:
            self._eof = True
            self._write_raw([np.zeros(n) for _ in range(self._nCh)])
            return
        self._write_raw(data)

    def _mono(self, arr):
        return arr.mean(axis=0) if self._nCh > 1 else arr[0]

    def _run_hpf(self, mono, update=True):
        """Run all HPF bands on mono signal. update=False for non-destructive peek."""
        out = np.zeros((len(self._filters), len(mono)))
        for i, (filt, state) in enumerate(zip(self._filters, self._filter_states)):
            y, new_state = scipy.signal.sosfilt(filt, mono, zi=state)
            out[i] = y
            if update:
                self._filter_states[i] = new_state
        return out

    def _prefill(self):
        """Pre-fill ring with 2*halfN samples; filter first halfN to prime rolling_prev."""
        halfN = self._halfN
        self._file_read(2 * halfN)

        first_mono = self._mono(self._read_ring(0, halfN))
        self._filter_states = [
            scipy.signal.sosfilt_zi(f) * first_mono[0]
            for f in self._filters
        ]

        hp = self._run_hpf(first_mono, update=True)
        self._rolling_prev = hp
        self._filter_pos = halfN

    def step(self, n):
        """
        Advance ring by n samples. Returns block data as list of channel arrays, or None at EOF.
        Replaces inFile.ReadDataBlock in the encode loop.
        """
        if self._encoder_pos >= self._total:
            return None

        self._file_read(n)

        mono = self._mono(self._read_ring(self._filter_pos, n))
        hp = self._run_hpf(mono, update=True)

        halfN = self._halfN
        if n >= halfN:
            self._rolling_prev = hp[:, -halfN:]
        else:
            self._rolling_prev = np.roll(self._rolling_prev, -n, axis=1)
            self._rolling_prev[:, -n:] = hp

        data_arr = self._read_ring(self._encoder_pos, n)

        self._encoder_pos += n
        self._filter_pos += n

        return [data_arr[ch] for ch in range(self._nCh)]

    def spe_peek(self):
        """
        Run tri-band SPE on the current lookahead window without advancing state.
        Returns absolute hit sample position (encoder_pos + halfN + offset), or None.
        """
        if self._filter_states is None:
            return None
        halfN = self._halfN
        peek_mono = self._mono(self._read_ring(self._filter_pos, halfN))
        hp_curr = self._run_hpf(peek_mono, update=False)

        N = 2 * halfN
        earliest = None
        for i in range(len(self._filters)):
            buf = np.concatenate([self._rolling_prev[i], hp_curr[i]])
            fired, offset = spe_block(buf, N)
            if fired:
                earliest = offset if earliest is None else min(earliest, offset)

        return (self._encoder_pos + halfN + earliest) if earliest is not None else None
