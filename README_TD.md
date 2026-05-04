# Transient Detection (TD) — SPE Module

## Overview

This documents the transient detection system built for the RMC codec, implemented across
two files:

- **`spe.py`** — core algorithm: tri-band Sub-block Peak Energy (SPE) detector
- **`td_spe.py`** — wrapper: bridges `spe.py` into the block switching pipeline

The detector replaces `simple_run.detectTransients` in `xrmc.py` with a single import swap.
It is a **drop-in replacement** — same function signature, same return format, different algorithm.

---

## Background: Why SPE?

The SPE method comes from:
> Fan et al., *"Transient Detection Methods for Audio Coding,"*
> AES 155th Convention, New York, 2023. Section 3.4.

SPE is designed specifically for audio codec block switching, not general-purpose onset
detection. Its key properties:
- **Block-by-block processing** — no whole-file look-ahead; compatible with real-time pipelines
- **Ratio-based** — fires on *relative* energy jumps (attack vs. preceding block), not absolute levels
- **High-pass filtered** — ignores low-frequency sustain/decay that would suppress attack ratios
- **Sub-block resolution** — 4 layers of sub-blocks (N/2 → N/16) catch attacks at different speeds

---

## Algorithm: `spe.py`

### Core Idea

For each codec block, SPE maintains a buffer of `2 × nMDCTLines` samples:

```
[ ← previous half (nMDCTLines samples) → | ← current half (nMDCTLines samples) → ]
```

This matches the codec's 50%-overlap MDCT window. The buffer is split into four layers of
sub-blocks:

| Layer | Sub-block size | Sub-blocks in current half | Threshold T |
|-------|---------------|---------------------------|-------------|
| L1    | N/2 = 1024    | 1                         | 0.45        |
| L2    | N/4 = 512     | 2                         | 0.45        |
| L3    | N/8 = 256     | 4                         | 0.07        |
| L4    | N/16 = 128    | 8                         | 0.07        |

A **transient fires** in a layer if, for any sub-block in the current half:

```
curr_peak × T  >  prev_peak    AND    curr_peak > ZERO_THRESHOLD
```

where `prev_peak` is the peak of the immediately preceding sub-block (cascading within the
buffer), and `curr_peak` is the peak of the current sub-block.

Equivalently: the ratio `curr/prev` must exceed `1/T`:
- L1/L2: ratio > **2.22×** (large block-level jump)
- L3/L4: ratio > **14.3×** (sharp sub-block spike, e.g. single drum hit)

A block is declared a transient if **any layer** fires.

### Tri-Band Architecture

A single HPF cutoff works well for percussive instruments but fails for others:
- **Violin** (bowed): harmonic energy mostly below 2 kHz — a 2 kHz HPF cuts too much of the attack
- **Glockenspiel**: ringing decay keeps `prev_peak` elevated in lower bands, suppressing the ratio
  for subsequent notes within the same decay cluster

Solution: run **three independent HPF passes** and take the **union** of detections.

| Band | HPF Cutoff | Primary Use |
|------|-----------|-------------|
| Primary   | 2500 Hz | Percussive / plucked (castanets, harpsichord) |
| Secondary | 1500 Hz | Bowed / sustained (violin) — attack energy below 2 kHz passes through |
| Tertiary  | 5000 Hz | Metallic resonators (glockenspiel) — HF attack decays faster than ringing |

Each band uses a **4th-order Butterworth HPF** (`scipy.signal.butter`), applied causally
with `sosfilt` and warm-started initial conditions so block boundaries are continuous.

### Boundary Deduplication

When an attack straddles a block boundary, the filter ringing can cause two consecutive blocks
to both fire (e.g., block N and N+1 for the same physical event). Post-filter rule:

> **In any consecutive pair (N, N+1), always keep N and drop N+1.**

Block N is the codec-relevant onset; N+1 is a boundary artifact. This is safe because the
tightest legitimate note spacing in the test set is ~2 blocks (~46 ms). A 1-block gap (~23 ms)
is below the perceptual fusion threshold and is virtually always a straddle artifact.

### Key Parameters

```python
THRESHOLDS     = (0.45, 0.45, 0.07, 0.07)  # per-layer T values (L1..L4)
ZERO_THRESHOLD = 725.0 / 32768.0            # ~0.022 — amplitude gate on HPF signal
HPF_CUTOFF     = 2500.0   # Hz — primary band
HPF_CUTOFF2    = 1500.0   # Hz — secondary band (violin)
HPF_CUTOFF3    = 5000.0   # Hz — tertiary band  (glockenspiel)
```

**What each controls:**

| Parameter | Effect when increased | Effect when decreased |
|-----------|----------------------|----------------------|
| `THRESHOLDS[0,1]` (T12) | Fewer detections (higher ratio required) | More detections — may add false positives |
| `THRESHOLDS[2,3]` (T34) | Fewer short-spike detections | More spike detections |
| `ZERO_THRESHOLD` | Fewer detections (stricter amplitude gate) | More detections — noise-floor false positives |
| `HPF_CUTOFF` | **Counterintuitively: more false positives** at very high values — see note below |
| `HPF_CUTOFF2` | Fewer violin detections (more energy cut) | More violin detections |
| `HPF_CUTOFF3` | Fewer glockenspiel within-cluster detections | More glockenspiel detections |

> **Why does raising the HPF cutoff cause more false positives?**
> A higher cutoff passes less energy overall. With less energy through the filter,
> more blocks fall below `ZERO_THRESHOLD` — but when a block does pass the gate, the
> `prev_peak` may have dropped to near-zero (silence after the filter cut), making the
> ratio `curr/prev` artificially large. Any tiny transient can now produce a huge ratio
> and trigger a false detection. The amplitude gate (`ZERO_THRESHOLD`) partially protects
> against this but cannot compensate for extremely sparse filtered energy.

### Validated Detection Counts

Tested on 7 audio files at `nMDCTLines=1024` (44.1 kHz):

| File | Detections | Notes |
|------|-----------|-------|
| castanets.wav | 86 | Each castanet hit detected cleanly |
| glockenspiel.wav | 44 | Target — tertiary band recovers within-cluster notes |
| harpsichord.wav | 10 | Clean detection of plucked attacks |
| Van_124.wav | 91 | Drum track at 124 BPM, ~88 after encoder min-spacing filter |
| violin.wav | 28 | Bow attacks; soft onsets near the ZERO_THRESHOLD boundary |
| violin2.wav | 13 | Softer bow attacks — SPE limitation on very gradual onsets |
| oboe.wav | 5 | Phrase-boundary onsets only — sustained wind, see note below |

> **Oboe / sustained wind instruments:** SPE is a ratio-based detector. Between notes,
> sustained instruments maintain a continuous signal so `prev_peak` stays high and ratios
> stay near 1.0. Only onsets from near-silence (phrase starts) are detectable. This is a
> known limitation of the SPE family — per-note detection for fully sustained instruments
> requires a different algorithm (spectral flux, pitch-period onset, etc.). For audio coding
> purposes, this is acceptable: pre-echo is only a concern at silence→sound boundaries,
> which SPE does catch.

### Public API

```python
from spe import detectTransientsSPE

blocks = detectTransientsSPE(
    audioPath,             # path to .wav file
    nMDCTLines=1024,       # codec block size (new samples per block)
    verbose=False,
)
# Returns: np.ndarray of int — sorted block indices where transients are detected
```

---

## Wrapper: `td_spe.py`

### Why a Wrapper?

The block switching pipeline (AC-2A Version B) needs more than just block indices — it needs
the **precise sample position of each attack onset** within its block. This is used to compute
`k_attack` (which 128-sample slot the transient falls in), which determines how the MEDIUM
block is positioned in the cascade.

`td_spe.py` bridges this gap:
1. Calls `detectTransientsSPE()` to get block indices
2. **Refines each block index** to a precise sample-level onset
3. Returns **event dicts** matching the `simple_run.detectTransients` format

### Onset Refinement: Within-Block Walk-Back

For each SPE-flagged block:

```
block_start = b_idx × nMDCTLines

1. Find PEAK: argmax(|audio|) inside [block_start, block_start + nMDCTLines)
2. Find ONSET: argmin(|audio|) inside [max(block_start, peak - 200), peak)
```

Step 2 is the **walk-back**: from the energy peak, walk back up to 200 samples (~4.5 ms at
44.1 kHz) looking for the local amplitude minimum — the point where the attack begins to rise.
The walk-back window is **clamped to the SPE-flagged block** (`max(block_start, ...)`), so the
onset always stays in the block that SPE said the attack is in. This is important: if the onset
crossed into the previous block, the encoder's `k_attack` calculation would be wrong.

### Event Format

Each event dict contains:

```python
{
    "block":         int,  # block index = sample_index // nMDCTLines
    "gate_block":    int,  # SPE-flagged block (same unless walk-back crossed boundary)
    "sample_offset": int,  # onset position within block = sample_index % nMDCTLines
    "sample_index":  int,  # absolute sample position of the onset
}
```

### Public API

Drop-in replacement for `simple_run.detectTransients`. Same signature:

```python
from td_spe import detectTransients

events = detectTransients(
    audioPath,
    forceBlockSize=1024,    # use this to set block size
    return_events=True,     # True → list of dicts; False → np.array of block indices
    verbose=False,
)
```

Most other kwargs (`sr`, `duration`, `use_auto_params`, `cwt_onset_threshold`,
`time_threshold_factor`) are accepted and ignored for signature compatibility.

---

## Integration with Block Switching (xrmc.py)

### The One-Line Change

In `xrmc.py`, replace:

```python
from simple_run import detectTransients
```

with:

```python
from td_spe import detectTransients
```

Everything else in `xrmc.py` is unchanged. The encoder receives event dicts from `td_spe.py`
and uses `sample_index` to compute `k_attack`:

```python
k_attack = sample_offset // 128   # which 128-sample slot (0–7) within the MEDIUM block
```

### Min-Spacing Filter

In `xrmc.py`, the AC2A path applies an additional spacing filter **after** SPE detection:

```python
min_spacing = 17 * (nMDCTLines // 8)   # = 2176 samples at nMDCTLines=1024
```

Any transient closer than 2176 samples to the previous one is dropped. This prevents cascade
collisions (two START/STOP sequences overlapping). For Van_124: 91 SPE events → 88 after this
filter.

### Full Pipeline

```
Van_124.wav
    │
    ├─► spe.py (tri-band HPF → sub-block peak ratio → union → dedup)
    │       → 91 block indices
    │
    ├─► td_spe.py (onset refinement → event dicts)
    │       → 91 event dicts with sample_index
    │
    ├─► xrmc.py (min-spacing filter)
    │       → 88 events passed to encoder
    │
    ├─► rmcfile.py / codec.py (AC-2A Version B cascade per transient)
    │       START → MEDIUM(k) → SHORTs → STOP
    │
    └─► Van_124_spe_192k.rmc
```

---

## How to Run

### Requirements

Install dependencies into your conda environment:

```bash
pip install numpy scipy tqdm numba
```

Drop `spe.py` and `td_spe.py` into the same directory as `xrmc.py`.

### Encode

```bash
# -b N means N kb/s per channel (stereo total = 2N kb/s)
# For 128 kb/s total: -b 64
# For 192 kb/s total: -b 96
# For 256 kb/s total: -b 128

python xrmc.py -c inputs/Van_124.wav Van_124_out.rmc -b 96 -t 124 -v
#                                                        ^    ^
#                                                   96 kb/s  BPM (only relevant if
#                                                 per channel  PREDICTION=True)
```

### Decode

```bash
python xrmc.py -d Van_124_out.rmc Van_124_out.wav -v
```

### Typical Encode Times (Van_124, ~15.5 sec audio)

| Bitrate flag | Total kb/s | Encode time | Realtime factor |
|-------------|-----------|------------|----------------|
| `-b 64`     | ~128 kb/s | ~11 sec    | 1.39×          |
| `-b 96`     | ~192 kb/s | ~9 sec     | 1.72×          |

Decode is always much faster (~1 second, ~15× realtime).

---

## Files Summary

| File | Role | Status |
|------|------|--------|
| `spe.py` | Core SPE transient detector | New — must be added to branch |
| `td_spe.py` | Wrapper for block switching integration | New — must be added to branch |
| `xrmc.py` | CLI — one import line changed | Modified |
| `features.py` | Feature flags — unchanged | Existing |
| `rmcfile.py` | Block-level encode/decode — unchanged | Existing |
| `codec.py` | MDCT + bit allocation — unchanged | Existing |

> **Note on `tns.py` / `subbass.py`:** These modules are referenced by imports on this branch
> but their implementations are missing (pre-existing issue). If you get `ModuleNotFoundError`
> for either, stub files are available — they expose the required constants and function
> signatures but raise `NotImplementedError` if called. Since `TNS=False` and
> `SUBBASS_HYBRID=False` in `features.py`, the stubs are never called at runtime.
