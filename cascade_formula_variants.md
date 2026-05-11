# Cascade Formula Variants

Two designs for AC-2A block switching. Toggle between them by applying the diffs below.

---

## Version A — Old formula, no gate (CURRENT)
Produces `Van_124_bs_adaptive_2.wav`. Transients mostly land in **STOP** (35/40 on Van_124).  
k ∈ [0,7], b_start ∈ [128, 896]. MEDIUM only when k ≥ 1.

## Version B — Second-block formula
Guarantees transients land in **SHORT** (39/40 on Van_124, 1 edge case at start of file).  
k ∈ [0,7], b_start ∈ [896, 1792]. MEDIUM always present.

---

## Switching A → B (old formula → second-block formula)

### `xrmc.py`
```python
# A (old):
min_spacing = 2 * nMDCTLines

# B (new):
min_spacing = 17 * (nMDCTLines // 8)  # 2176: ensures next transient d >= 1024
```

### `blockswitching.py` — `plan_cascade`
```python
# A (old):
def plan_cascade(k_attack):
    if k_attack <= 0:
        return []
    return [(k_attack * 128, 128)]

# B (new):
def plan_cascade(k_attack):
    # Transient always in second lookahead block (d ∈ [1024,2047]), b_start always > 128
    return [((7 + k_attack) * 128, 128)]
```

### `rmcfile.py` — decoder b_start recovery (~line 138)
```python
# A (old):
b_start = max(halfN_short, k_attack_read * halfN_short)

# B (new):
b_start = (7 + k_attack_read) * halfN_short
```

### `rmcfile.py` — WriteDataBlock encoder k formula (~line 482)
```python
# A (old):
_k_encoded = max(0, min(7, _raw_offset // halfN_short - 1))
_b_start = max(halfN_short, _k_encoded * halfN_short)

# B (new):
_k_encoded = max(0, min(7, (_raw_offset - halfN) // halfN_short))
_b_start = (7 + _k_encoded) * halfN_short
```

### `rmcfile.py` — lookahead k formula (~line 1064)
```python
# A (old):
_k_next = max(0, min(7, _raw_next // halfN_short - 1))
_b_next = max(halfN_short, _k_next * halfN_short)

# B (new):
_k_next = max(0, min(7, (_raw_next - halfN) // halfN_short))
_b_next = (7 + _k_next) * halfN_short
```

### `rmcfile.py` — WriteFileHeader initial nSamplesPerBlock (~line 413)
```python
# A (old):
_k0 = max(0, min(7, _raw0 // codingParams.nMDCTLines_short - 1))
_bs0 = max(codingParams.nMDCTLines_short, _k0 * codingParams.nMDCTLines_short)

# B (new):
_k0 = max(0, min(7, (_raw0 - codingParams.nMDCTLines) // codingParams.nMDCTLines_short))
_bs0 = (7 + _k0) * codingParams.nMDCTLines_short
```

### `rmcfile.py` — priorBlock update in WriteDataBlock (~line 543)
```python
# A (old) — always keep halfN samples:
new_prior.append(all_samples[-halfN:])

# B (new) — keep b_start samples after START so MEDIUM left overlap is covered:
_keep = (codingParams.cascade_b
         if AC2A_BLOCK_SWITCHING and codingParams.blockType == START
            and codingParams.cascade_b > halfN
         else halfN)
new_prior.append(all_samples[-_keep:])
```

---

## Key tradeoff

| | Version A | Version B |
|---|---|---|
| Transient placement | Mostly STOP | Always SHORT (except first block) |
| b_start range | 128–896 | 896–1792 |
| MEDIUM always present | No (k=0 → none) | Yes |
| min_spacing | 2048 samples | 2176 samples |
| Perceptual result | Sounds smoother on Van_124 | Transients more temporally precise |

Version A sounded better on Van_124 at 96 kbps as of 2026-04-26. Root cause of B's artifact not yet diagnosed — psych model and TDAC are both correct; likely something about large asymmetric MEDIUM windows.

---

## Analysis: why Version A sounds better (2026-04-26)

### k is always 7 in Version A

Version A formula: `k = max(0, min(7, d // 128 - 1))`

All second-block transients have d ∈ [1024, 2047]. For d ≥ 1024: `d // 128 ≥ 8`, so `8 - 1 = 7`. k is always clamped to 7 — the 3-bit bitstream field carries no useful information on this material. `b_start = max(128, 7*128) = 896` every single time. The MEDIUM block is always exactly 1024 samples (b_start=896, SHORT=128, 896+128=1024). Version A's cascade is effectively a constant.

### Version B MEDIUM blocks are much larger

With `b_start = (7 + k) * 128`:
- k=0 → b_start=896, MEDIUM=1024 samples (same as Version A)
- k=7 → b_start=1792, MEDIUM=1920 samples

Large, variable MEDIUM windows likely cause the audible artifact. TDAC is verified correct at all boundaries; psych model adapts via DesignSFBands. Most likely culprit is the large asymmetric MEDIUM window shape itself — it covers almost two full LONG blocks of time, and the 576-line MDCT at this huge asymmetry may interact badly with quantization noise at low-to-medium bitrate.

### Transient detector latency makes Version B fragile

`TD.py` uses a 0.5ms attack envelope follower (22 samples at 44100 Hz). `extractTransient` gates the signal then finds `hit[0]` — the first raw audio sample above 0.001 in the gated window. This introduces a systematic latency of roughly **10–50 samples** relative to the true acoustic onset (the actual first non-zero sample of the transient).

Version B places SHORT at the *detected* position. If the detector is late by 30 samples and b_start was set for the detected position, the actual transient onset falls in the MEDIUM block, not SHORT. Version A is immune: it always uses the same fixed cascade (b_start=896) regardless of how accurate the detection is.

**Version B is precise about an imprecise measurement. Version A is robust because it ignores the measurement.**

### What was needed for Version B to work well: local-minimum onset detection (SOLVED 2026-04-26)

**Root cause confirmed**: the envelope-difference gate opens when `fast_env ≈ 10% of peak`, which coincides with the 10% relative walk-back threshold — both land at the same point. Fixed offset (-20 samples) also insufficient and made some events worse by overshooting. The real solution: walk backward from the local envelope peak to find the **local minimum** of the fast envelope in the 200-sample window before the peak. That valley is the bottom of the pre-attack dip — where the transient actually lifts off from background.

**Implementation in `simple_run.py`** (in the event-range loop):
```python
# Find local peak of fast envelope, then find the local minimum
# between the background and that peak.
LOOKAHEAD = 100
LOOKBACK = 200
peak_start = max(0, detected - 50)
peak_end = min(len(fast_env_mono), detected + LOOKAHEAD)
peak_abs = peak_start + int(np.argmax(fast_env_mono[peak_start:peak_end]))
min_start = max(0, peak_abs - LOOKBACK)
min_rel = int(np.argmin(fast_env_mono[min_start:peak_abs]))
sample_index = min_start + min_rel
```

`fast_env_mono = np.maximum(fast_envelope[0], fast_envelope[1])` must be computed before the loop.

**Result**: positions move 37–193 samples earlier than gate opening, Version B sounds much better (confirmed on Van_124 at 96 kbps, 2026-04-26). 38/39 transients land in SHORT (1 edge case at start of file; also lost 1 close pair at 9.78s due to min_spacing filter).

### Gate-filter fix for adaptive MAX_SHIFT (SOLVED 2026-04-26)

Adaptive MAX_SHIFT (100 when gap>8192, else 50) improved sharpness vs fixed 50 but introduced a phantom event at 3.95s. Root cause: shifting an event's `sample_index` back by 100 samples changed its block assignment by 1, widening the gap to the next event just enough to let a previously-filtered event through the spacing filter.

**Fix**: spacing filter (`enforceMinEventSpacing`) uses `gate_block = detected // blockSize` (pre-shift) rather than `sample_index // blockSize` (post-shift). This decouples the local-min walk-back from the filter decision — walk-back can freely move `sample_index` without changing which events survive. The `gate_block` field is stored on each event dict; `enforceMinEventSpacing` falls back to `block` when absent.

**Result**: 40 events (vs 39 with fixed-shift filtering; the 9.78s close pair both survive since their gate positions are genuinely 4 blocks apart). No 4s artifact. Adaptive MAX_SHIFT gives full sharpness benefit.

**Encodes for reference**:
- `Van_124_bs_B_gatefilter_96k.wav` — Version B + local-min + adaptive MAX_SHIFT + gate-filter (current best)
- `Van_124_bs_B_localmin.wav` — Version B + local-min detection, fixed MAX_SHIFT=100, shift-based filter
- `Van_124_bs_B_compensated.wav` — Version B + fixed 20-sample offset (mixed results)
- `Van_124_bs_adaptive_2.wav` — Version A (fixed cascade, transients mostly in STOP)
