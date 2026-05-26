# Cascade Formula Variants

Three designs for AC-2A block switching.

---

## Version A — Fixed cascade, imprecise placement
Produces `Van_124_bs_adaptive_2.wav`. Transients mostly land in **STOP** (35/40 on Van_124).  
`b_start = 896` always (k is always clamped to 7 on real material — the 3-bit field carries no useful information).  
One MEDIUM of fixed size (1024 samples), same window shape every time.  
Robust because it ignores detector precision; same structure regardless of where the transient is detected.

## Version B — Variable MEDIUM, precise placement (CURRENT)
Guarantees transients land in **SHORT** (positions 128–255 in the 256-sample window, always the right/new-samples half).  
`b_start = (7+k)*128`, range 896–1152 samples (K_ATTACK_MAX=2 caps k at 2). One MEDIUM block whose window shape changes with k (left overlap varies).  
k ∈ [0, K_ATTACK_MAX]. MEDIUM always present.  
SPE detector (80 events on Van_124 at 96 kbps) replaced the envelope-follower + local-min approach (40 events).

## Version C — Fixed MEDIUM, count adapts (NOT YET IMPLEMENTED)
Transients always land in **SHORT**.  
MEDIUM is a fixed size (e.g. 512 samples, 256/128 overlaps) — same window shape every time.  
k controls **how many MEDIUMs** appear on the lead/tail sides, not the window shape.  
Chain: `START → MEDIUM → MEDIUM → ... → SHORT(s) → MEDIUM → ... → STOP`  
Multiple SHORTs also possible — choice of MEDIUMs vs SHORTs is arithmetic: if transient is `d` samples in, use `floor(d / MEDIUM_SIZE)` MEDIUMs to get close, then SHORT for the last hop.  
Advantage over B: every MEDIUM has identical frequency/time resolution, so psych model tables can be pre-computed and quantization noise is consistent across the cascade regardless of k.

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

## Why (7+k) is correct: OAA tracing

A naive analysis of block type at `currentSamplePos` when the cascade fires gives wrong results because it ignores that `nSamplesPerBlock` for the START block is pre-set to `b_start` by the *previous* LONG block's end-of-block lookahead.

### Cascade sequence (Version B, K_ATTACK_MAX=2)

Let P1 = `currentSamplePos` after the last LONG block completes. Transient at absolute position T.

```
Block         nSamplesPerBlock  PCM consumed           MDCT window
LONG (prev)   1024              [P1-1024, P1)          left=1024, right=1024
START         b_start           [P1, P1+b_start)       left=1024, right=b_start
MEDIUM        128               [P1+b_start, P1+b_start+128)  left=b_start, right=128
SHORT         128               [P1+b_start+128, P1+b_start+256)  left=128, right=128
STOP          1024              [P1+b_start+256, P1+b_start+1280) left=128, right=1024
```

The SHORT MDCT window (N=256) layout:
- n=0..127 (left half):  PCM [P1+b_start, P1+b_start+128)  — MEDIUM's right IMDCT tail
- n=128..255 (right half): PCM [P1+b_start+128, P1+b_start+256) — SHORT's own new samples

Sine window weight: `w(n) = sin(π*(n+0.5)/256)`. Peaks at n≈128 (w≈1.0), near zero at n=0 and n=255.

### k formula derivation

From the end-of-LONG-block lookahead:
```
_raw_next  = T - P1
k_encoded  = max(0, min(K_ATTACK_MAX, (_raw_next - 1024) // 128))
b_start    = (7 + k_encoded) * 128
```

Transient position within the SHORT window:
```
pos_in_short = T - (P1 + b_start) = _raw_next - (7 + k) * 128
```

For the bucket boundary `_raw_next = 1024 + 128*k` (exact lower edge of k-range):
```
pos_in_short = 1024 + 128*k - (7+k)*128 = 1024 - 896 = 128   ← window center, w≈1.0
```

For the top of the bucket `_raw_next = 1024 + 128*k + 127`:
```
pos_in_short = 128 + 127 = 255   ← window end, w≈0.006
```

So `(7+k)*128` **always** places the transient in n=128..255 (the right/new-samples half), with position 128 (maximum sine weight) when the transient aligns exactly with the bucket boundary.

### Why (8+k) is wrong

With `b_start = (8+k)*128`:
```
pos_in_short = 1024 + 128*k - (8+k)*128 = 0   ← window start, w≈0.006
```

The transient lands at n=0, where the sine window is essentially zero. The onset is invisible to the quantizer — (8+k) is catastrophically wrong.

### Why naive analysis gives "0/40 transients in SHORT"

A script that traces `codingParams.blockType` at WriteDataBlock **entry** (before the state machine runs) reads the block type of the *previous* block, not the current one. The LONG block that becomes START has blockType=LONG at entry. Its state machine fires at lines ~498-510 and changes blockType to START before processing — so the print at entry shows "LONG" for what is actually the START block. Every block type appears shifted by one, making the Short block appear to contain nothing.

---

## Key tradeoff

| | Version A | Version B |
|---|---|---|
| Transient placement | Mostly STOP | Always in SHORT right half (n=128..255) |
| b_start range | 128–896 (always 896) | 896–1152 (K_ATTACK_MAX=2) |
| MEDIUM always present | No (k=0 → none) | Yes |
| min_spacing | 2048 samples | 2176 samples |
| Transient detector | Envelope follower + local-min | SPE (128-sample resolution) |
| Events on Van_124 | 40 | 80 |

Version A sounded better on Van_124 at 96 kbps as of 2026-04-26 with the envelope-follower detector. SPE integration (2026-05-01) is more sensitive and finer-grained — Version B + SPE not yet A/B-tested against Version A.

---

## Analysis: history of Version A vs B comparison (2026-04-26 → 2026-05-01)

### k is always 7 in Version A

Version A formula: `k = max(0, min(7, d // 128 - 1))`

All second-block transients have d ∈ [1024, 2047]. For d ≥ 1024: `d // 128 ≥ 8`, so `8 - 1 = 7`. k is always clamped to 7 — the 3-bit bitstream field carries no useful information on this material. `b_start = max(128, 7*128) = 896` every single time. Version A's cascade is effectively a constant: same window shape every block.

### Transient detector precision mattered for Version B

With the envelope-follower detector, `TD.py`'s gate opened ~10–50 samples after the true acoustic onset. If b_start was set for a late detection, the actual transient onset fell in the MEDIUM block. Version A was immune — it ignored the detected position.

Local-minimum onset detection (2026-04-26) fixed this: walk backward from the envelope peak to the local minimum in the 200-sample pre-peak window. This moved positions 37–193 samples earlier, and Version B improved significantly.

Gate-filter fix (2026-04-26): adaptive MAX_SHIFT walk-back introduced a phantom event by changing block assignment. Fix: spacing filter uses `gate_block` (pre-shift) rather than post-shift `sample_index`. Result: 40 clean events on Van_124, all in SHORT.

### SPE replaces envelope follower (2026-05-01)

SPE detects transients from energy ratios between sub-blocks (128-sample resolution, 4 layers). No gate latency, no walk-back needed. Reports `sample_index` directly at the 128-sample resolution of the finest firing layer. On Van_124 at 96 kbps: 80 events (vs 40 with local-min). K_ATTACK_MAX=2 caps b_start at 1152 samples.

Version B + SPE vs Version A has not yet been A/B-tested perceptually. With K_ATTACK_MAX=2 (b_start ∈ [896, 1152]) the MEDIUM windows are small — the "large MEDIUM" concern from the k=7 era (b_start up to 1792) no longer applies.

**Encodes for reference**:
- `Van_124_bs_B_adaptive.rmc` / `.wav` — Version B + SPE (K_ATTACK_MAX=2, 80 events, current)
- `Van_124_bs_B_localmin.wav` — Version B + envelope local-min, 40 events
- `Van_124_bs_B_compensated.wav` — Version B + fixed 20-sample offset (mixed results)
- `Van_124_bs_adaptive_2.wav` — Version A (fixed cascade, transients mostly in STOP)
