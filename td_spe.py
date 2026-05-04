"""
td_spe.py -- SPE-based transient detector wrapper for AC-2A block switching.

Bridges the tri-band SPE detector (spe.py) into the event-dict format that
xrmc.py / blockswitching.py expect:

    [{"block": int, "gate_block": int,
      "sample_offset": int, "sample_index": int}, ...]

Drop-in replacement for `simple_run.detectTransients`.  Same signature,
same return shape, but uses the validated SPE detector instead of the
envelope-difference + walk-back detector.

Why a separate module:
  Keeps simple_run.py untouched so you can A/B test the two detectors by
  changing only one import line in xrmc.py:
      from simple_run import detectTransients   # old detector
      from td_spe       import detectTransients   # SPE detector

How block index → sample_index conversion works:
  SPE returns block indices b_idx (each block = nMDCTLines new samples).
  The "current half" of the SPE buffer for block b_idx covers samples
  [b_idx*nMDCTLines, (b_idx+1)*nMDCTLines).  The actual attack onset is
  somewhere in that range.  We refine by:
    1. Finding the local-maximum |sample| inside that block (the peak).
    2. Walking back from the peak to the local minimum of |audio| in a
       LOOKBACK window — that is where the rise begins (acoustic onset),
       analogous to simple_run.py's local-min walk-back.
  This gives AC-2A Version B a precise sample_index so the cascade k value
  reflects the true attack position within the block.

Music 422 -- RMC Project
"""

import numpy as np
import scipy.io.wavfile

from spe import detectTransientsSPE

# ---------------------------------------------------------------------------
# Walk-back parameters (mirror simple_run.py local-min logic)
# ---------------------------------------------------------------------------

# Walk back this many samples from the in-block peak to find the local minimum.
# An attack onset rises over ~5 ms; 200 samples (~4.5 ms at 44.1 kHz) is enough
# to span the rising edge.  Refinement is also clamped to stay WITHIN the SPE-
# flagged block — SPE specifically said the attack is in block N, so we never
# shift the sample_index out of block N.
_PEAK_LOOKBACK = 200


def _refine_onset(audio_mono: np.ndarray,
                  block_start: int,
                  block_size: int) -> int:
    """
    Refine an SPE block index to a precise sample-level onset position
    INSIDE the SPE-flagged block.

    Strategy:
      peak  = argmax(|audio|) in [block_start, block_start + block_size)
      onset = argmin(|audio|) in [max(block_start, peak - LOOKBACK), peak)

    The within-block clamp guarantees `sample_index // block_size == b_idx`
    so SPE's block-level decision is preserved.  Falls back to block_start
    on degenerate inputs.
    """
    n = len(audio_mono)
    pk_lo = max(0, block_start)
    pk_hi = min(n, block_start + block_size)
    if pk_hi <= pk_lo:
        return block_start

    peak = pk_lo + int(np.argmax(np.abs(audio_mono[pk_lo:pk_hi])))

    # Walk-back range stays inside the SPE-flagged block.
    mn_lo = max(block_start, peak - _PEAK_LOOKBACK)
    mn_hi = peak
    if mn_hi <= mn_lo:
        return peak

    onset = mn_lo + int(np.argmin(np.abs(audio_mono[mn_lo:mn_hi])))
    return onset


# ---------------------------------------------------------------------------
# Public API: same signature as simple_run.detectTransients
# ---------------------------------------------------------------------------

def detectTransients(audioPath: str,
                     sr: int = 22050,             # ignored, taken from file
                     duration=None,                # ignored
                     use_auto_params: bool = True, # ignored (SPE has its own tuning)
                     blockSize: int = 1024,
                     forceBlockSize=None,
                     cwt_onset_threshold: float = 0.3,    # ignored
                     time_threshold_factor: float = 0.25, # ignored
                     return_events: bool = False,
                     verbose: bool = True):
    """
    Detect transients via tri-band SPE and return either an array of block
    indices (`return_events=False`, legacy behaviour) or a list of
    block-switching event dicts (`return_events=True`).

    Most kwargs are accepted-and-ignored for signature compatibility with
    simple_run.detectTransients; SPE has its own internal parameters tuned
    in spe.py.
    """
    block_size = int(forceBlockSize if forceBlockSize is not None else blockSize)

    # ---- Load audio for onset refinement --------------------------------
    sr_file, raw = scipy.io.wavfile.read(audioPath)
    if raw.dtype == np.int16:
        audio = raw.astype(np.float64) / 32768.0
    elif raw.dtype == np.int32:
        audio = raw.astype(np.float64) / 2147483648.0
    else:
        audio = raw.astype(np.float64)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)   # mono for refinement only

    # ---- Run SPE --------------------------------------------------------
    blocks = detectTransientsSPE(audioPath,
                                 nMDCTLines=block_size,
                                 verbose=False)

    # ---- Build event dicts ----------------------------------------------
    events = []
    for b_idx in blocks:
        block_start = int(b_idx) * block_size
        sample_index = _refine_onset(audio, block_start, block_size)
        events.append({
            "block":         int(sample_index // block_size),
            "gate_block":    int(b_idx),                       # SPE-flagged block
            "sample_offset": int(sample_index %  block_size),
            "sample_index":  int(sample_index),
        })

    if verbose:
        print(f"  SPE transient events ({len(events)}):")
        for e in events[:20]:
            print(f"    block={e['block']:>4} sample_idx={e['sample_index']:>8} "
                  f"offset={e['sample_offset']:>4}  (gate_block={e['gate_block']})")
        if len(events) > 20:
            print(f"    ... and {len(events) - 20} more")

    if return_events:
        return events
    return np.array([e["block"] for e in events], dtype=int)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os, sys
    test_files = [
        os.path.join("inputs", n) for n in
        ("castanets.wav", "glockenspiel.wav", "harpsichord.wav",
         "Van_124.wav", "oboe.wav", "violin.wav", "violin2.wav")
    ]
    print("td_spe smoke test")
    print("=" * 60)
    for path in test_files:
        if not os.path.exists(path):
            print(f"[skip] {path}")
            continue
        events = detectTransients(path, return_events=True,
                                  forceBlockSize=1024, verbose=False)
        print(f"{os.path.basename(path):<22}: {len(events):>3} events")
        if events:
            sample_idxs = [e["sample_index"] for e in events]
            print(f"   first 5 sample_idx: {sample_idxs[:5]}")
            print(f"   first 5 offsets:    {[e['sample_offset'] for e in events[:5]]}")
