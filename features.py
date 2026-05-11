"""
features.py -- Feature flags for the RMC codec.

Set a flag to False to disable that feature. Encoder and decoder must
use the same settings; .rmc files are only valid under the features.py
they were encoded with.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RMCFeatures:
    # Use entropy coding to compress mantissas
    ENTROPY_CODING: bool = False
    # adaptive entropy-inflation EMA; False = inflation fixed at 1.0
    VARIABLE_BIT_RATE: bool = False
    # transient-driven SHORT blocks; False = all LONG
    BLOCK_SWITCHING: bool = False
    # M/S stereo; False = always L/R
    MID_SIDE_CODING: bool = False
    # rhythmic prediction search; False = no prediction
    PREDICTION: bool = False
    # 2× bit budget for SHORT blocks; False = 1× (requires BLOCK_SWITCHING)
    SHORT_BLOCK_BITBOOST: bool = False
    # per-SFB complex gains; False = real scalar gain only (requires PREDICTION)
    COMPLEX_PREDICTION: bool = True
    # Per-band prediction enable criterion (mutually exclusive; default is plain max comparison)
    PRED_ENABLE_RMS: bool = True  # compare per-band RMS instead of peak (captures crest factor / underflow risk)
    # compare ScaleFactor(peak) — conservative power-of-2 threshold
    PRED_ENABLE_SF: bool = False
    # nLines-aware break-even threshold (COMPLEX_PREDICTION only).
    # When True, replaces flat PRED_ENABLE_RATIO with per-band threshold derived from the
    # break-even condition: savings(n_lines * reduction_dB/6) >= 7-bit per-band cost.
    # => threshold(b) = 10^(-42/(20*nLines[b]))
    PRED_NLINES_THRESH: bool = True
    # Fallback flat RMS ratio used when PRED_NLINES_THRESH=False and COMPLEX_PREDICTION=True.
    # 0.5 = require ≥6 dB improvement.
    PRED_ENABLE_RATIO: float = 0.75
    # Prediction search tuning, add 2bar/4bar options; expands pred_type field from 2→3 bits in bitstream
    PRED_EXTENDED_RANGE: bool = True
    # Maximum SFB index (exclusive) eligible for prediction; None = all bands.
    # Bands >= PRED_MAX_SFB are never predicted. Useful for evaluating whether
    # high-frequency prediction is bit-positive after the 7-bit per-band overhead.
    PRED_MAX_SFB: int | None = None
    # position-adaptive START/STOP windows that align with detected transient; requires BLOCK_SWITCHING
    AC2A_BLOCK_SWITCHING: bool = True
    # telescoping pre-attack cascade: START*(b=256/512) + MEDIUM(s) replace pre-attack SHORTs; requires AC2A_BLOCK_SWITCHING
    ADAPTIVE_CASCADE: bool = True
    # cap on k_attack (0..15); limits max MEDIUM window size; requires ADAPTIVE_CASCADE
    K_ATTACK_MAX: int = 15
