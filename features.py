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
    # transient-driven SHORT blocks; False = all LONG
    BLOCK_SWITCHING: bool = False
    # rhythmic prediction search; False = no prediction
    PREDICTION: bool = False
    # nLines-aware break-even threshold for complex prediction.
    # When True, uses per-band threshold derived from the break-even condition:
    # savings(n_lines * reduction_dB/6) >= 7-bit per-band cost.
    # => threshold(b) = 10^(-42/(20*nLines[b]))
    PRED_NLINES_THRESH: bool = True
    # Fallback flat RMS ratio used when PRED_NLINES_THRESH=False.
    # 0.5 = require ≥6 dB improvement.
    PRED_ENABLE_RATIO: float = 0.75
    # Maximum SFB index (exclusive) eligible for prediction; None = all bands.
    # Bands >= PRED_MAX_SFB are never predicted.
    PRED_MAX_SFB: int | None = None
