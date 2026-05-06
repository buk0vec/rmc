"""
features.py -- Feature flags for the RMC codec.

Set a flag to False to disable that feature. Encoder and decoder must
use the same settings; .rmc files are only valid under the features.py
they were encoded with.
"""

# range-coded mantissas; False = raw bit writes
ENTROPY_CODING = False
# adaptive entropy-inflation EMA; False = inflation fixed at 1.0
VARIABLE_BIT_RATE = False
# transient-driven SHORT blocks; False = all LONG
BLOCK_SWITCHING = False
# M/S stereo; False = always L/R
MID_SIDE_CODING = False
# rhythmic prediction search; False = no prediction
PREDICTION = True
# 2× bit budget for SHORT blocks; False = 1× (requires BLOCK_SWITCHING)
SHORT_BLOCK_BITBOOST = False
# per-SFB complex gains; False = real scalar gain only
COMPLEX_PREDICTION = True


# Per-band prediction enable criterion (mutually exclusive; default is plain max comparison)
PRED_ENABLE_RMS = True  # compare per-band RMS instead of peak (captures crest factor / underflow risk)
PRED_ENABLE_SF = False  # compare ScaleFactor(peak) — conservative power-of-2 threshold

# nLines-aware break-even threshold (COMPLEX_PREDICTION only).
# When True, replaces flat PRED_ENABLE_RATIO with per-band threshold derived from the
# break-even condition: savings(n_lines * reduction_dB/6) >= 7-bit per-band cost.
# => threshold(b) = 10^(-66/(20*nLines[b]))
# Low bands (2-5 lines) need 13-22 dB; wide high bands (30-152 lines) need 1-2 dB.
PRED_NLINES_THRESH = False

# Fallback flat RMS ratio used when PRED_NLINES_THRESH=False and COMPLEX_PREDICTION=True.
# 0.5 = require ≥6 dB improvement.
PRED_ENABLE_RATIO = 0.75

# Prediction search tuning, add 2bar/4bar options; expands pred_type field from 2→3 bits in bitstream
PRED_EXTENDED_RANGE = True
