"""
features.py -- Feature flags for the RMC codec.

Set a flag to False to disable that feature. Encoder and decoder must
use the same settings; .rmc files are only valid under the features.py
they were encoded with.
"""

ENTROPY_CODING       = True   # range-coded mantissas; False = raw bit writes
VARIABLE_BIT_RATE    = True   # adaptive entropy-inflation EMA; False = inflation fixed at 1.0
BLOCK_SWITCHING      = True   # transient-driven SHORT blocks; False = all LONG
MID_SIDE_CODING      = True   # M/S stereo; False = always L/R
PREDICTION           = True   # rhythmic prediction search; False = no prediction
SHORT_BLOCK_BITBOOST = True   # 2× bit budget for SHORT blocks; False = 1× (requires BLOCK_SWITCHING)
