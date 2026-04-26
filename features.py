"""
features.py -- Feature flags for the RMC codec.

Set a flag to False to disable that feature. Encoder and decoder must
use the same settings; .rmc files are only valid under the features.py
they were encoded with.
"""

ENTROPY_CODING       = False  # range-coded mantissas; False = raw bit writes
VARIABLE_BIT_RATE    = False  # adaptive entropy-inflation EMA; False = inflation fixed at 1.0
BLOCK_SWITCHING      = True   # transient-driven SHORT blocks; False = all LONG
MID_SIDE_CODING      = False  # M/S stereo; False = always L/R
PREDICTION           = False  # rhythmic prediction search; False = no prediction
SHORT_BLOCK_BITBOOST = False  # 2× bit budget for SHORT blocks; False = 1× (requires BLOCK_SWITCHING)
SUBBASS_HYBRID       = False  # sub-bass stays in LONG window during transients; change True/False to toggle (requires BLOCK_SWITCHING)
TNS                  = False  # Temporal Noise Shaping for SHORT blocks; helps reduce transient coloration
AC2A_BLOCK_SWITCHING = True   # position-adaptive START/STOP windows that align with detected transient; False = fixed Edler windows (requires BLOCK_SWITCHING)
ADAPTIVE_CASCADE     = True   # telescoping pre-attack cascade: START*(b=256/512) + MEDIUM(s) replace pre-attack SHORTs; False = fixed 7-SHORT events (requires AC2A_BLOCK_SWITCHING)
