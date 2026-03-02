"""
blockswitching.py -- Block switching and transient detection for the RMC codec.

Implements the logic to switch between long and short MDCT blocks to improve
time resolution near transients (attacks), reducing pre-echo artifacts.

When a transient is detected, the encoder transitions:
    LONG -> START -> SHORT (x N) -> STOP -> LONG

-----------------------------------------------------------------------
Music 422 -- RMC Project
-----------------------------------------------------------------------
"""

import numpy as np
from psychoac import ScaleFactorBands, AssignMDCTLinesFromFreqLimits

# ---------------------------------------------------------------------------
# Block type constants
# ---------------------------------------------------------------------------

LONG  = 0   # Normal long block (high frequency resolution)
START = 1   # Transition block: long -> short
SHORT = 2   # Short block (high time resolution, used near transients)
STOP  = 3   # Transition block: short -> long

# Number of short blocks that replace one long block
N_SHORT_BLOCKS = 8


# ---------------------------------------------------------------------------
# Transient detection
# ---------------------------------------------------------------------------

def DetectTransient(data, prev_data, threshold=10.0):
    """
    Detects whether a transient (sudden amplitude increase) occurs in `data`.

    Splits `data` into N_SHORT_BLOCKS equal sub-blocks and computes the RMS
    energy of each. Compares each sub-block's energy to the average RMS energy
    of `prev_data`. A transient is declared if any sub-block's energy exceeds
    the previous block's average energy by more than `threshold` (a ratio).

    This catches attacks like drum hits or plucked strings where energy rises
    sharply within a block, causing pre-echo if a long MDCT window is used.

    Arguments:
        data        -- numpy array of N time-domain samples (current block)
        prev_data   -- numpy array of N time-domain samples (previous block),
                       used to establish baseline energy
        threshold   -- energy ratio above which a transient is declared
                       (e.g. 10.0 means 10x energy increase -> transient)

    Returns:
        True if a transient is detected, False otherwise
    """
    #if all zeros (so first block) don't say it's a transient
    if np.max(np.abs(prev_data)) == 0:
        return False

    N = len(data)
    sub_size = N // N_SHORT_BLOCKS
    prev_rms = np.sqrt(np.mean(prev_data**2))

    for i in range(N_SHORT_BLOCKS):
        sub = data[i*sub_size:(i+1)*sub_size]
        sub_rms = np.sqrt(np.mean(sub**2))
        if sub_rms > threshold * prev_rms:
            return True
    return False


# ---------------------------------------------------------------------------
# Block type selection (state machine)
# ---------------------------------------------------------------------------

def SelectBlockType(transient_detected, prev_block_type):
    """
    State machine that decides the block type to use for the *current* block,
    given whether a transient was detected and what the previous block's type was.

    Valid transitions:
        LONG  + no transient  -> LONG
        LONG  + transient     -> START
        START + (any)         -> SHORT   (committed to short blocks now)
        SHORT + transient     -> SHORT   (stay in short blocks)
        SHORT + no transient  -> STOP    (begin transition back)
        STOP  + (any)         -> LONG    (back to normal)

    Arguments:
        transient_detected  -- bool, output of DetectTransient()
        prev_block_type     -- one of {LONG, START, SHORT, STOP}

    Returns:
        The block type for the current block: one of {LONG, START, SHORT, STOP}
    """
    if transient_detected:
        if prev_block_type == LONG:
            return START
        elif prev_block_type == START:
            return SHORT
        elif prev_block_type == SHORT:
            return SHORT
        elif prev_block_type == STOP:
            return LONG
    else:
        if prev_block_type == SHORT:
            return STOP
        elif prev_block_type == STOP:
            return LONG
        elif prev_block_type == LONG:
            return LONG
        elif prev_block_type == START:
            return SHORT


# ---------------------------------------------------------------------------
# Window functions
# ---------------------------------------------------------------------------

def LongWindowFunc(N):
    """
    Returns the sine window of length N used for LONG blocks.

    w[n] = sin(pi * (n + 0.5) / N)   for n in [0, N)

    This is the same formula as SineWindow in window.py, but written here
    so the block switcher controls its own windowing independently.

    Arguments:
        N -- total window length (= 2 * nMDCTLines_long, e.g. 2048)

    Returns:
        numpy array of shape (N,)
    """
    w = np.zeros(N)
    for n in range(N):
        w[n] = np.sin(np.pi * (n + 0.5) / N)

    return w


def ShortWindowFunc(N_short):
    """
    Returns the sine window of length N_short used for SHORT blocks.

    Same formula as LongWindowFunc but for the short block size.
    This window is applied to each of the N_SHORT_BLOCKS sub-blocks
    independently during a run of short blocks.

    Arguments:
        N_short -- total short window length (= 2 * nMDCTLines_short, e.g. 256)

    Returns:
        numpy array of shape (N_short,)
    """
    w = np.zeros(N_short)
    for n in range(N_short):
        w[n] = np.sin(np.pi * (n + 0.5) / N_short)

    return w


def StartWindowFunc(N_long, N_short):
    """
    Returns the START transition window of length N_long.

    Used for the long block immediately *before* switching to short blocks.
    The left half is a normal long sine window. The right half flattens out,
    applies a short sine taper, and pads with zeros -- so the effective right
    overlap region matches what a short block expects.

    Shape for N_long=2048, N_short=256 (pad = (N_long - N_short) // 2 = 896):
        n in [0,                    N_long//2 - 1]:  sin(pi*(n+0.5)/N_long)  <- normal left half
        n in [N_long//2,            N_long//2 + pad - 1]:  1.0               <- flat top
        n in [N_long//2 + pad,      N_long//2 + pad + N_short//2 - 1]:
                                    right half of short sine window            <- taper down
        n in [N_long//2 + pad + N_short//2, N_long - 1]:  0.0               <- zero pad

    Arguments:
        N_long  -- long window length (= 2 * nMDCTLines_long)
        N_short -- short window length (= 2 * nMDCTLines_short)

    Returns:
        numpy array of shape (N_long,)
    """
    w = np.zeros(N_long)
    pad = (N_long//4 - N_short//4)

    #normal first half of window
    for n in range(N_long // 2):
        w[n] = np.sin(np.pi * (n + 0.5) / N_long)

    #pad of 1's
    for n in range(N_long // 2, N_long // 2 +pad):
        w[n] = 1.0
    
    #transition to short first half
    for n in range(N_short//2):
        w[n+ (N_long//2+pad)] = np.sin(np.pi * ((0.5 + n+N_short//2)/ N_short))

    return w


def StopWindowFunc(N_long, N_short):
    """
    Returns the STOP transition window of length N_long.

    Mirror image of StartWindowFunc. Used for the long block immediately
    *after* a run of short blocks. The left half zero-pads then rises with
    the left half of a short sine window. The right half is a normal long
    sine window.

    Shape for N_long=2048, N_short=256 (pad = (N_long - N_short) // 2 = 896):
        n in [0,                            pad - 1]:           0.0           <- zero pad
        n in [pad,                          pad + N_short//2 - 1]:
                                    left half of short sine window             <- rise
        n in [pad + N_short//2,     N_long//2 - 1]:            1.0           <- flat top
        n in [N_long//2,            N_long - 1]:  sin(pi*(n+0.5)/N_long)    <- normal right half

    Arguments:
        N_long  -- long window length (= 2 * nMDCTLines_long)
        N_short -- short window length (= 2 * nMDCTLines_short)

    Returns:
        numpy array of shape (N_long,)
    """
    w = np.zeros(N_long)
    pad = (N_long//4 - N_short//4)
    
    #short right side
    for n in range(N_short//2):
        w[n+pad] = np.sin(np.pi * (0.5 + n)/N_short)
    
    #1's padding
    for n in range(pad):
        w[n+N_short//2+pad] = 1.0
    
    #long right side
    for n in range(N_long//2):
        w[n + N_short//2+pad+pad] = np.sin(np.pi* (n+ N_long//2+ 0.5)/ N_long)

    return w


def WindowForBlockType(block_type, N_long, N_short):
    """
    Convenience dispatcher: returns the window array for the given block type.

    Arguments:
        block_type -- one of {LONG, START, SHORT, STOP}
        N_long     -- long window length (= 2 * nMDCTLines_long)
        N_short    -- short window length (= 2 * nMDCTLines_short)

    Returns:
        numpy array of length N_long (for LONG/START/STOP) or N_short (for SHORT)
    """
    if block_type == LONG:
        return LongWindowFunc(N_long)
    elif block_type == SHORT:
        return ShortWindowFunc(N_short)
    elif block_type == START:
        return StartWindowFunc(N_long, N_short)
    elif block_type == STOP:
        return StopWindowFunc(N_long, N_short)


# ---------------------------------------------------------------------------
# Short-block scale factor bands
# ---------------------------------------------------------------------------

def ShortBlockSFBands(nMDCTLines_short, sampleRate):
    """
    Returns a ScaleFactorBands object sized for short MDCT blocks.

    Short blocks have far fewer MDCT lines than long blocks (e.g. 128 vs 1024),
    so the critical-band mapping must be recomputed for the smaller block.
    Calls AssignMDCTLinesFromFreqLimits with nMDCTLines_short to get the
    right line-to-band assignment, then wraps it in a ScaleFactorBands object.

    Note: some bands may end up with 0 lines at this resolution -- you may
    want to filter those out before passing to BitAlloc.

    Arguments:
        nMDCTLines_short -- number of MDCT lines in a short block (e.g. 128)
        sampleRate       -- audio sample rate in Hz (e.g. 44100)

    Returns:
        ScaleFactorBands object appropriate for short-block encoding
    """
    nLines = np.array([4, 4, 4, 4, 4, 8, 8, 8, 12, 12, 12, 16, 16, 16]) #from AAC short window documentation
  
    return ScaleFactorBands(nLines)


# ---------------------------------------------------------------------------
# Window grouping (spectral similarity-based)
# ---------------------------------------------------------------------------

def _band_log_energy(mdct_lines, sfBands):
    """
    Helper: returns a vector of per-band log energies (dB) for a set of MDCT lines.

    For each scale factor band, sums the squared MDCT coefficients and converts
    to dB. Bands with zero energy return -inf.

    Arguments:
        mdct_lines -- numpy array of nMDCTLines_short MDCT coefficients
        sfBands    -- ScaleFactorBands object for the short block

    Returns:
        numpy array of shape (sfBands.nBands,) with per-band energy in dB
    """
    pass  ### YOUR CODE STARTS HERE ###


def SelectWindowGroups(data, nMDCTLines_short, sampleRate, similarity_threshold=3.0, max_groups=4):
    """
    Decides how to group the N_SHORT_BLOCKS short windows for scale factor sharing,
    based on spectral similarity between adjacent windows.

    Windows with similar spectral content are grouped together to share scale
    factors and bit allocations, saving bits without hurting quality. Windows
    with dissimilar spectra are kept separate so each gets its own scale factors.

    Algorithm:
        1. Compute the MDCT of each of the N_SHORT_BLOCKS sub-windows in `data`
           using ShortWindowFunc.
        2. Compute per-band log energy via _band_log_energy for each window.
        3. Start with each window as its own singleton group.
        4. Iteratively merge adjacent groups whose per-band log energy vectors
           are most similar (lowest mean absolute difference across bands),
           as long as that difference is below similarity_threshold (in dB)
           and len(groups) > 1.
        5. Stop merging when no adjacent pair is below the threshold or
           len(groups) <= 1. Also respect max_groups as a hard cap.

    The group structure is encoded in the bitstream as a list of group lengths
    (e.g. [3, 1, 4]) which can be represented in 7 bits (one boundary bit per
    window 1-7). This is only transmitted when block_type == SHORT.

    Arguments:
        data                 -- numpy array of N_long time-domain samples (full
                                long block, not windowed)
        nMDCTLines_short     -- number of MDCT lines in a short block (e.g. 128)
        sampleRate           -- audio sample rate in Hz (e.g. 44100)
        similarity_threshold -- max mean absolute dB difference between adjacent
                                groups' spectral profiles to allow merging (default 3.0)
        max_groups           -- maximum number of groups allowed (default 4)

    Returns:
        groups -- list of lists of window indices, e.g.
                  [[0,1,2], [3], [4,5,6,7]] means windows 0-2 share one scale
                  factor set, window 3 has its own, windows 4-7 share one.
                  No grouping: [[0],[1],[2],[3],[4],[5],[6],[7]]
    """
    pass  ### YOUR CODE STARTS HERE ###


def GroupedSFBands(groups, nMDCTLines_short, sampleRate):
    """
    Returns a list of ScaleFactorBands objects, one per group.

    Each group shares one scale factor set across all its windows. Since every
    window still has its own nMDCTLines_short MDCT lines and its own mantissas,
    the sfBands structure is the same for every group regardless of group size
    -- it is always built from nMDCTLines_short lines via ShortBlockSFBands.

    In codec.py, for each group you will:
        - Find the worst-case (max) scale factor across all windows in the group
          for each band, and transmit that one value
        - Quantize each window's mantissas using that shared scale factor

    Arguments:
        groups           -- list of lists of window indices (output of
                            SelectWindowGroups)
        nMDCTLines_short -- number of MDCT lines in a single short window
        sampleRate       -- audio sample rate in Hz

    Returns:
        list of ScaleFactorBands objects, one per group (len == len(groups))
    """
    pass  ### YOUR CODE STARTS HERE ###
