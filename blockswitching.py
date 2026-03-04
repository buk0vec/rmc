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
from window import KBDWindow

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
        Index (0..N_SHORT_BLOCKS-1) of the first sub-block where the transient
        occurs, or -1 if no transient is detected.
    """
    #if all zeros (so first block) don't say it's a transient
    if np.max(np.abs(prev_data)) == 0:
        return -1

    N = len(data)
    sub_size = N // N_SHORT_BLOCKS
    prev_rms = np.sqrt(np.mean(prev_data**2))

    for i in range(N_SHORT_BLOCKS):
        sub = data[i*sub_size:(i+1)*sub_size]
        sub_rms = np.sqrt(np.mean(sub**2))
        if sub_rms > threshold * prev_rms:
            return i
    return -1


# ---------------------------------------------------------------------------
# Block type selection (state machine)
# ---------------------------------------------------------------------------

def SelectBlockType(k_attack, prev_block_type):
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
        k_attack        -- int, output of DetectTransient(): attack sub-window
                           index (0..N_SHORT_BLOCKS-1), or -1 if no transient
        prev_block_type -- one of {LONG, START, SHORT, STOP}

    Returns:
        The block type for the current block: one of {LONG, START, SHORT, STOP}
    """
    if k_attack >= 0:
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
    Returns the KBD window of length N used for LONG blocks (alpha=4).

    AAC uses KBD windows for long blocks instead of sine windows.
    KBD has much lower spectral sidelobes (~-100 dB vs ~-30 dB for sine),
    reducing leakage and improving psychoacoustic model accuracy.

    Arguments:
        N -- total window length (= 2 * nMDCTLines_long, e.g. 2048)

    Returns:
        numpy array of shape (N,)
    """
    return KBDWindow(np.ones(N), alpha=4)


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

    # left half: KBD long window (rising side)
    kbd_long = KBDWindow(np.ones(N_long), alpha=4)
    w[:N_long//2] = kbd_long[:N_long//2]

    # flat top
    w[N_long//2 : N_long//2 + pad] = 1.0

    # taper down with right half of short sine window
    for n in range(N_short//2):
        w[N_long//2 + pad + n] = np.sin(np.pi * (0.5 + n + N_short//2) / N_short)

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

    # rise with left half of short sine window
    for n in range(N_short//2):
        w[pad + n] = np.sin(np.pi * (0.5 + n) / N_short)

    # flat top
    w[N_short//2 + pad : N_long//2] = 1.0

    # right half: KBD long window (falling side)
    kbd_long = KBDWindow(np.ones(N_long), alpha=4)
    w[N_long//2:] = kbd_long[N_long//2:]

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
    nLines = np.array([4, 4, 4, 4, 4, 8, 8, 8, 12, 12, 12, 16, 16, 16]) #from AAC short window 
  
    return ScaleFactorBands(nLines)


# ---------------------------------------------------------------------------
# Window grouping (attack-isolation)
# ---------------------------------------------------------------------------

GROUPING_BITS = N_SHORT_BLOCKS - 1  # 7 bits to encode group boundaries


def group_lens_to_mask(group_lens):
    """
    Converts a list of group lengths to a 7-bit integer mask.

    Bit i (LSB=0) is set if a new group starts after short window i.
    For example, group_lens=[3,1,4] means boundaries after windows 2 and 3,
    giving mask = 0b0001100 = 12.

    mask=0   means all 8 windows in one group (maximum sharing).
    mask=127 means every window in its own group (no sharing).

    Arguments:
        group_lens -- list of ints summing to N_SHORT_BLOCKS, each giving
                      the number of windows in that group (e.g. [3, 1, 4])

    Returns:
        7-bit integer mask
    """
    mask = 0
    idx = 0
    for gl in group_lens[:-1]:   # skip last group -- no boundary after it
        idx += gl                # idx is now the first window of the next group
        mask |= (1 << (idx - 1)) # so there's a boundary after window (idx-1)
    return int(mask)


def mask_to_group_lens(mask):
    """
    Converts a 7-bit integer mask to a list of group lengths.

    Inverse of group_lens_to_mask. Reads bits 0-6; each set bit marks a
    group boundary after that window index. Returns a list of group lengths
    summing to N_SHORT_BLOCKS.

    Arguments:
        mask -- 7-bit integer (0 = all one group, 127 = all singletons)

    Returns:
        list of group lengths, e.g. [3, 1, 4]
    """
    group_lens = []
    current_len = 0
    for i in range(N_SHORT_BLOCKS - 1):   # bits 0..6
        current_len += 1
        if mask & (1 << i):               # boundary after window i
            group_lens.append(current_len)
            current_len = 0
    current_len += 1                      # last window always ends a group
    group_lens.append(current_len)
    return group_lens


def SelectWindowGroups(k_attack, max_groups=4):
    """
    Returns a 7-bit group mask for the N_SHORT_BLOCKS short windows using
    attack-isolation grouping.

    Isolates the attack window (found by DetectTransient) in its own group,
    lumping pre-attack and post-attack windows into their own groups.
    If k_attack == -1 (no attack), returns mask=0 (all 8 windows in one group).

    Algorithm:
        1. If k_attack < 0: return mask=0 (all one group).
        2. Otherwise isolate the attack window k_attack:
               pre-attack:  [0 .. k_attack-1]   (omitted if k_attack == 0)
               attack:      [k_attack]
               post-attack: [k_attack+1 .. 7]   (omitted if k_attack == 7)
        3. Merge the smallest group into its smaller neighbor until
           len(groups) <= max_groups.
        4. Convert group lengths to mask via group_lens_to_mask and return.

    Arguments:
        k_attack   -- int, attack sub-window index from DetectTransient(),
                      or -1 if no transient (returns mask=0)
        max_groups -- maximum number of groups allowed (default 4)

    Returns:
        7-bit integer mask (use mask_to_group_lens to recover group lengths)
    """
    # 1. No attack -> all one group
    if k_attack < 0:
        return 0

    # 2. Isolate the attack window
    groups = []
    if k_attack > 0:
        groups.append(k_attack)           # pre-attack group length
    groups.append(1)                      # attack window alone
    post = N_SHORT_BLOCKS - k_attack - 1
    if post > 0:
        groups.append(post)               # post-attack group length

    # 5. Merge smallest group into its smaller neighbor until <= max_groups
    while len(groups) > max_groups:
        # find the smallest group
        idx = int(np.argmin(groups))
        if idx == 0:
            # merge with right neighbor
            groups[1] += groups[0]
            groups.pop(0)
        elif idx == len(groups) - 1:
            # merge with left neighbor
            groups[-2] += groups[-1]
            groups.pop()
        else:
            # merge with the smaller of left/right neighbor
            if groups[idx - 1] <= groups[idx + 1]:
                groups[idx - 1] += groups[idx]
            else:
                groups[idx + 1] += groups[idx]
            groups.pop(idx)

    # 6. Convert to mask
    return group_lens_to_mask(groups)
