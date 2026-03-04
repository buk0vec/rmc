"""
Music 422
-----------------------------------------------------------------------
(c) 2009-2026 Marina Bosi  -- All rights reserved
-----------------------------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt

import window, mdct, psychoac

# Question 1.c)
def BitAllocUniform(bitBudget, maxMantBits, nBands, nLines, SMR=None):
    """
    Returns a hard-coded vector that, in the case of the signal used in HW#4,
    gives the allocation of mantissa bits in each scale factor band when
    bits are uniformly distributed for the mantissas.
    """
    nLines = np.asanyarray(nLines, dtype=int)
    total_lines = int(np.sum(nLines))

    if bitBudget <= 0 or total_lines <= 0:
        return np.zeros(nBands, dtype= int)
    
    #uniformly distribute bit budget into total lines
    bits_per_line = int(bitBudget // total_lines)

    #if uniform distribution is less than maxMant then do that, if not then distribute with maxMantBits
    bits_per_line = max(0, min(int(maxMantBits), bits_per_line))

    return bits_per_line *np.ones(nBands, dtype=int)

def _sanitize_bits(bits, maxMantBits):
    bits = np.asarray(bits, dtype=int)
    #enforces min of 0 and max of maxMantBits
    bits = np.clip(bits, 0, int(maxMantBits))

    #1 is illegal in this format
    bits[bits == 1] = 0  
    return bits


def BitAllocConstSNR(bitBudget, maxMantBits, nBands, nLines, peakSPL):
    """
    Returns a hard-coded vector that, in the case of the signal used in HW#4,
    gives the allocation of mantissa bits in each scale factor band when
    bits are distributed for the mantissas to try and keep a constant
    quantization noise floor (assuming a noise floor 6 dB per bit below
    the peak SPL line in the scale factor band).
    """
    nLines = np.asarray(nLines, dtype=int)
    peakSPL = np.asarray(peakSPL, dtype=float)

    #if out of bits set all critical bands to zero
    if bitBudget <= 0:
        return np.zeros(nBands, dtype = int)
    
    #bi-section search range for C which will be the target noise floor 
    lo = -400.0
    hi = float(np.max(peakSPL)) + 400.0

    def alloc_for_C(C):
        #ideal amount of mantissa bits per MDCT line 
        b = (peakSPL -C) / 6.0
        #if ideal is less than 0 set to 0, if ideal is greater than max Mant set to max, make sure ideal is in range
        b =np.clip(b, 0.0, float(maxMantBits))
        #set floor for fractions to be bit int value
        bi = np.floor(b).astype(int)
        #run sanitize for "no 1" requirement
        bi = _sanitize_bits(bi, maxMantBits)
        #how many lines per band multiplied by how many bits used
        used = int(np.sum(bi*nLines))
        #difference between ideal amount of bits and rounded down amount
        frac = b - np.floor(b)
        return bi, used, frac
    
    #to be filled
    best_bits = None
    best_frac = None

    #after 80 halvings the bisection should be very small
    for _ in range(80): 
        #find mid point
        mid = 0.5 * (lo+hi)
        bi, used, frac = alloc_for_C(mid)
        if used > bitBudget:
            # too many bits, raise C
            lo= mid 
        else: 
            #within budget, try lowering C
            hi = mid 
            best_bits = bi
            best_frac = frac
    #compute leftover bits after the "best" bisection allocation
    bits = best_bits.copy()
    #subtract from the total bitBudget 
    rem = bitBudget - int(np.sum(bits * nLines))
    #spend leftover bits
    #sort remainders from highest to lowest (bands that didn't get the amount they probably "wanted")
    order = np.argsort(-best_frac)

    #prioritized highest remainders
    for b in order:

        #loop through most deserving to least deserving remainders until there are none
        if rem <= 0:
            break
        #if out of bits then exit loop
        if nLines[b] <= 0:
            continue
        
        #to turn on a band go to 2 because 1 is illegal
        if bits[b] == 0:
            step = 2
            cost = 2 * int(nLines[b])

        #otherwise go up one step and calculate line cost
        else:
            step = 1
            cost = int(nLines[b])

        #if band has than two bits add one 
        if bits[b] + step > maxMantBits:
            continue
        
        #subtract cost from remaining extra bits budget, can't exceed max cap
        if cost <= rem:
            bits[b] += step
            rem -= cost
    #make sure nothing violates the "no 1 bit" or max Mant bits
    bits = _sanitize_bits(bits, maxMantBits)
    return bits


def BitAllocConstNMR(bitBudget, maxMantBits, nBands, nLines, SMR):
    """
    Returns a hard-coded vector that, in the case of the signal used in HW#4,
    gives the allocation of mantissa bits in each scale factor band when
    bits are distributed for the mantissas to try and keep the quantization
    noise floor a constant distance below (or above, if bit starved) the
    masked threshold curve (assuming a quantization noise floor 6 dB per
    bit below the peak SPL line in the scale factor band).
    """
    nLines = np.asarray(nLines, dtype=int)
    SMR = np.asarray(SMR, dtype =float)

    #if bitBudget is zero set all critical bands to zero
    if bitBudget <= 0:
        return np.zeros(nBands, dtype= int)
    
    #set super lowest SPL and super highest SPL for bisecting 
    lo = -400.0
    hi = float(np.max(SMR)+400.0)

    def alloc_for_C(C):
        #for NMR use SMR-C instead of peakSPL-C, gives more bits where signal is "more audible relative to mask" instead of relative to signal level
        b = (SMR - C) / 6.0
        b = np.clip(b, 0.0, float(maxMantBits))
        bi = np.floor(b).astype(int)
        bi = _sanitize_bits(bi, maxMantBits)
        used = int(np.sum(bi*nLines))
        frac = b - np.floor(b)
        return bi, used, frac
    
    best_bits = None
    best_frac = None

    for _ in range(80):
        mid = 0.5 * (lo+hi)
        bi, used, frac = alloc_for_C(mid)

        if used > bitBudget:
            lo = mid
        else:
            hi = mid
            best_bits = bi
            best_frac = frac

    bits = best_bits.copy()
    rem = bitBudget - int(np.sum(bits *nLines))

    order = np.argsort(-best_frac)

    for b in order:
        if rem <= 0:
            break
        if nLines[b] <= 0:
            continue

        if bits[b] == 0:
            #to "turn on" a band, must jump 0 to 2 (since 1 is illegal)
            step = 2
            cost = 2 * int(nLines[b])
        else:
            step = 1
            cost = int(nLines[b])

        if bits[b] + step > maxMantBits:
            continue

        if cost <= rem:
            bits[b] += step
            rem -= cost

    bits = _sanitize_bits(bits, maxMantBits)
    return bits

#1e.

# Pick what BitAlloc() returns
"""BITALLOC_MODE = "constnmr"   # "uniform", "constsnr", "constnmr"

_BITBUDGET_128 = 1161
_BITBUDGET_192 = 1844

_BITS_UNIFORM_128 = np.array([
    2,2,2,2,2,2,2,2,2,2,
    2,2,2,2,2,2,2,2,2,2,
    2,2,2,2,2
], dtype=int)

_BITS_CONSTSNR_128 = np.array([
     9,10,15,13,14, 9, 8,12, 9, 6,
     5, 4, 3, 3, 2, 2, 2, 8,11, 0,
     0,10, 0, 0, 0
], dtype=int)

_BITS_CONSTNMR_128 = np.array([
     7, 9,13,11,12, 8, 6,12, 8, 5,
     5, 4, 3, 3, 3, 2, 2, 8,11, 0,
     0,10, 0, 0, 0
], dtype=int)

_BITS_UNIFORM_192 = np.array([
    3,3,3,3,3,3,3,3,3,3,
    3,3,3,3,3,3,3,3,3,3,
    3,3,3,3,3
], dtype=int)

_BITS_CONSTSNR_192 = np.array([
    12,13,16,16,16,12,11,15,12, 9,
     8, 7, 6, 6, 5, 5, 5,10,14, 3,
     3,12, 3, 0, 0
], dtype=int)

_BITS_CONSTNMR_192 = np.array([
    10,12,15,14,15,10, 9,14,10, 8,
     7, 7, 6, 6, 5, 5, 5,10,13, 4,
     3,13, 3, 0, 0
], dtype=int)


def _sanitize(bits, maxMantBits):
    bits = np.asarray(bits, dtype=int)
    bits = np.clip(bits, 0, int(maxMantBits))
    bits[bits == 1] = 0
    return bits


def _pick_rate(bitBudget):
    bb = float(bitBudget)
    # choose whichever budget it is closest to
    return 128 if abs(bb - _BITBUDGET_128) < abs(bb - _BITBUDGET_192) else 192


def _select(bitBudget, bits_128, bits_192, nBands, maxMantBits):
    rate = _pick_rate(bitBudget)
    bits = bits_128 if rate == 128 else bits_192
    if len(bits) != nBands:
        raise ValueError(f"Hard-coded bits length {len(bits)} != nBands {nBands}")
    return _sanitize(bits, maxMantBits)

# These three must match BitAlloc() signature
def BitAllocUniform(bitBudget, maxMantBits, nBands, nLines, SMR):
    return _select(bitBudget, _BITS_UNIFORM_128, _BITS_UNIFORM_192, nBands, maxMantBits)

def BitAllocConstSNR(bitBudget, maxMantBits, nBands, nLines, SMR):
    return _select(bitBudget, _BITS_CONSTSNR_128, _BITS_CONSTSNR_192, nBands, maxMantBits)

def BitAllocConstNMR(bitBudget, maxMantBits, nBands, nLines, SMR):
    return _select(bitBudget, _BITS_CONSTNMR_128, _BITS_CONSTNMR_192, nBands, maxMantBits)


# codec.py calls THIS
def BitAlloc(bitBudget, maxMantBits, nBands, nLines, SMR):
    m = BITALLOC_MODE.lower()
    if m == "uniform":
        return BitAllocUniform(bitBudget, maxMantBits, nBands, nLines, SMR)
    if m == "constsnr":
        return BitAllocConstSNR(bitBudget, maxMantBits, nBands, nLines, SMR)
    if m == "constnmr":
        return BitAllocConstNMR(bitBudget, maxMantBits, nBands, nLines, SMR)
    raise ValueError(f"Unknown BITALLOC_MODE='{BITALLOC_MODE}'")

"""
# Question 2a.
def BitAlloc(bitBudget, maxMantBits, nBands, nLines, SMR):

    """Allocates bits to scale factor bands so as to flatten the NMR across the spectrum

       Arguments:
           bitBudget is total number of mantissa bits to allocate
           maxMantBits is max mantissa bits that can be allocated per line
           nBands is total number of scale factor bands
           nLines[nBands] is number of lines in each scale factor band
           SMR[nBands] is signal-to-mask ratio in each scale factor band

        Returns:
            bits[nBands] is number of bits allocated to each scale factor band
        """
    #basically chose similar outline to NMR for BitAlloc
    nLines = np.asarray(nLines, dtype=int)
    #SMR_tuned = np.clip(SMR, -10.0, 50.0)
    #SMR_tuned = SMR.copy()
    #SMR_tuned[20:] += 1.0 
    SMR = np.asarray(SMR, dtype=float)

    #integer budget in "mantissa bits" (not including scale factors, headers, etc)
    rem_budget = int(np.floor(bitBudget))
    if rem_budget <= 0 or nBands <= 0:
        return np.zeros(nBands, dtype=int)

    lo = -400.0
    hi = float(np.max(SMR)) + 400.0

    best_bits = np.zeros(nBands, dtype=int)
    best_frac = np.zeros(nBands, dtype=float)

    def alloc_for_C(C):
        b_float = (SMR - C) / 6.0
        b_float = np.clip(b_float, 0.0, float(maxMantBits))
        b_int = np.floor(b_float).astype(int)
        b_int = _sanitize_bits(b_int, maxMantBits)
        used = int(np.sum(b_int * nLines))
        frac = np.clip(b_float - b_int, 0.0, None)
        return b_int, used, frac

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        b_int, used, frac = alloc_for_C(mid)

        if used > rem_budget:
            lo = mid  #too many bits then raise C (allocate less)
        else:
            hi = mid  #within budget then try lower C (allocate more)
            best_bits = b_int
            best_frac = frac


    bits = best_bits.copy()
    used = int(np.sum(bits * nLines))
    rem = rem_budget - used

    order = np.argsort(-best_frac)

    for b in order:
        if rem <= 0:
            break
        if nLines[b] <= 0:
            continue

        if bits[b] >= maxMantBits:
            continue

        if bits[b] == 0:
            #to "turn on" a band, must jump 0 to 2 (since 1 is illegal)
            step = 2
            cost = 2 * int(nLines[b])
        else:
            step = 1
            cost = int(nLines[b])

        if bits[b] + step > maxMantBits:
            continue
        if cost > rem:
            continue

        bits[b] += step
        rem -= cost

    bits = _sanitize_bits(bits, maxMantBits)
    return bits.astype(np.uint64)

#-----------------------------------------------------------------------------

#Testing code
def mdct_center_freqs(Fs, N):
    k = np.arange(N//2, dtype=np.float64)
    return (k + 0.5) * (Fs / N)

if __name__ == "__main__":
    Fs = 48000
    N = 1024
    MDCTscale = 0
    maxMantBits = 10

    #make the HW4 signal
    A = [0.40, 0.24, 0.18, 0.08, 0.04, 0.02]
    freqs = [220, 330, 440, 880, 4400, 8800]
    n = np.arange(N, dtype=np.float64)
    x = sum(Ai*np.cos(2*np.pi*fi*n/Fs) for Ai, fi in zip(A, freqs))

    #window + MDCT (use Sine for consistency with psychoac.CalcSMRs)
    xw = window.SineWindow(x)
    Xmdct = mdct.MDCT(xw, N//2, N//2, isInverse=False)

    #build sfBands using psychoac helpers
    nMDCTLines = N // 2
    nLines = psychoac.AssignMDCTLinesFromFreqLimits(nMDCTLines, Fs)
    sfBands = psychoac.ScaleFactorBands(nLines)
    nBands = sfBands.nBands

    #masked threshold (per line) and SMR (per band)
    spl_masked = psychoac.getMaskedThreshold(x, Xmdct, MDCTscale, Fs, sfBands)
    SMR = psychoac.CalcSMRs(x, Xmdct, MDCTscale, Fs, sfBands)

    #need signal SPL per MDCT line (for plots) + peakSPL per band (for SNR + noise floor)
    w = window.SineWindow(np.ones(N, dtype=np.float64))
    w2_avg = np.mean(w**2)
    mdct_unscaled = Xmdct / (2.0 ** MDCTscale)
    I_mdct = (2.0 / w2_avg) * (mdct_unscaled * mdct_unscaled)
    spl_sig = psychoac.SPL(I_mdct)

    peakSPL = np.full(nBands, -np.inf, dtype=np.float64)
    for b in range(nBands):
        lo = int(sfBands.lowerLine[b])
        hi = int(sfBands.upperLine[b])
        if lo >= 0 and hi >= lo:
            peakSPL[b] = float(np.max(spl_sig[lo:hi+1]))

    def run_for_bitrate(bitrate_bps, tag):
        #compute bitBudget from bitrate minus overhead
        hop = N // 2
        blocks_per_sec = Fs / hop
        bits_per_block = int(np.floor(bitrate_bps / blocks_per_sec))

        overhead = 4 + nBands * (4 + 4)   # header + (scale factor + bit alloc info) per band
        bitBudget = bits_per_block - overhead

        #allocate bits
        bits_u = BitAllocUniform(bitBudget, maxMantBits, nBands, nLines)
        bits_s = BitAllocConstSNR(bitBudget, maxMantBits, nBands, nLines, peakSPL)
        bits_n = BitAllocConstNMR(bitBudget, maxMantBits, nBands, nLines, SMR)
        bits_hw = BitAlloc(bitBudget, maxMantBits, nBands, nLines, SMR)
        

        print(f"\n{tag}")
        #print tables
        

        print("band  lo hi  nLines  uniform  constSNR  constNMR")
        for b in range(nBands):
            lo = int(sfBands.lowerLine[b])
            hi = int(sfBands.upperLine[b])
            print(f"{b+1:3d}  {lo:3d} {hi:3d}  {int(nLines[b]):6d}  {bits_u[b]:7d}  {bits_s[b]:8d}  {bits_n[b]:8d}")
        #2b. table
        print(f"\n[2b] Bit allocation per critical band using BitAlloc() ({tag})")
        print("band\tlo\thi\tnLines\tmantBits")
        for b in range(nBands):
            lo = int(sfBands.lowerLine[b])
            hi = int(sfBands.upperLine[b])
            print(f"{b+1}\t{lo}\t{hi}\t{int(nLines[b])}\t{int(bits_hw[b])}")


        #plots

        def noise_floor_line(bits_band):
            nf_line = np.full(nMDCTLines, np.nan, dtype=np.float64)
            for b in range(nBands):
                lo = int(sfBands.lowerLine[b])
                hi = int(sfBands.upperLine[b])
                if lo >= 0 and hi >= lo:
                    nf = peakSPL[b] - 6.0 * float(bits_band[b])
                    nf_line[lo:hi+1] = nf
            return nf_line

        def plot_one(title, bits_band, filename):
            nf_line = noise_floor_line(bits_band)
            plt.figure()
            plt.semilogx(f_k, spl_sig, label="MDCT SPL")
            plt.semilogx(f_k, spl_masked, label="Masked Threshold")
            plt.step(f_k, nf_line, where="mid", label="Noise floor (bandwise)")
            plt.grid(True, which="both")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("dB SPL")
            plt.title(title)
            plt.legend()
            plt.savefig(filename, dpi=200, bbox_inches="tight")

        plot_one(f"{tag} (i) Uniform",   bits_u, f"{tag}_uniform.png")
        plot_one(f"{tag} (ii) ConstSNR", bits_s, f"{tag}_constSNR.png")
        plot_one(f"{tag} (iii) ConstNMR",bits_n, f"{tag}_constNMR.png")
        #2b: plot ( BitAlloc only) 
        plot_one(f"{tag} 2b. BitAlloc()", bits_hw, f"{tag}_2b_BitAlloc.png")



    #frequency for each MDCT line
    f_k = mdct_center_freqs(Fs, N)

    #run both parts
    run_for_bitrate(128000.0, "128kbps")
    run_for_bitrate(192000.0, "192kbps")

    plt.show()
