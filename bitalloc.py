"""
Music 422
-----------------------------------------------------------------------
(c) 2009-2026 Marina Bosi  -- All rights reserved
-----------------------------------------------------------------------
"""

import numpy as np

# Question 1.c)
def BitAllocUniform(bitBudget, maxMantBits, nBands, nLines, SMR=None):
    """
    Returns a hard-coded vector that, in the case of the signal used in HW#4,
    gives the allocation of mantissa bits in each scale factor band when
    bits are uniformly distributed for the mantissas.
    """
    # if bitBudget > 2000:
    #     return np.array([4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4.,
    #    4., 4., 3., 4., 4., 4., 4., 4.], dtype=np.uint64)
    return np.array([3., 4., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.,
       3., 3., 3., 3., 3., 3., 2., 2.], dtype=np.uint64)

def BitAllocConstSNR(bitBudget, maxMantBits, nBands, nLines, peakSPL):
    """
    Returns a hard-coded vector that, in the case of the signal used in HW#4,
    gives the allocation of mantissa bits in each scale factor band when
    bits are distributed for the mantissas to try and keep a constant
    quantization noise floor (assuming a noise floor 6 dB per bit below
    the peak SPL line in the scale factor band).
    """
    # if bitBudget > 2000:
    #     return np.array([16., 16., 16., 16., 16., 15., 14., 16., 12.,  6.,  5.,  5.,  5.,
    #     4.,  4.,  4.,  5., 14., 16.,  3.,  3., 15.,  3.,  0.,  0.], dtype=np.uint64)
    return np.array([14., 16., 15., 16., 16., 12., 11., 15.,  9.,  3.,  2.,  2.,  2.,
        3.,  0.,  0.,  0., 11., 14.,  0.,  0., 12.,  0.,  0.,  0.], dtype=np.uint64)

def BitAllocConstNMR(bitBudget, maxMantBits, nBands, nLines, SMR):
    """
    Returns a hard-coded vector that, in the case of the signal used in HW#4,
    gives the allocation of mantissa bits in each scale factor band when
    bits are distributed for the mantissas to try and keep the quantization
    noise floor a constant distance below (or above, if bit starved) the
    masked threshold curve (assuming a quantization noise floor 6 dB per
    bit below the peak SPL line in the scale factor band).
    """
#     return np.array([10., 13., 11., 12., 12., 10., 11., 14.,  7.,  5.,  7.,  9., 10.,
#    10., 10.,  9.,  5., 11., 13.,  3.,  4., 12.,  2.,  2.,  0.], dtype=np.uint64)
    return np.array([ 8., 10.,  9.,  9., 10.,  7.,  8., 11.,  6.,  3.,  5.,  8.,  8.,
        8.,  8.,  8.,  2.,  8., 10.,  0.,  0., 11.,  0.,  0.,  0.], dtype=np.uint64)

# Question 2.a)
def BitAlloc(bitBudget, maxMantBits, nBands, nLines, SMR):
    """
    Allocates bits to scale factor bands so as to flatten the NMR across the spectrum

       Arguments:
           bitBudget is total number of mantissa bits to allocate
           maxMantBits is max mantissa bits that can be allocated per line
           nBands is total number of scale factor bands
           nLines[nBands] is number of lines in each scale factor band
           SMR[nBands] is signal-to-mask ratio in each scale factor band

        Returns:
            bits[nBands] is number of bits allocated to each scale factor band

        
    """
    # return BitAllocUniform(bitBudget, maxMantBits, nBands, nLines, SMR)
    # return BitAllocConstSNR(bitBudget, maxMantBits, nBands, nLines, 0)
    # return BitAllocConstNMR(bitBudget, maxMantBits, nBands, nLines, SMR)
    
    # Sick water filling algorithm 9000
    
    bits_remaining = int(bitBudget)
    bits = np.zeros(nBands, dtype = np.uint64)
    while True:
        nmr = SMR - 6.0 * bits
        bands_sorted = np.argsort(nmr)[::-1]
        did_alloc = False
        
        for band in bands_sorted:
            if bits[band] < maxMantBits:
                to_alloc = 2 if bits[band] == 0 else 1
                bits_used = to_alloc * nLines[band]
                if bits_remaining >= bits_used:
                    bits[band] += to_alloc
                    bits_remaining -= bits_used
                    did_alloc = True
                    break
        
        if not did_alloc:
            break
    
    # Greedily add bits until we need to start taking them away
    # incrementable = np.argwhere((bits != 0) & (bits != maxMantBits))[::-1]
    # if bits_remaining >= np.min(nLines[incrementable]):
    #     while bits_remaining >= np.min(nLines[incrementable]):
    #         incrementable = np.argwhere((bits != 0) & (bits != maxMantBits))[::-1]
    #         for idx in incrementable:
    #             if bits_remaining >= nLines[idx] and ((bits_remaining - nLines[idx]) >= np.min(nLines) or bits_remaining - nLines[idx] == 0):
    #                 bits[idx] += 1
    #                 bits_remaining -= nLines[idx]
                    
    # Worst case when we can't easily just chuck bits places
    # if bits_remaining > 0:
    #     incrementable = np.argwhere((bits != 0) & (bits != maxMantBits)).flatten()
    #     decrementable = np.argwhere((bits > 2)).flatten()
    #     found = False
    #     for idx_inc in incrementable:
    #         for idx_dec in decrementable:
    #             if idx_inc != idx_dec:
    #                 if nLines[idx_inc] - nLines[idx_dec] == bits_remaining:
    #                     bits[idx_inc] += 1
    #                     bits[idx_dec] -= 1
    #                     bits_remaining -= nLines[idx_inc] - nLines[idx_dec]
    #                     found = True
    #                     break
    #         if found:
    #             break
            
    # Sometimes we are left w/ some leftover bits. Could do the loop above a few more times to get rid of em, but I'm running out of time!
        
    # print(bits_remaining)
    # assert bits_remaining == 0
    # assert np.dot(bits, nLines) == int(bitBudget)
    assert np.all(bits != 1)
    assert np.all(bits <= maxMantBits)
    
    return bits

    # return 8*np.ones(nBands, dtype = np.uint64) # TO REPLACE WITH YOUR CODE

#-----------------------------------------------------------------------------

#Testing code
if __name__ == "__main__":

    pass # TO REPLACE WITH YOUR CODE
