"""
codec.py -- The actual encode/decode functions for the perceptual audio codec

-----------------------------------------------------------------------
© 2019-2026 Marina Bosi & Richard E. Goldberg -- All rights reserved
-----------------------------------------------------------------------
"""

import numpy as np  # used for arrays

# used by Encode and Decode
from mdct import MDCT,IMDCT  # fast MDCT implementation (uses numpy FFT)
from quantize import *  # using vectorized versions (to use normal versions, uncomment lines 18,67 below defining vMantissa and vDequantize)

# used only by Encode
from psychoac import CalcSMRs  # calculates SMRs for each scale factor band
from bitalloc import BitAlloc  #allocates bits to scale factor bands given SMRs
from blockswitching import WindowForBlockType, LONG, SHORT, N_SHORT_BLOCKS

def Decode(scaleFactor,bitAlloc,mantissa,overallScaleFactor,codingParams,mdct_pred=None):
    """Reconstitutes a single-channel block of encoded data into a block of
    signed-fraction data based on the parameters in a PACFile object"""

    halfN_long = codingParams.nMDCTLines
    N_long = 2*halfN_long
    halfN_short = codingParams.nMDCTLines_short 
    N_short = 2*halfN_short
    sfBands_short = codingParams.sfBands_short
    # reconstitute the first halfN MDCT lines of this channel from the stored data
    if codingParams.blockType == SHORT:
        # Decode 8 short sub-blocks, window each, and overlap-add into a long-block-sized
        # output buffer. The 'pad' centers the short blocks within the long block window.
        N = N_short
        halfN = halfN_short
        pad = N_long//4 - N_short//4
        overlap_and_add = np.zeros(N_long)
        for i in range(N_SHORT_BLOCKS):
            mdctLine = np.zeros(halfN,dtype=np.float64)
            iMant = 0
            sf, ba, mant, ovs = scaleFactor[i], bitAlloc[i], mantissa[i], overallScaleFactor[i]
            rescaleLevel = 1.*(1<<ovs)
            for iBand in range(sfBands_short.nBands):
                nLines =sfBands_short.nLines[iBand]
                if ba[iBand]:
                    mdctLine[iMant:(iMant+nLines)]=vDequantize(sf[iBand], mant[iMant:(iMant+nLines)],codingParams.nScaleBits, ba[iBand])
                iMant += nLines
            mdctLine /= rescaleLevel  # put overall gain back to original level
            # IMDCT and window the data for this channel
            data = WindowForBlockType(codingParams.blockType, N_long, N_short) * IMDCT(mdctLine, halfN, halfN) # takes in halfN MDCT coeffs
            overlap_and_add[pad + i*halfN_short : pad + i*halfN_short + N_short] += data

        return overlap_and_add
    
    else: 
        rescaleLevel = 1.*(1<<overallScaleFactor)
        N = N_long
        halfN = halfN_long
        mdctLine = np.zeros(halfN,dtype=np.float64)
        iMant = 0
        for iBand in range(codingParams.sfBands.nBands):
            nLines =codingParams.sfBands.nLines[iBand]
            if bitAlloc[iBand]:
                mdctLine[iMant:(iMant+nLines)]=vDequantize(scaleFactor[iBand], mantissa[iMant:(iMant+nLines)],codingParams.nScaleBits, bitAlloc[iBand])
            iMant += nLines
        mdctLine /= rescaleLevel  # put overall gain back to original level
        if mdct_pred is not None:
            mdctLine += mdct_pred
        # IMDCT and window the data for this channel
        data = WindowForBlockType(codingParams.blockType, N_long, N_short) * IMDCT(mdctLine, halfN, halfN)

        # end loop over channels, return reconstituted time samples (pre-overlap-and-add)
        return data


    


def ExpandMantissa(mantissa_compact, bitAlloc, sfBands, halfN):
    """Expand compact mantissa (allocated bands only) to full-length array of size halfN."""
    mantissa_full = np.zeros(halfN, dtype=np.int32)
    iCompact = 0
    iFull = 0
    for iBand in range(sfBands.nBands):
        nLines = sfBands.nLines[iBand]
        if bitAlloc[iBand]:
            mantissa_full[iFull:iFull+nLines] = mantissa_compact[iCompact:iCompact+nLines]
            iCompact += nLines
        iFull += nLines
    return mantissa_full


def Encode(data,codingParams):
    """Encodes a multi-channel block of signed-fraction data based on the parameters in a PACFile object.
    Block type and grouping must be set on codingParams before calling."""
    scaleFactor = []
    bitAlloc = []
    mantissa = []
    overallScaleFactor = []

    # loop over channels and separately encode each one
    for iCh in range(codingParams.nChannels):
        codingParams._masking_signal = codingParams.masking_signals[iCh] if codingParams.masking_signals is not None else None
        codingParams._mdct_pred_correction = codingParams.mdct_pred_corrections[iCh] if codingParams.mdct_pred_corrections is not None else None
        codingParams._block_overhead = codingParams.block_overhead[iCh] if codingParams.block_overhead is not None else 0
        (s,b,m,o) = EncodeSingleChannel(data[iCh],codingParams)
        scaleFactor.append(s)
        bitAlloc.append(b)
        mantissa.append(m)
        overallScaleFactor.append(o)
    # return results bundled over channels

    return (scaleFactor,bitAlloc,mantissa,overallScaleFactor)


def EncodeSingleChannel(data,codingParams):
    """Encodes a single-channel block of signed-fraction data based on the parameters in a PACFile object"""

    # prepare various constants
    halfN_long = codingParams.nMDCTLines
    N_long = 2*halfN_long
    halfN_short = codingParams.nMDCTLines_short
    N_short = 2 * halfN_short
    nScaleBits = codingParams.nScaleBits
    maxMantBits = (1<<codingParams.nMantSizeBits)  # 1 isn't an allowed bit allocation so n size bits counts up to 2^n
    if maxMantBits>16: maxMantBits = 16  # to make sure we don't ever overflow mantissa holders
    sfBands_long = codingParams.sfBands
    sfBands_short = codingParams.sfBands_short
    if codingParams.blockType == SHORT:
        N = N_short
        halfN = halfN_short
        pad = N_long//4 - N_short//4
        group_lens = codingParams.group_lens

        # Phase 1: MDCT all 8 sub-blocks, compute per-sub-block overallScale and SMRs
        sub_mdct_scaled = []
        sub_ovs = []
        sub_smrs = []
        sub_time = []
        for i in range(N_SHORT_BLOCKS):
            timeSamples = data[pad + i*halfN : pad + i*halfN + N]
            sub_time.append(timeSamples)
            mdctTimeSamples = WindowForBlockType(codingParams.blockType, N_long, N_short) * timeSamples
            mdctLines = MDCT(mdctTimeSamples, halfN, halfN)[:halfN]

            maxLine = np.max(np.abs(mdctLines))
            overallScale = ScaleFactor(maxLine, nScaleBits)
            mdctLines_scaled = mdctLines * (1 << overallScale)

            masking_samples = codingParams._masking_signal[pad + i*halfN : pad + i*halfN + N] if codingParams._masking_signal is not None else timeSamples
            SMRs = CalcSMRs(masking_samples, mdctLines_scaled, overallScale, codingParams.sampleRate, sfBands_short)

            sub_ovs.append(overallScale)
            sub_mdct_scaled.append(mdctLines_scaled)
            sub_smrs.append(SMRs)

        # Phase 2: grouped encoding — shared sf/ba per group, per-sub-block mantissa
        all_sf = []
        all_ba = []
        all_mant = []
        all_ovs = []
        sub_idx = 0
        for G in group_lens:
            subs = list(range(sub_idx, sub_idx + G))

            # Pooled bit budget: G sub-blocks share sf/ba overhead once
            # SHORT blocks need extra bits: wider noise bandwidth per MDCT line
            # (~172 Hz vs ~21 Hz for LONG) makes quantization noise harder to mask
            bitBudget = codingParams.targetBitsPerSample * halfN * G * 2.0
            bitBudget -= nScaleBits * G                              # per-sub-block overallScale
            bitBudget -= nScaleBits * sfBands_short.nBands           # ONE set of band scale factors
            bitBudget -= codingParams.nMantSizeBits * sfBands_short.nBands  # ONE set of bit allocs
            bitBudget -= codingParams._block_overhead * G / N_SHORT_BLOCKS  # proportional share of bitstream overhead
            bitBudget = max(int(bitBudget), 0)

            # Inflate bit budget to exploit entropy coding compression
            bitBudget = max(int(bitBudget * codingParams._entropy_inflation_short), 0)

            # Group SMR = max across sub-blocks per band
            combined_SMRs = np.max([sub_smrs[i] for i in subs], axis=0)

            # BitAlloc: nLines * G because mantissa bits are spent on G sub-blocks
            bitAlloc = BitAlloc(bitBudget, maxMantBits, sfBands_short.nBands,
                                sfBands_short.nLines * G, combined_SMRs)

            # Shared per-band scale factor: max across sub-blocks in group
            scaleFactor = np.empty(sfBands_short.nBands, dtype=np.uint64)
            for iBand in range(sfBands_short.nBands):
                lo = sfBands_short.lowerLine[iBand]
                hi = sfBands_short.upperLine[iBand] + 1
                maxBand = max(np.max(np.abs(sub_mdct_scaled[i][lo:hi])) for i in subs)
                scaleFactor[iBand] = ScaleFactor(maxBand, nScaleBits, bitAlloc[iBand])

            # Quantize each sub-block using shared sf/ba
            for i in subs:
                nMant = halfN
                for iBand in range(sfBands_short.nBands):
                    if not bitAlloc[iBand]: nMant -= sfBands_short.nLines[iBand]
                mantissa = np.empty(nMant, dtype=np.int32)
                iMant = 0
                for iBand in range(sfBands_short.nBands):
                    lowLine = sfBands_short.lowerLine[iBand]
                    highLine = sfBands_short.upperLine[iBand] + 1
                    nLines = sfBands_short.nLines[iBand]
                    if bitAlloc[iBand]:
                        mantissa[iMant:iMant+nLines] = vMantissa(
                            sub_mdct_scaled[i][lowLine:highLine],
                            scaleFactor[iBand], nScaleBits, bitAlloc[iBand])
                        iMant += nLines

                all_sf.append(scaleFactor)
                all_ba.append(bitAlloc)
                all_mant.append(mantissa)
                all_ovs.append(sub_ovs[i])

            sub_idx += G

        return (all_sf, all_ba, all_mant, all_ovs)
    
    else: 
        N = N_long
        halfN = halfN_long
        # compute target mantissa bit budget for this block of halfN MDCT mantissas
        bitBudget = codingParams.targetBitsPerSample * halfN  # this is overall target bit rate
        bitBudget -=  nScaleBits*(sfBands_long.nBands +1)  # less scale factor bits (including overall scale factor)
        bitBudget -= codingParams.nMantSizeBits*sfBands_long.nBands  # less mantissa bit allocation bits
        bitBudget -= codingParams._block_overhead  # bitstream overhead (block type, pred, M/S, nBytes field)
        bitBudget = max(bitBudget, 0)

        # Inflate bit budget to exploit entropy coding compression
        bitBudget = max(int(bitBudget * codingParams._entropy_inflation), 0)

        # window data for side chain FFT and also window and compute MDCT
        timeSamples = data
        mdctTimeSamples = WindowForBlockType(codingParams.blockType, N_long, N_short) * timeSamples
        mdctLines = MDCT(mdctTimeSamples, halfN, halfN)[:halfN]

        # per-band prediction correction: cancel subtraction in non-enabled bands
        # mdctLines = (X - alpha_q*P) + alpha_q*P*(1-enable_mask) = X - alpha_q*P*enable_mask
        if codingParams._mdct_pred_correction is not None:
            mdctLines = mdctLines + codingParams._mdct_pred_correction

        # compute overall scale factor for this block and boost mdctLines using it
        maxLine = np.max( np.abs(mdctLines) )
        overallScale = ScaleFactor(maxLine,nScaleBits)  #leading zeroes don't depend on nMantBits
        mdctLines *= (1<<overallScale)

        # compute the mantissa bit allocations
        # compute SMRs in side chain FFT (use original signal for masking if available)
        masking_samples = codingParams._masking_signal if codingParams._masking_signal is not None else timeSamples
        SMRs = CalcSMRs(masking_samples, mdctLines, overallScale, codingParams.sampleRate, sfBands_long)
        # perform bit allocation using SMR results
        bitAlloc = BitAlloc(bitBudget, maxMantBits, sfBands_long.nBands, sfBands_long.nLines, SMRs)

        # given the bit allocations, quantize the mdct lines in each band
        scaleFactor = np.empty(sfBands_long.nBands,dtype=np.uint64)
        nMant=halfN
        for iBand in range(sfBands_long.nBands):
            if not bitAlloc[iBand]: nMant-= sfBands_long.nLines[iBand]  # account for mantissas not being transmitted
        mantissa=np.empty(nMant,dtype=np.int32)
        iMant=0
        for iBand in range(sfBands_long.nBands):
            lowLine = sfBands_long.lowerLine[iBand]
            highLine = sfBands_long.upperLine[iBand] + 1  # extra value is because slices don't include last value
            nLines= sfBands_long.nLines[iBand]
            scaleLine = np.max(np.abs( mdctLines[lowLine:highLine] ) )
            scaleFactor[iBand] = ScaleFactor(scaleLine, nScaleBits, bitAlloc[iBand])
            if bitAlloc[iBand]:
                mantissa[iMant:iMant+nLines] = vMantissa(mdctLines[lowLine:highLine],scaleFactor[iBand], nScaleBits, bitAlloc[iBand])
                iMant += nLines
        # end of loop over scale factor bands

        # return results
        return (scaleFactor, bitAlloc, mantissa, overallScale)
