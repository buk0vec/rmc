"""
codec.py -- The actual encode/decode functions for the perceptual audio codec

-----------------------------------------------------------------------
© 2019-2026 Marina Bosi & Richard E. Goldberg -- All rights reserved
-----------------------------------------------------------------------
"""

import numpy as np  # used for arrays

# used by Encode and Decode
from window import SineWindow  # current window used for MDCT -- implement KB-derived?
from mdct import MDCT,IMDCT  # fast MDCT implementation (uses numpy FFT)
from quantize import *  # using vectorized versions (to use normal versions, uncomment lines 18,67 below defining vMantissa and vDequantize)

# used only by Encode
from psychoac import CalcSMRs  # calculates SMRs for each scale factor band
from bitalloc import BitAlloc  #allocates bits to scale factor bands given SMRs
from blockswitching import SelectBlockType, WindowForBlockType, ShortBlockSFBands, LONG, SHORT, N_SHORT_BLOCKS

def Decode(scaleFactor,bitAlloc,mantissa,overallScaleFactor,codingParams,mdct_pred=None):
    """Reconstitutes a single-channel block of encoded data into a block of
    signed-fraction data based on the parameters in a PACFile object"""

    

    halfN_long = codingParams.nMDCTLines
    N_long = 2*halfN_long
    halfN_short = codingParams.nMDCTLines_short 
    N_short = 2*halfN_short
    sfBands_short = ShortBlockSFBands(codingParams.nMDCTLines_short, codingParams.sampleRate)
    # vectorizing the Dequantize function call
#    vDequantize = np.vectorize(Dequantize)

    # reconstitute the first halfN MDCT lines of this channel from the stored data
    if codingParams.blockType == SHORT:
        
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
            if mdct_pred is not None:  # mdct_pred is a list of per-sub-block arrays for SHORT
                mdctLine += mdct_pred[i]
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
    """Encodes a multi-channel block of signed-fraction data based on the parameters in a PACFile object"""
    scaleFactor = []
    bitAlloc = []
    mantissa = []
    overallScaleFactor = []
    
    block_idx = getattr(codingParams, 'blockIndex', 0)
    transient_map = getattr(codingParams, 'transientBlocks', {})
    k_attack = transient_map.get(block_idx, -1)
    codingParams.blockType = SelectBlockType(k_attack, codingParams.prevBlockType)
    codingParams.prevBlockType = codingParams.blockType
    codingParams.blockIndex = block_idx + 1
    # loop over channels and separately encode each one
    masking_signals = getattr(codingParams, 'masking_signals', None)
    mdct_corrections = getattr(codingParams, 'mdct_pred_corrections', None)
    pred_bits_freed = getattr(codingParams, 'pred_bits_freed', None)
    for iCh in range(codingParams.nChannels):
        codingParams._masking_signal = masking_signals[iCh] if masking_signals is not None else None
        codingParams._mdct_pred_correction = mdct_corrections[iCh] if mdct_corrections is not None else None
        codingParams._pred_bits_freed = pred_bits_freed[iCh] if pred_bits_freed is not None else 0.0
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
    sfBands_short = ShortBlockSFBands(codingParams.nMDCTLines_short, codingParams.sampleRate)
    # vectorizing the Mantissa function call
    #    vMantissa = np.vectorize(Mantissa)
    if codingParams.blockType == SHORT:
        N = N_short
        halfN = halfN_short
        all_sf = []
        all_ba = []
        all_mant = []
        all_ovs = []
        for i in range(N_SHORT_BLOCKS):
            # compute target mantissa bit budget for this block of halfN MDCT mantissas
            bitBudget = codingParams.targetBitsPerSample * halfN  # this is overall target bit rate
            bitBudget -=  nScaleBits*(sfBands_short.nBands +1)  # less scale factor bits (including overall scale factor)
            bitBudget -= codingParams.nMantSizeBits*sfBands_short.nBands  # less mantissa bit allocation bits


            # window data for side chain FFT and also window and compute MDCT
            pad = N_long//4 - N_short//4
            timeSamples = data[pad + i*halfN : pad + i*halfN + N]
            mdctTimeSamples = WindowForBlockType(codingParams.blockType, N_long, N_short) *timeSamples
            mdctLines = MDCT(mdctTimeSamples, halfN, halfN)[:halfN]

            # compute overall scale factor for this block and boost mdctLines using it
            maxLine = np.max( np.abs(mdctLines) )
            overallScale = ScaleFactor(maxLine,nScaleBits)  #leading zeroes don't depend on nMantBits
            mdctLines *= (1<<overallScale)

            # compute the mantissa bit allocations
            # compute SMRs in side chain FFT (use original signal for masking if available)
            masking_signal = getattr(codingParams, '_masking_signal', None)
            masking_samples = masking_signal[pad + i*halfN : pad + i*halfN + N] if masking_signal is not None else timeSamples
            SMRs = CalcSMRs(masking_samples, mdctLines, overallScale, codingParams.sampleRate, sfBands_short)
            # perform bit allocation using SMR results
            bitAlloc = BitAlloc(bitBudget, maxMantBits, sfBands_short.nBands, sfBands_short.nLines, SMRs)

            # given the bit allocations, quantize the mdct lines in each band
            scaleFactor = np.empty(sfBands_short.nBands,dtype=np.uint64)
            nMant=halfN
            for iBand in range(sfBands_short.nBands):
                if not bitAlloc[iBand]: nMant-= sfBands_short.nLines[iBand]  # account for mantissas not being transmitted
            mantissa=np.empty(nMant,dtype=np.int32)
            iMant=0
            for iBand in range(sfBands_short.nBands):
                lowLine = sfBands_short.lowerLine[iBand]
                highLine = sfBands_short.upperLine[iBand] + 1  # extra value is because slices don't include last value
                nLines= sfBands_short.nLines[iBand]
                scaleLine = np.max(np.abs( mdctLines[lowLine:highLine] ) )
                scaleFactor[iBand] = ScaleFactor(scaleLine, nScaleBits, bitAlloc[iBand])
                if bitAlloc[iBand]:
                    mantissa[iMant:iMant+nLines] = vMantissa(mdctLines[lowLine:highLine],scaleFactor[iBand], nScaleBits, bitAlloc[iBand])
                    iMant += nLines
            # end of loop over scale factor bands

            # return results
            all_sf.append(scaleFactor)
            all_ba.append(bitAlloc)
            all_mant.append(mantissa)
            all_ovs.append(overallScale)
            
        return (all_sf, all_ba, all_mant, all_ovs)
    
    else: 
        N = N_long
        halfN = halfN_long
        # compute target mantissa bit budget for this block of halfN MDCT mantissas
        bitBudget = codingParams.targetBitsPerSample * halfN  # this is overall target bit rate
        bitBudget -=  nScaleBits*(sfBands_long.nBands +1)  # less scale factor bits (including overall scale factor)
        bitBudget -= codingParams.nMantSizeBits*sfBands_long.nBands  # less mantissa bit allocation bits
        # prediction savings: each dB of residual reduction frees 1/6 bit per line
        bitBudget -= int(getattr(codingParams, '_pred_bits_freed', 0.0))
        bitBudget = max(bitBudget, 0)


        # window data for side chain FFT and also window and compute MDCT
        timeSamples = data
        mdctTimeSamples = WindowForBlockType(codingParams.blockType, N_long, N_short) * timeSamples
        mdctLines = MDCT(mdctTimeSamples, halfN, halfN)[:halfN]

        # per-band prediction correction: cancel subtraction in non-enabled bands
        # mdctLines = (X - alpha_q*P) + alpha_q*P*(1-enable_mask) = X - alpha_q*P*enable_mask
        correction = getattr(codingParams, '_mdct_pred_correction', None)
        if correction is not None:
            mdctLines = mdctLines + correction

        # compute overall scale factor for this block and boost mdctLines using it
        maxLine = np.max( np.abs(mdctLines) )
        overallScale = ScaleFactor(maxLine,nScaleBits)  #leading zeroes don't depend on nMantBits
        mdctLines *= (1<<overallScale)

        # compute the mantissa bit allocations
        # compute SMRs in side chain FFT (use original signal for masking if available)
        masking_signal = getattr(codingParams, '_masking_signal', None)
        masking_samples = masking_signal if masking_signal is not None else timeSamples
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

    



