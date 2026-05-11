"""
codec.py -- The actual encode/decode functions for the perceptual audio codec

-----------------------------------------------------------------------
© 2019-2026 Marina Bosi & Richard E. Goldberg -- All rights reserved
-----------------------------------------------------------------------
"""

import numpy as np  # used for arrays
from features import SHORT_BLOCK_BITBOOST, AC2A_BLOCK_SWITCHING
from blockswitching import LONG, START, SHORT, STOP, MEDIUM, DesignSFBands
from psychoac import ScaleFactorBands, AssignMDCTLinesFromFreqLimits

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

    if codingParams.blockType == SHORT and AC2A_BLOCK_SWITCHING:
        # AC-2A: single 256-sample SHORT sub-block
        rescaleLevel = 1.*(1<<overallScaleFactor)
        mdctLine = np.zeros(halfN_short, dtype=np.float64)
        iMant = 0
        for iBand in range(sfBands_short.nBands):
            nLines = sfBands_short.nLines[iBand]
            if bitAlloc[iBand]:
                mdctLine[iMant:iMant+nLines] = vDequantize(
                    scaleFactor[iBand], mantissa[iMant:iMant+nLines],
                    codingParams.nScaleBits, bitAlloc[iBand])
            iMant += nLines
        mdctLine /= rescaleLevel
        return WindowForBlockType(SHORT, N_long, N_short) * IMDCT(mdctLine, halfN_short, halfN_short)

    elif codingParams.blockType == MEDIUM and AC2A_BLOCK_SWITCHING:
        # Cascade intermediate block: IMDCT with variable (a, b) overlap sizes
        ca = codingParams.cascade_a   # left overlap
        cb = codingParams.cascade_b   # right overlap
        halfN_med = (ca + cb) // 2
        sfBands_med = DesignSFBands(halfN_med, codingParams.sampleRate)
        rescaleLevel = 1.*(1<<overallScaleFactor)
        mdctLine = np.zeros(halfN_med, dtype=np.float64)
        iMant = 0
        for iBand in range(sfBands_med.nBands):
            nLines = sfBands_med.nLines[iBand]
            if bitAlloc[iBand]:
                mdctLine[iMant:iMant+nLines] = vDequantize(
                    scaleFactor[iBand], mantissa[iMant:iMant+nLines],
                    codingParams.nScaleBits, bitAlloc[iBand])
            iMant += nLines
        mdctLine /= rescaleLevel
        return WindowForBlockType(MEDIUM, N_long, N_short,
                                  cascade_a=ca, cascade_b=cb) * IMDCT(mdctLine, ca, cb)

    elif codingParams.blockType in (START, STOP) and AC2A_BLOCK_SWITCHING:
        # AC-2A: asymmetric transition block (b may vary for cascade START*)
        ca = getattr(codingParams, 'cascade_a', halfN_long)
        cb = getattr(codingParams, 'cascade_b', halfN_short)
        halfN_trans = (ca + cb) // 2
        sfBands_trans = DesignSFBands(halfN_trans, codingParams.sampleRate) \
            if halfN_trans != getattr(codingParams, 'nMDCTLines_trans', halfN_trans) \
            else codingParams.sfBands_trans
        rescaleLevel = 1.*(1<<overallScaleFactor)
        mdctLine = np.zeros(halfN_trans, dtype=np.float64)
        iMant = 0
        for iBand in range(sfBands_trans.nBands):
            nLines = sfBands_trans.nLines[iBand]
            if bitAlloc[iBand]:
                mdctLine[iMant:iMant+nLines] = vDequantize(
                    scaleFactor[iBand], mantissa[iMant:iMant+nLines],
                    codingParams.nScaleBits, bitAlloc[iBand])
            iMant += nLines
        mdctLine /= rescaleLevel
        if mdct_pred is not None:
            mdctLine += mdct_pred
        if codingParams.blockType == START:
            data = WindowForBlockType(START, N_long, N_short,
                                      cascade_b=cb) * IMDCT(mdctLine, ca, cb)
        else:
            data = WindowForBlockType(STOP, N_long, N_short,
                                      cascade_a=ca) * IMDCT(mdctLine, ca, cb)
        return data

    elif codingParams.blockType == SHORT:
        # Edler: 8 sub-blocks assembled into N_long output
        N = N_short
        halfN = halfN_short
        pad = N_long // 4 - N_short // 4
        overlap_and_add = np.zeros(N_long)
        for i in range(N_SHORT_BLOCKS):
            mdctLine = np.zeros(halfN, dtype=np.float64)
            iMant = 0
            sf, ba, mant, ovs = scaleFactor[i], bitAlloc[i], mantissa[i], overallScaleFactor[i]
            rescaleLevel = 1.*(1<<ovs)
            for iBand in range(sfBands_short.nBands):
                nLines = sfBands_short.nLines[iBand]
                if ba[iBand]:
                    mdctLine[iMant:iMant+nLines] = vDequantize(
                        sf[iBand], mant[iMant:iMant+nLines], codingParams.nScaleBits, ba[iBand])
                iMant += nLines
            mdctLine /= rescaleLevel
            data = WindowForBlockType(SHORT, N_long, N_short) * IMDCT(mdctLine, halfN, halfN)
            overlap_and_add[pad + i*halfN_short : pad + i*halfN_short + N_short] += data
        return overlap_and_add

    else:
        # LONG (and non-AC2A START/STOP via Edler windows)
        rescaleLevel = 1.*(1<<overallScaleFactor)
        halfN = halfN_long
        mdctLine = np.zeros(halfN, dtype=np.float64)
        iMant = 0
        for iBand in range(codingParams.sfBands.nBands):
            nLines = codingParams.sfBands.nLines[iBand]
            if bitAlloc[iBand]:
                mdctLine[iMant:iMant+nLines] = vDequantize(
                    scaleFactor[iBand], mantissa[iMant:iMant+nLines],
                    codingParams.nScaleBits, bitAlloc[iBand])
            iMant += nLines
        mdctLine /= rescaleLevel
        if mdct_pred is not None:
            mdctLine += mdct_pred
        data = WindowForBlockType(codingParams.blockType, N_long, N_short) * IMDCT(mdctLine, halfN, halfN)
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
        (s,b,m,o,_) = EncodeSingleChannel(data[iCh],codingParams)
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
    if codingParams.blockType == SHORT and AC2A_BLOCK_SWITCHING:
        # AC-2A: single 256-sample SHORT sub-block
        halfN = halfN_short
        N = N_short
        sfBands = sfBands_short
        nScaleBits = codingParams.nScaleBits

        timeSamples = data  # 256 samples
        mdctTimeSamples = WindowForBlockType(SHORT, N_long, N_short) * timeSamples
        mdctLines = MDCT(mdctTimeSamples, halfN, halfN)[:halfN]

        maxLine = np.max(np.abs(mdctLines))
        overallScale = ScaleFactor(maxLine, nScaleBits)
        mdctLines_scaled = mdctLines * (1 << overallScale)

        masking_samples = codingParams._masking_signal if codingParams._masking_signal is not None else timeSamples
        SMRs = CalcSMRs(masking_samples, mdctLines_scaled, overallScale, codingParams.sampleRate, sfBands)

        _short_boost = 2.0 if SHORT_BLOCK_BITBOOST else 1.0
        bitBudget = int(codingParams.targetBitsPerSample * halfN * _short_boost)
        bitBudget -= nScaleBits * (sfBands.nBands + 1)
        bitBudget -= codingParams.nMantSizeBits * sfBands.nBands
        bitBudget -= codingParams._block_overhead
        bitBudget = max(int(bitBudget * codingParams._entropy_inflation_short), 0)

        bitAlloc_s = BitAlloc(bitBudget, maxMantBits, sfBands.nBands, sfBands.nLines, SMRs)
        scaleFactor_s = np.empty(sfBands.nBands, dtype=np.uint64)
        nMant = halfN
        for iBand in range(sfBands.nBands):
            if not bitAlloc_s[iBand]: nMant -= sfBands.nLines[iBand]
        mantissa_s = np.empty(nMant, dtype=np.int32)
        iMant = 0
        for iBand in range(sfBands.nBands):
            lowLine = int(sfBands.lowerLine[iBand])
            highLine = int(sfBands.upperLine[iBand]) + 1
            nLines = int(sfBands.nLines[iBand])
            band_lines = mdctLines_scaled[lowLine:highLine]
            scaleLine = np.max(np.abs(band_lines))
            scaleFactor_s[iBand] = ScaleFactor(scaleLine, nScaleBits, bitAlloc_s[iBand])
            if bitAlloc_s[iBand]:
                mantissa_s[iMant:iMant+nLines] = vMantissa(
                    band_lines, scaleFactor_s[iBand], nScaleBits, bitAlloc_s[iBand])
                iMant += nLines
        return (scaleFactor_s, bitAlloc_s, mantissa_s, overallScale, None)

    elif codingParams.blockType == MEDIUM and AC2A_BLOCK_SWITCHING:
        # Cascade intermediate block
        ca = codingParams.cascade_a
        cb = codingParams.cascade_b
        halfN = (ca + cb) // 2
        sfBands = DesignSFBands(halfN, codingParams.sampleRate)

        timeSamples = data
        mdctTimeSamples = WindowForBlockType(MEDIUM, N_long, N_short,
                                             cascade_a=ca, cascade_b=cb) * timeSamples
        mdctLines = MDCT(mdctTimeSamples, ca, cb)[:halfN]

        maxLine = np.max(np.abs(mdctLines))
        overallScale = ScaleFactor(maxLine, nScaleBits)
        mdctLines *= (1 << overallScale)

        masking_samples = codingParams._masking_signal if codingParams._masking_signal is not None else timeSamples
        SMRs = CalcSMRs(masking_samples, mdctLines, overallScale, codingParams.sampleRate, sfBands)

        bitBudget = codingParams.targetBitsPerSample * halfN
        bitBudget -= nScaleBits * (sfBands.nBands + 1)
        bitBudget -= codingParams.nMantSizeBits * sfBands.nBands
        bitBudget -= codingParams._block_overhead
        bitBudget = max(int(bitBudget * codingParams._entropy_inflation), 0)

        bitAlloc_m = BitAlloc(bitBudget, maxMantBits, sfBands.nBands, sfBands.nLines, SMRs)
        scaleFactor_m = np.empty(sfBands.nBands, dtype=np.uint64)
        nMant = halfN
        for iBand in range(sfBands.nBands):
            if not bitAlloc_m[iBand]: nMant -= sfBands.nLines[iBand]
        mantissa_m = np.empty(nMant, dtype=np.int32)
        iMant = 0
        for iBand in range(sfBands.nBands):
            lowLine = sfBands.lowerLine[iBand]
            highLine = sfBands.upperLine[iBand] + 1
            nLines = sfBands.nLines[iBand]
            band_lines = mdctLines[lowLine:highLine]
            scaleLine = np.max(np.abs(band_lines))
            scaleFactor_m[iBand] = ScaleFactor(scaleLine, nScaleBits, bitAlloc_m[iBand])
            if bitAlloc_m[iBand]:
                mantissa_m[iMant:iMant+nLines] = vMantissa(
                    band_lines, scaleFactor_m[iBand], nScaleBits, bitAlloc_m[iBand])
                iMant += nLines
        return (scaleFactor_m, bitAlloc_m, mantissa_m, overallScale, None)

    elif codingParams.blockType in (START, STOP) and AC2A_BLOCK_SWITCHING:
        # AC-2A: asymmetric transition block (b variable for cascade START*)
        ca = getattr(codingParams, 'cascade_a', halfN_long)
        cb = getattr(codingParams, 'cascade_b', halfN_short)
        halfN_trans = (ca + cb) // 2
        halfN = halfN_trans
        sfBands = DesignSFBands(halfN_trans, codingParams.sampleRate) \
            if halfN_trans != getattr(codingParams, 'nMDCTLines_trans', halfN_trans) \
            else codingParams.sfBands_trans

        timeSamples = data
        mdctTimeSamples = WindowForBlockType(codingParams.blockType, N_long, N_short,
                                             cascade_a=ca, cascade_b=cb) * timeSamples
        mdctLines = MDCT(mdctTimeSamples, ca, cb)[:halfN_trans]

        if codingParams._mdct_pred_correction is not None:
            mdctLines = mdctLines + codingParams._mdct_pred_correction

        maxLine = np.max(np.abs(mdctLines))
        overallScale = ScaleFactor(maxLine, nScaleBits)
        mdctLines *= (1 << overallScale)

        masking_samples = codingParams._masking_signal if codingParams._masking_signal is not None else timeSamples
        SMRs = CalcSMRs(masking_samples, mdctLines, overallScale, codingParams.sampleRate, sfBands)

        bitBudget = codingParams.targetBitsPerSample * halfN
        bitBudget -= nScaleBits * (sfBands.nBands + 1)
        bitBudget -= codingParams.nMantSizeBits * sfBands.nBands
        bitBudget -= codingParams._block_overhead
        bitBudget = max(int(bitBudget * codingParams._entropy_inflation), 0)

        bitAlloc_t = BitAlloc(bitBudget, maxMantBits, sfBands.nBands, sfBands.nLines, SMRs)
        scaleFactor_t = np.empty(sfBands.nBands, dtype=np.uint64)
        nMant = halfN
        for iBand in range(sfBands.nBands):
            if not bitAlloc_t[iBand]: nMant -= sfBands.nLines[iBand]
        mantissa_t = np.empty(nMant, dtype=np.int32)
        iMant = 0
        for iBand in range(sfBands.nBands):
            lowLine = sfBands.lowerLine[iBand]
            highLine = sfBands.upperLine[iBand] + 1
            nLines = sfBands.nLines[iBand]
            band_lines = mdctLines[lowLine:highLine]
            scaleLine = np.max(np.abs(band_lines))
            scaleFactor_t[iBand] = ScaleFactor(scaleLine, nScaleBits, bitAlloc_t[iBand])
            if bitAlloc_t[iBand]:
                mantissa_t[iMant:iMant+nLines] = vMantissa(
                    band_lines, scaleFactor_t[iBand], nScaleBits, bitAlloc_t[iBand])
                iMant += nLines
        return (scaleFactor_t, bitAlloc_t, mantissa_t, overallScale, None)

    elif codingParams.blockType == SHORT:
        # Edler: 8-sub-block group encoding
        N = N_short
        halfN = halfN_short
        pad = N_long // 4 - N_short // 4
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
            _short_boost = 2.0 if SHORT_BLOCK_BITBOOST else 1.0
            bitBudget = codingParams.targetBitsPerSample * halfN * G * _short_boost
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

            # Precompute allocated-band metadata once (shared across all sub-blocks in group)
            alloc_bands = [
                (iBand, int(sfBands_short.lowerLine[iBand]),
                 int(sfBands_short.upperLine[iBand]) + 1,
                 int(sfBands_short.nLines[iBand]))
                for iBand in range(sfBands_short.nBands) if bitAlloc[iBand]
            ]
            nMant = sum(nLines for _, _, _, nLines in alloc_bands)

            # Quantize each sub-block using shared sf/ba
            for i in subs:
                mantissa = np.empty(nMant, dtype=np.int32)
                iMant = 0
                for iBand, lowLine, highLine, nLines in alloc_bands:
                    mantissa[iMant:iMant+nLines] = vMantissa(
                        sub_mdct_scaled[i][lowLine:highLine],
                        scaleFactor[iBand], nScaleBits, bitAlloc[iBand])
                    iMant += nLines

                all_sf.append(scaleFactor)
                all_ba.append(bitAlloc)
                all_mant.append(mantissa)
                all_ovs.append(sub_ovs[i])

            sub_idx += G

        return (all_sf, all_ba, all_mant, all_ovs, None)
    
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
            band_lines = mdctLines[lowLine:highLine]
            scaleLine = np.max(np.abs(band_lines))
            scaleFactor[iBand] = ScaleFactor(scaleLine, nScaleBits, bitAlloc[iBand])
            if bitAlloc[iBand]:
                mantissa[iMant:iMant+nLines] = vMantissa(band_lines, scaleFactor[iBand], nScaleBits, bitAlloc[iBand])
                iMant += nLines
        # end of loop over scale factor bands

        # return results
        return (scaleFactor, bitAlloc, mantissa, overallScale, None)
