"""
codec.py -- The actual encode/decode functions for the perceptual audio codec

-----------------------------------------------------------------------
© 2019-2026 Marina Bosi & Richard E. Goldberg -- All rights reserved
-----------------------------------------------------------------------
"""

import numpy as np  # used for arrays

from bitalloc import BitAlloc  # allocates bits to scale factor bands given SMRs
from blockswitching import LONG, N_SHORT_BLOCKS, SHORT, START, STOP, MEDIUM, DesignSFBands, WindowForBlockType

# used by Encode and Decode
from mdct import IMDCT, MDCT  # fast MDCT implementation (uses numpy FFT)

# used only by Encode
from psychoac import CalcSMRs  # calculates SMRs for each scale factor band
from quantize import *  # using vectorized versions (to use normal versions, uncomment lines 18,67 below defining vMantissa and vDequantize)


def Decode(
    scaleFactor, bitAlloc, mantissa, overallScaleFactor, codingParams, mdct_pred=None,
):
    """Reconstitutes a single-channel block of encoded data into a block of
    signed-fraction data based on the parameters in a PACFile object"""
    halfN_long = codingParams.nMDCTLines
    N_long = 2 * halfN_long
    halfN_short = codingParams.nMDCTLines_short
    N_short = 2 * halfN_short
    sfBands_short = codingParams.sfBands_short

    if codingParams.blockType == SHORT:
        # AC-2A: single 128-sample SHORT sub-block
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

    elif codingParams.blockType == MEDIUM:
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

    elif codingParams.blockType in (START, STOP):
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

    else:
        rescaleLevel = 1.0 * (1 << overallScaleFactor)
        halfN = halfN_long
        mdctLine = np.zeros(halfN, dtype=np.float64)
        iMant = 0
        for iBand in range(codingParams.sfBands.nBands):
            nLines = codingParams.sfBands.nLines[iBand]
            if bitAlloc[iBand]:
                mdctLine[iMant : (iMant + nLines)] = vDequantize(
                    scaleFactor[iBand],
                    mantissa[iMant : (iMant + nLines)],
                    codingParams.nScaleBits,
                    bitAlloc[iBand],
                )
            iMant += nLines
        mdctLine /= rescaleLevel
        if mdct_pred is not None:
            mdctLine += mdct_pred
        data = WindowForBlockType(codingParams.blockType, N_long, N_short) * IMDCT(
            mdctLine, halfN, halfN
        )
        return data


def ExpandMantissa(mantissa_compact, bitAlloc, sfBands, halfN):
    """Expand compact mantissa (allocated bands only) to full-length array of size halfN."""
    mantissa_full = np.zeros(halfN, dtype=np.int32)
    iCompact = 0
    iFull = 0
    for iBand in range(sfBands.nBands):
        nLines = sfBands.nLines[iBand]
        if bitAlloc[iBand]:
            mantissa_full[iFull : iFull + nLines] = mantissa_compact[
                iCompact : iCompact + nLines
            ]
            iCompact += nLines
        iFull += nLines
    return mantissa_full


def Encode(data, codingParams):
    """Encodes a multi-channel block of signed-fraction data based on the parameters in a PACFile object.
    Block type and grouping must be set on codingParams before calling."""
    scaleFactor = []
    bitAlloc = []
    mantissa = []
    overallScaleFactor = []
    for iCh in range(codingParams.nChannels):
        codingParams._masking_signal = (
            codingParams.masking_signals[iCh]
            if codingParams.masking_signals is not None
            else None
        )
        codingParams._mdct_pred_correction = (
            codingParams.mdct_pred_corrections[iCh]
            if codingParams.mdct_pred_corrections is not None
            else None
        )
        codingParams._block_overhead = (
            codingParams.block_overhead[iCh]
            if codingParams.block_overhead is not None
            else 0
        )
        codingParams._pool_draw = codingParams._pool_draws[iCh]
        (s, b, m, o) = EncodeSingleChannel(data[iCh], codingParams)
        scaleFactor.append(s)
        bitAlloc.append(b)
        mantissa.append(m)
        overallScaleFactor.append(o)
    return (scaleFactor, bitAlloc, mantissa, overallScaleFactor)


def EncodeSingleChannel(data, codingParams):
    """Encodes a single-channel block of signed-fraction data based on the parameters in a PACFile object"""

    halfN_long = codingParams.nMDCTLines
    N_long = 2 * halfN_long
    halfN_short = codingParams.nMDCTLines_short
    N_short = 2 * halfN_short
    nScaleBits = codingParams.nScaleBits
    maxMantBits = (
        1 << codingParams.nMantSizeBits
    )  # 1 isn't an allowed bit allocation so n size bits counts up to 2^n
    if maxMantBits > 16:
        maxMantBits = 16  # to make sure we don't ever overflow mantissa holders
    sfBands_long = codingParams.sfBands
    sfBands_short = codingParams.sfBands_short

    if codingParams.blockType == SHORT:
        # AC-2A: single 128-sample SHORT sub-block
        halfN = halfN_short
        N = N_short
        sfBands = sfBands_short

        timeSamples = data  # 128 samples
        mdctTimeSamples = WindowForBlockType(SHORT, N_long, N_short) * timeSamples
        mdctLines = MDCT(mdctTimeSamples, halfN, halfN)[:halfN]

        maxLine = np.max(np.abs(mdctLines))
        overallScale = ScaleFactor(maxLine, nScaleBits)
        mdctLines_scaled = mdctLines * (1 << overallScale)

        masking_samples = codingParams._masking_signal if codingParams._masking_signal is not None else timeSamples
        SMRs = CalcSMRs(masking_samples, mdctLines_scaled, overallScale, codingParams.sampleRate, sfBands)

        bitBudget = int(codingParams.targetBitsPerSample * halfN)
        bitBudget += codingParams._pool_draw
        bitBudget -= nScaleBits * (sfBands.nBands + 1)
        bitBudget -= codingParams.nMantSizeBits * sfBands.nBands
        bitBudget -= codingParams._block_overhead
        bitBudget = max(bitBudget, 0)

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
        return (scaleFactor_s, bitAlloc_s, mantissa_s, overallScale)

    elif codingParams.blockType == MEDIUM:
        # Cascade intermediate block
        ca = codingParams.cascade_a
        cb = codingParams.cascade_b
        halfN = (ca + cb) // 2
        sfBands = DesignSFBands(halfN, codingParams.sampleRate)

        timeSamples = data
        mdctTimeSamples = WindowForBlockType(MEDIUM, N_long, N_short,
                                             cascade_a=ca, cascade_b=cb) * timeSamples
        mdctLines = MDCT(mdctTimeSamples, ca, cb)[:halfN]

        if codingParams._mdct_pred_correction is not None:
            mdctLines = mdctLines + codingParams._mdct_pred_correction

        maxLine = np.max(np.abs(mdctLines))
        overallScale = ScaleFactor(maxLine, nScaleBits)
        mdctLines *= (1 << overallScale)

        masking_samples = codingParams._masking_signal if codingParams._masking_signal is not None else timeSamples
        SMRs = CalcSMRs(masking_samples, mdctLines, overallScale, codingParams.sampleRate, sfBands)

        bitBudget = codingParams.targetBitsPerSample * halfN
        bitBudget += codingParams._pool_draw
        bitBudget -= nScaleBits * (sfBands.nBands + 1)
        bitBudget -= codingParams.nMantSizeBits * sfBands.nBands
        bitBudget -= codingParams._block_overhead
        bitBudget = max(int(bitBudget), 0)

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
        return (scaleFactor_m, bitAlloc_m, mantissa_m, overallScale)

    elif codingParams.blockType in (START, STOP):
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
        bitBudget += codingParams._pool_draw
        bitBudget -= nScaleBits * (sfBands.nBands + 1)
        bitBudget -= codingParams.nMantSizeBits * sfBands.nBands
        bitBudget -= codingParams._block_overhead
        bitBudget = max(int(bitBudget), 0)

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
        return (scaleFactor_t, bitAlloc_t, mantissa_t, overallScale)

    else:
        halfN = halfN_long
        bitBudget = codingParams.targetBitsPerSample * halfN
        bitBudget += codingParams._pool_draw
        bitBudget -= nScaleBits * (sfBands_long.nBands + 1)
        bitBudget -= codingParams.nMantSizeBits * sfBands_long.nBands
        bitBudget -= codingParams._block_overhead
        bitBudget = max(bitBudget, 0)

        timeSamples = data
        mdctTimeSamples = (
            WindowForBlockType(codingParams.blockType, N_long, N_short) * timeSamples
        )
        mdctLines = MDCT(mdctTimeSamples, halfN, halfN)[:halfN]

        if codingParams._mdct_pred_correction is not None:
            mdctLines = mdctLines + codingParams._mdct_pred_correction

        maxLine = np.max(np.abs(mdctLines))
        overallScale = ScaleFactor(maxLine, nScaleBits)
        mdctLines *= 1 << overallScale

        masking_samples = (
            codingParams._masking_signal
            if codingParams._masking_signal is not None
            else timeSamples
        )
        SMRs = CalcSMRs(
            masking_samples,
            mdctLines,
            overallScale,
            codingParams.sampleRate,
            sfBands_long,
        )
        bitAlloc = BitAlloc(
            bitBudget, maxMantBits, sfBands_long.nBands, sfBands_long.nLines, SMRs
        )

        scaleFactor = np.empty(sfBands_long.nBands, dtype=np.uint64)
        nMant = halfN
        for iBand in range(sfBands_long.nBands):
            if not bitAlloc[iBand]:
                nMant -= sfBands_long.nLines[iBand]
        mantissa = np.empty(nMant, dtype=np.int32)
        iMant = 0
        for iBand in range(sfBands_long.nBands):
            lowLine = sfBands_long.lowerLine[iBand]
            highLine = sfBands_long.upperLine[iBand] + 1
            nLines = sfBands_long.nLines[iBand]
            band_lines = mdctLines[lowLine:highLine]
            scaleLine = np.max(np.abs(band_lines))
            scaleFactor[iBand] = ScaleFactor(scaleLine, nScaleBits, bitAlloc[iBand])
            if bitAlloc[iBand]:
                mantissa[iMant : iMant + nLines] = vMantissa(
                    band_lines, scaleFactor[iBand], nScaleBits, bitAlloc[iBand]
                )
                iMant += nLines

        return (scaleFactor, bitAlloc, mantissa, overallScale)
