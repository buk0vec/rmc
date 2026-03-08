"""
prediction.py -- Multi-model block prediction for the RMC rhythmic encoder.

For rhythmically structured music, samples N beats ago are often highly
correlated with the current block. This module implements predictor selection
by evaluating four candidates and choosing the one with minimum residual energy.

Predictor indices (2 bits, stored per-block in the bitstream):
    PRED_NONE    (0b00) -- no prediction, encode block directly (fallback)
    PRED_1BEAT   (0b01) -- predict from reconstructed block 1 beat ago
    PRED_2BEATS  (0b10) -- predict from reconstructed block 2 beats ago
    PRED_MEASURE (0b11) -- predict from reconstructed block 1 measure ago (4 beats)

Causality: all predictors use only previously *reconstructed* (decoded) blocks,
never the original input. The recon buffer is updated after every encoded block
by running a simulated decode pass.
"""

import numpy as np

PRED_NONE    = 0
PRED_1BEAT   = 1
PRED_2BEATS  = 2
PRED_MEASURE = 3

N_PREDICTORS = 4


def GetPredictionOffset(pred_idx, beatPeriodBlocks):
    """Return the block lookback offset for a given predictor index."""
    return [0, beatPeriodBlocks, 2 * beatPeriodBlocks, 4 * beatPeriodBlocks][pred_idx]


def GetPrediction(codingParams, iCh, pred_idx):
    """
    Return the 2N-sample prediction array for channel iCh.

    The 2N window fed into the MDCT consists of [priorBlock | currentBlock].
    The prediction mirrors this: [reconBuffer[-(O+1)] | reconBuffer[-O]],
    where O is the beat-aligned lookback offset in blocks.

    Returns a zero array if pred_idx == PRED_NONE or there is not yet
    enough reconstructed history in the buffer.
    """
    N = codingParams.nMDCTLines
    bp = getattr(codingParams, 'beatPeriodBlocks', 0)
    O = GetPredictionOffset(pred_idx, bp)

    if O == 0:
        return np.zeros(2 * N)

    buf = codingParams.reconBuffer[iCh]
    if len(buf) < O:
        return np.zeros(2 * N)  # not enough history yet — fall back to zeros

    # reconBuffer[j] ≈ data[j-1] due to 1-block MDCT delay.
    # 2N window for block k = [data[k-1-O] | data[k-O]]
    #   = [buf[-O] | buf[-(O-1)]]
    return np.concatenate([buf[-O], buf[-(O - 1)]])


def MeasureResidualEnergy(data, prediction, codingParams):
    """
    Estimate how well `prediction` fits `data` by computing the time-domain
    energy of the residual (data - prediction). Lower energy = better prediction.

    Time-domain energy is equivalent to MDCT energy by Parseval's theorem and
    works for all block types (LONG, SHORT, START, STOP) without needing to
    know the window shape during predictor selection.
    """
    return float(np.sum((data - prediction) ** 2))


def _mask_prediction_for_short(prediction, codingParams):
    """
    Zero out prediction at the padded positions of a SHORT block's 2N window.

    SHORT block IMDCT outputs zero at positions [0:pad] and [pad + 8*halfN_short + halfN_short : 2N].
    Adding prediction at those positions injects phantom signal that the START/STOP
    transition windows are not designed to cancel. Masking restores the zero assumption.
    """
    from blockswitching import N_SHORT_BLOCKS
    N_long = 2 * codingParams.nMDCTLines
    N_short = 2 * codingParams.nMDCTLines_short
    halfN_short = N_short // 2
    pad = N_long // 4 - N_short // 4
    tail = pad + (N_SHORT_BLOCKS - 1) * halfN_short + N_short
    masked = prediction.copy()
    masked[:pad] = 0.0
    masked[tail:] = 0.0
    return masked


def _expand_mantissa_long(mantissa, bitAlloc, sfBands, halfN):
    """Expand compact LONG-block mantissa to full halfN array (zeros for unallocated bands)."""
    full = np.zeros(halfN, dtype=np.int32)
    iCompact = 0
    iFull = 0
    for iBand in range(sfBands.nBands):
        nLines = sfBands.nLines[iBand]
        if bitAlloc[iBand]:
            full[iFull:iFull + nLines] = mantissa[iCompact:iCompact + nLines]
            iCompact += nLines
        iFull += nLines
    return full


def _expand_mantissa_short(mantissa_list, bitAlloc_list, sfBands, halfN):
    """Expand list of compact SHORT-block mantissas to list of full halfN arrays."""
    from blockswitching import N_SHORT_BLOCKS
    result = []
    for i in range(N_SHORT_BLOCKS):
        result.append(_expand_mantissa_long(mantissa_list[i], bitAlloc_list[i], sfBands, halfN))
    return result


def UpdateReconBuffer(codingParams, iCh, scaleFactor, bitAlloc, mantissa, overallScaleFactor):
    """
    Decode the just-encoded block, add back the prediction, apply the
    encoder-side overlap-and-add, and append the output to the recon buffer.

    Must be called after EncodeSingleChannel() for each channel, while
    codingParams.predictorIndex and codingParams.blockType are still valid
    for the current block.

    Uses codingParams.encodeOAA[iCh] to track the encoder-side OAA state
    separately from the decoder-side codingParams.overlapAndAdd.
    """
    from codec import Decode  # late import to avoid circular dependency
    from blockswitching import SHORT, ShortBlockSFBands

    N = codingParams.nMDCTLines

    # Decode() expects full-size mantissa arrays; expand from compact encoder output
    if codingParams.blockType == SHORT:
        sfBands_short = ShortBlockSFBands(codingParams.nMDCTLines_short, codingParams.sampleRate)
        mantissa_full = _expand_mantissa_short(mantissa, bitAlloc, sfBands_short, codingParams.nMDCTLines_short)
    else:
        mantissa_full = _expand_mantissa_long(mantissa, bitAlloc, codingParams.sfBands, N)

    # Decode the residual back to time domain
    residual = Decode(scaleFactor, bitAlloc, mantissa_full, overallScaleFactor, codingParams)

    # Add prediction back to recover the reconstructed signal.
    # For SHORT blocks, mask prediction at padded positions to avoid
    # injecting phantom signal where the IMDCT outputs zero.
    prediction = GetPrediction(codingParams, iCh, codingParams.predictorIndex)
    if codingParams.blockType == SHORT:
        prediction = _mask_prediction_for_short(prediction, codingParams)
    reconstructed = residual + prediction

    # Apply encoder-side overlap-and-add to get the output block
    oaa = codingParams.encodeOAA[iCh]
    output = oaa + reconstructed[:N]
    codingParams.encodeOAA[iCh] = reconstructed[N:].copy()

    # Append output block to recon buffer and trim to max needed length
    buf = codingParams.reconBuffer[iCh]
    # DEBUG: save encoder reconBuffer entries for comparison with decoder
    import prediction as _self
    if not hasattr(_self, '_enc_recon_log'):
        _self._enc_recon_log = {}
    blk = getattr(codingParams, 'blockIndex', 1) - 1
    key = (iCh, blk)
    _self._enc_recon_log[key] = output[:4].copy()
    buf.append(output.copy())
    bp = getattr(codingParams, 'beatPeriodBlocks', 0)
    max_len = 4 * bp + 2 if bp > 0 else 8
    if len(buf) > max_len:
        del buf[:-max_len]
