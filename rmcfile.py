"""
rmcfile.py -- RMC audio codec file format handler.

Extends the baseline PAC codec with block switching, M/S stereo, rhythmic
prediction, context-adaptive entropy coding, and adaptive bit budgeting.
See RMC.md for full technical reference.

Based on pacfile.py © 2009-2026 Marina Bosi & Richard E. Goldberg
"""

from audiofile import * # base class
from bitpack import *  # class for packing data into an array of bytes where each item's number of bits is specified
import codec    # module where the actual PAC coding functions reside(this module only specifies the PAC file format)
from psychoac import ScaleFactorBands, AssignMDCTLinesFromFreqLimits  # defines the grouping of MDCT lines into scale factor bands
from entropy import BlockEntropyCoder
from blockswitching import LONG, SHORT, N_SHORT_BLOCKS, ShortBlockSFBands, WindowForBlockType, SelectBlockType, mask_to_group_lens
from mdct import MDCT, IMDCT
from search import get_best_region, PRED_MAP, GAIN_TABLE, pred_type_to_samples, update_search_buffer

import numpy as np  # to allow conversion of data blocks to numpy's array object
MAX16BITS = 32767
PRED_MAP_REV = {v: k for k, v in PRED_MAP.items()}  # int → string for decoder


class RMCFile(AudioFile):
    """
    Handlers for a perceptually coded audio file I am encoding/decoding
    """

    # a file tag to recognize PAC coded files
    tag=b'RMC '

    def ReadFileHeader(self):
        """
        Reads the PAC file header from a just-opened PAC file and uses it to set
        object attributes.  File pointer ends at start of data portion.
        """
        # check file header tag to make sure it is the right kind of file
        tag=self.fp.read(4)
        if tag!=self.tag: raise RuntimeError("Tried to read a non-PAC file into a PACFile object")
        # use struct.unpack() to load up all the header data
        (sampleRate, nChannels, numSamples, nMDCTLines, nScaleBits, nMantSizeBits) \
                 = unpack('<LHLLHH',self.fp.read(calcsize('<LHLLHH')))
        nBands = unpack('<L',self.fp.read(calcsize('<L')))[0]
        nLines=  unpack('<'+str(nBands)+'H',self.fp.read(calcsize('<'+str(nBands)+'H')))
        tempo = unpack('<L',self.fp.read(calcsize('<L')))[0]
        sfBands=ScaleFactorBands(nLines)
        # load up a CodingParams object with the header data
        myParams=CodingParams()
        myParams.tempo = tempo
        myParams.sampleRate = sampleRate
        myParams.nChannels = nChannels
        myParams.numSamples = numSamples
        myParams.nMDCTLines = myParams.nSamplesPerBlock = nMDCTLines
        myParams.nScaleBits = nScaleBits
        myParams.nMantSizeBits = nMantSizeBits
        #short block switching additions
        myParams.nMDCTLines_short = nMDCTLines // 8
        myParams.sfBands_short = ShortBlockSFBands(myParams.nMDCTLines_short, sampleRate)
        myParams.prevBlockType = LONG
        myParams.blockType = LONG
        # add in scale factor band information
        myParams.sfBands =sfBands

        #RMC extras
        myParams.numSamplesQuarterNote = int((60.0/tempo) * sampleRate)
        myParams.numSamplesHalfBar = int(((60.0/tempo) * sampleRate)*2)
        myParams.numSamplesBar = int(((60.0/tempo) * sampleRate)*4)
        myParams.search_range = 255
        myParams.search_buffer = [
            np.zeros(myParams.numSamplesBar + myParams.search_range)
            for _ in range(myParams.nChannels)
        ]
        myParams.prev_use_ms = False  # tracks M/S mode of previous block for OLA transition

        # entropy coders
        myParams.entropyCoder_long = BlockEntropyCoder(14)
        myParams.entropyCoder_short = BlockEntropyCoder(14)
        # start w/o all zeroes as data from prior block to overlap-and-add for output
        overlapAndAdd = []
        for iCh in range(nChannels): overlapAndAdd.append(np.zeros(nMDCTLines, dtype=np.float64) )
        myParams.overlapAndAdd=overlapAndAdd
        return myParams



    def ReadDataBlock(self, codingParams):
        """
        Reads a block of coded data from a PACFile object that has already
        executed OpenForReading() and returns those samples as reconstituted
        signed-fraction data
        """
        halfN = codingParams.nMDCTLines
        N = 2 * halfN
        N_short = 2 * codingParams.nMDCTLines_short
        sfBands_short = codingParams.sfBands_short

        # First pass: read all channels and decode (M/S or L/R depending on flag)
        raw_decoded = []
        use_ms = False
        for iCh in range(codingParams.nChannels):
            s = self.fp.read(calcsize("<L"))
            if not s:
                if codingParams.overlapAndAdd:
                    overlapAndAdd = codingParams.overlapAndAdd
                    codingParams.overlapAndAdd = 0
                    # Convert M/S overlapAndAdd to L/R before returning final tail
                    if codingParams.prev_use_ms and codingParams.nChannels == 2:
                        M_ola, S_ola = overlapAndAdd[0], overlapAndAdd[1]
                        overlapAndAdd = [M_ola + S_ola, M_ola - S_ola]
                    return overlapAndAdd
                else:
                    return
            nBytes = unpack("<L", s)[0]
            pb = PackedBits()
            pb.SetPackedData(self.fp.read(nBytes))
            if pb.nBytes < nBytes:
                raise "Only read a partial block of coded PACFile data"

            codingParams.blockType = pb.ReadBits(2)
            pred_type = pb.ReadBits(2)
            # M/S flag: stored only in channel 0 for stereo files
            if iCh == 0 and codingParams.nChannels == 2:
                use_ms = bool(pb.ReadBits(1))

            pred_offset = 0
            pred_alpha_q = 1.0
            pred_enable_flags = None
            if pred_type != PRED_MAP[None]:
                pred_sign = pb.ReadBits(1)
                pred_offset = pb.ReadBits(8)
                if pred_sign == 1:
                    pred_offset *= -1
                pred_alpha_q = GAIN_TABLE[pb.ReadBits(3)]
                if codingParams.blockType != SHORT:  # enable flags for LONG/START/STOP blocks
                    pred_enable_flags = [bool(pb.ReadBits(1))
                                         for _ in range(codingParams.sfBands.nBands)]

            if codingParams.blockType == SHORT:
                grouping_mask = pb.ReadBits(7)
                group_lens = mask_to_group_lens(grouping_mask)
                overallScaleFactor = []
                scaleFactor = []
                bitAlloc = []
                mantissa = []
                for G in group_lens:
                    # Read shared ba/sf once for this group
                    shared_ba = []
                    shared_sf = []
                    for iBand in range(sfBands_short.nBands):
                        ba = pb.ReadBits(codingParams.nMantSizeBits)
                        if ba: ba += 1
                        shared_ba.append(ba)
                        shared_sf.append(pb.ReadBits(codingParams.nScaleBits))
                    # Read per-sub-block ovs + entropy mantissas
                    for g in range(G):
                        overallScaleFactor.append(pb.ReadBits(codingParams.nScaleBits))
                        mant_i = codingParams.entropyCoder_short.decode_block(
                            pb, shared_ba, sfBands_short, codingParams.nMDCTLines_short
                        )
                        scaleFactor.append(shared_sf)
                        bitAlloc.append(shared_ba)
                        mantissa.append(mant_i)
            else:
                overallScaleFactor = pb.ReadBits(codingParams.nScaleBits)
                scaleFactor = []
                bitAlloc = []
                for _ in range(codingParams.sfBands.nBands):
                    ba = pb.ReadBits(codingParams.nMantSizeBits)
                    if ba: ba += 1
                    bitAlloc.append(ba)
                    scaleFactor.append(pb.ReadBits(codingParams.nScaleBits))
                mantissa = codingParams.entropyCoder_long.decode_block(
                    pb, bitAlloc, codingParams.sfBands, codingParams.nMDCTLines
                )

            # Compute prediction signal from the search buffer.
            # Search buffer always holds L/R; for M/S blocks compute M or S on-the-fly.
            mdct_P = None
            if pred_type != PRED_MAP[None] and codingParams.blockType != SHORT:
                start_offset = pred_type_to_samples(PRED_MAP_REV[pred_type], codingParams)
                if use_ms and codingParams.nChannels == 2:
                    buf_L = codingParams.search_buffer[0]
                    buf_R = codingParams.search_buffer[1]
                    buf = (buf_L + buf_R) * 0.5 if iCh == 0 else (buf_L - buf_R) * 0.5
                else:
                    buf = codingParams.search_buffer[iCh]
                seg_start = len(buf) - start_offset + pred_offset
                if 0 <= seg_start and seg_start + N <= len(buf):
                    candidate = buf[seg_start : seg_start + N]
                    window = WindowForBlockType(codingParams.blockType, N, N_short)
                    mdct_P_raw = MDCT(window * candidate, halfN, halfN)[:halfN]
                    enable_mask = np.zeros(halfN)
                    if pred_enable_flags is not None:
                        for iBand, enabled in enumerate(pred_enable_flags):
                            if enabled:
                                lo = codingParams.sfBands.lowerLine[iBand]
                                hi = codingParams.sfBands.upperLine[iBand] + 1
                                enable_mask[lo:hi] = 1.0
                    mdct_P = pred_alpha_q * mdct_P_raw * enable_mask

            decodedData = self.Decode(scaleFactor, bitAlloc, mantissa, overallScaleFactor,
                                      codingParams, mdct_pred=mdct_P)
            raw_decoded.append(decodedData)

        # M/S↔L/R domain transition: if previous block used a different stereo mode,
        # convert its overlap-and-add tail to the current block's domain before summing
        if codingParams.nChannels == 2 and use_ms != codingParams.prev_use_ms:
            ola0, ola1 = codingParams.overlapAndAdd[0], codingParams.overlapAndAdd[1]
            if use_ms:
                # Transition L/R → M/S
                codingParams.overlapAndAdd[0] = (ola0 + ola1) * 0.5
                codingParams.overlapAndAdd[1] = (ola0 - ola1) * 0.5
            else:
                # Transition M/S → L/R
                codingParams.overlapAndAdd[0] = ola0 + ola1
                codingParams.overlapAndAdd[1] = ola0 - ola1
        codingParams.prev_use_ms = use_ms

        # Overlap-and-add in current block's domain (M/S or L/R)
        data_current_domain = []
        for iCh in range(codingParams.nChannels):
            decodedData = raw_decoded[iCh]
            out = np.maximum(-1., np.minimum(1.,
                np.add(codingParams.overlapAndAdd[iCh], decodedData[:halfN])
            ))
            data_current_domain.append(out)
            codingParams.overlapAndAdd[iCh] = decodedData[halfN:]

        # Convert M/S output to L/R
        if use_ms and codingParams.nChannels == 2:
            M_out, S_out = data_current_domain[0], data_current_domain[1]
            data = [M_out + S_out, M_out - S_out]
        else:
            data = data_current_domain

        # Buffer update: search buffer always stores L/R reconstructed audio
        if use_ms and codingParams.nChannels == 2:
            M_full, S_full = raw_decoded[0], raw_decoded[1]
            lr_for_buffer = [M_full + S_full, M_full - S_full]
        else:
            lr_for_buffer = raw_decoded
        for iCh in range(codingParams.nChannels):
            update_search_buffer(codingParams.search_buffer[iCh], lr_for_buffer[iCh], halfN, N)

        return data


    def WriteFileHeader(self,codingParams):
        """
        Writes the PAC file header for a just-opened PAC file and uses codingParams
        attributes for the header data.  File pointer ends at start of data portion.
        """
        # write a header tag
        self.fp.write(self.tag)
        # make sure that the number of samples in the file is a multiple of the
        # number of MDCT half-blocksize, otherwise zero pad as needed
        if not codingParams.numSamples%codingParams.nMDCTLines:
            codingParams.numSamples += (codingParams.nMDCTLines
                        - codingParams.numSamples%codingParams.nMDCTLines) # zero padding for partial final PCM block
        # also add in the delay block for the second pass w/ the last half-block
        codingParams.numSamples+= codingParams.nMDCTLines  # due to the delay in processing the first samples on both sides of the MDCT block
        # write the coded file attributes
        self.fp.write(pack('<LHLLHH',
            codingParams.sampleRate, codingParams.nChannels,
            codingParams.numSamples, codingParams.nMDCTLines,
            codingParams.nScaleBits, codingParams.nMantSizeBits  ))
        # create a ScaleFactorBand object to be used by the encoding process and write its info to header
        sfBands=ScaleFactorBands( AssignMDCTLinesFromFreqLimits(codingParams.nMDCTLines,
                                                                codingParams.sampleRate)
                                )
        codingParams.sfBands=sfBands
        #short block switching additions
        codingParams.nMDCTLines_short = codingParams.nMDCTLines // 8
        codingParams.sfBands_short = ShortBlockSFBands(codingParams.nMDCTLines_short, codingParams.sampleRate)
        codingParams.blockType = LONG    
        codingParams.entropyCoder_long = BlockEntropyCoder(14)
        codingParams.entropyCoder_short = BlockEntropyCoder(14)

        #RMC extras
        codingParams.numSamplesQuarterNote = int((60.0/codingParams.tempo) * codingParams.sampleRate)
        codingParams.numSamplesHalfBar = int(((60.0/codingParams.tempo) * codingParams.sampleRate)*2)
        codingParams.numSamplesBar = int(((60.0/codingParams.tempo) * codingParams.sampleRate)*4)
        codingParams.search_range = 255 #byte per block + 1 for sign bit
        codingParams.search_buffer = [np.zeros(codingParams.numSamplesBar + codingParams.search_range) for _ in range(codingParams.nChannels)]

        self.fp.write(pack('<L',sfBands.nBands))
        self.fp.write(pack('<'+str(sfBands.nBands)+'H',*(sfBands.nLines.tolist()) ))
        self.fp.write(pack('<L', codingParams.tempo))
        # start w/o all zeroes as prior block of unencoded data for other half of MDCT block
        priorBlock = []
        for iCh in range(codingParams.nChannels):
            priorBlock.append(np.zeros(codingParams.nMDCTLines,dtype=np.float64) )
        codingParams.priorBlock = priorBlock
        #initialize prevBlockType
        codingParams.prevBlockType = LONG
        # prediction stats
        codingParams._stat_total_blocks = 0
        codingParams._stat_pred_blocks = 0
        codingParams._stat_band_frac_sum = 0.0
        codingParams._stat_ms_blocks = 0
        # Adaptive entropy inflation: tracks compression ratio and inflates bit budget
        codingParams._entropy_ratio = 1.0
        codingParams._entropy_inflation = 1.0
        codingParams._entropy_ratio_short = 1.0
        codingParams._entropy_inflation_short = 1.0
        # Defaults for per-encode-pass state (set properly before second encode)
        codingParams.masking_signals = None
        codingParams.mdct_pred_corrections = None
        codingParams.block_overhead = None
        return


    def WriteDataBlock(self, data, codingParams):
        """
        Writes a block of signed-fraction data to a PACFile object that has
        already executed OpenForWriting()"""

        # Concatenate previous and current half-blocks to form full MDCT input windows (L/R domain)
        fullBlockData_ = []
        for iCh in range(codingParams.nChannels):
            fullBlockData_.append(np.concatenate((codingParams.priorBlock[iCh], data[iCh])))
        codingParams.priorBlock = data

        sfBands_short = codingParams.sfBands_short
        halfN = codingParams.nMDCTLines
        N = 2 * halfN
        N_short = 2 * codingParams.nMDCTLines_short

        # Block-level M/S stereo decision: use M/S if side energy < min(L energy, R energy)
        use_ms = False
        if codingParams.nChannels == 2:
            L, R = fullBlockData_[0], fullBlockData_[1]
            M = (L + R) * 0.5
            S = (L - R) * 0.5
            if np.var(S) < min(np.var(L), np.var(R)):
                use_ms = True
                fullBlockData_ = [M, S]

        # Block type state machine: LONG → START → SHORT → STOP → LONG
        # Driven by pre-computed transient map (k_attack >= 0 triggers transition to SHORT)
        k_attack = codingParams.transientBlocks.get(codingParams.blockIndex, -1)
        codingParams.blockType = SelectBlockType(k_attack, codingParams.prevBlockType)
        codingParams.prevBlockType = codingParams.blockType
        codingParams.blockIndex += 1
        if codingParams.blockType == SHORT:
            # All 8 sub-blocks in one group: minimizes sf/ba overhead and lets the
            # 2x bit budget go to mantissas. Without exact sub-window transient
            # localization, isolating a specific window is a blind guess anyway.
            codingParams.group_lens = [N_SHORT_BLOCKS]
            codingParams.grouping_mask = 0
        else:
            codingParams.group_lens = None
            codingParams.grouping_mask = 0
        current_block_type = codingParams.blockType

        # Compute M/S search buffers on-the-fly from L/R buffers (search buffers always hold L/R)
        if use_ms and codingParams.nChannels == 2:
            buf_L = codingParams.search_buffer[0]
            buf_R = codingParams.search_buffer[1]
            search_bufs = [(buf_L + buf_R) * 0.5, (buf_L - buf_R) * 0.5]
        else:
            search_bufs = [codingParams.search_buffer[iCh] for iCh in range(codingParams.nChannels)]

        # Per-channel prediction search and residual computation.
        # Produces three key outputs per channel:
        #   enable_flags_list[iCh] — bool[nBands]: which bands use prediction (written to bitstream)
        #   enable_masks[iCh]      — float[halfN]: same info expanded to MDCT lines (used in encode/decode)
        #   corrections[iCh]       — float[halfN]: cancels prediction in disabled bands so encoder
        #                             sees original signal there (None if no prediction)
        fullBlockData = []
        ranges = []
        offsets = []
        pred_mdcts = []
        alpha_idxs = []
        alpha_qs = []
        enable_flags_list = []
        enable_masks = []
        corrections = []
        for iCh in range(codingParams.nChannels):
            if current_block_type != SHORT:
                window = WindowForBlockType(current_block_type, N, N_short)
                mdct_X = MDCT(window * fullBlockData_[iCh], halfN, halfN)[:halfN]
                range_type, pcm_residual, rel_offset, mdct_P, alpha_idx, alpha_q = get_best_region(
                    mdct_X, fullBlockData_[iCh], codingParams,
                    search_bufs[iCh], block_type=current_block_type
                )
                if range_type is not None:
                    # Per-band enable: apply prediction only where it reduces energy
                    residual_full = mdct_X - alpha_q * mdct_P
                    enable_f = np.zeros(codingParams.sfBands.nBands, dtype=bool)
                    enable_m = np.zeros(halfN)
                    for iBand in range(codingParams.sfBands.nBands):
                        lo = codingParams.sfBands.lowerLine[iBand]
                        hi = codingParams.sfBands.upperLine[iBand] + 1
                        if np.dot(residual_full[lo:hi], residual_full[lo:hi]) < \
                                np.dot(mdct_X[lo:hi], mdct_X[lo:hi]):
                            enable_f[iBand] = True
                            enable_m[lo:hi] = 1.0
                    if not np.any(enable_f):
                        range_type, pcm_residual, rel_offset, mdct_P, alpha_idx, alpha_q = \
                            None, fullBlockData_[iCh], 0, None, 0, 1.0
                        enable_f = np.zeros(codingParams.sfBands.nBands, dtype=bool)
                        enable_m = np.zeros(halfN)
                        correction = None
                    else:
                        # Cancel prediction in disabled bands: encoder subtracts alpha*P from
                        # the full signal, so we add back alpha*P where prediction is off
                        correction = alpha_q * mdct_P * (1.0 - enable_m)
                else:
                    enable_f = np.zeros(codingParams.sfBands.nBands, dtype=bool)
                    enable_m = np.zeros(halfN)
                    correction = None
            else:  # SHORT block: skip prediction (transients are not repetitive)
                range_type, pcm_residual, rel_offset, mdct_P, alpha_idx, alpha_q = \
                    None, fullBlockData_[iCh], 0, None, 0, 1.0
                enable_f = np.array([], dtype=bool)
                enable_m = np.zeros(halfN)
                correction = None
            fullBlockData.append(pcm_residual)
            ranges.append(range_type)
            offsets.append(rel_offset)
            pred_mdcts.append(mdct_P)
            alpha_idxs.append(alpha_idx)
            alpha_qs.append(alpha_q)
            enable_flags_list.append(enable_f)
            enable_masks.append(enable_m)
            corrections.append(correction)

        # Prediction stats
        codingParams._stat_total_blocks += 1
        if use_ms:
            codingParams._stat_ms_blocks += 1
        pred_channels = [iCh for iCh in range(codingParams.nChannels) if ranges[iCh] is not None]
        if pred_channels:
            codingParams._stat_pred_blocks += 1
            # SHORT blocks have no per-band enables (full prediction) → count as 100%
            fracs = [(np.mean(enable_flags_list[iCh]) if len(enable_flags_list[iCh]) > 0 else 1.0)
                     for iCh in pred_channels]
            codingParams._stat_band_frac_sum += np.mean(fracs)

        # Compute per-channel bitstream overhead not accounted for in base bit budget
        block_overhead = []
        for iCh in range(codingParams.nChannels):
            oh = 32 + 2 + 2  # nBytes field (4 bytes) + block type + pred type
            if iCh == 0 and codingParams.nChannels == 2:
                oh += 1  # M/S flag
            if ranges[iCh] is not None:
                oh += 1 + 8 + 3  # sign + offset + gain
                if current_block_type != SHORT:
                    oh += codingParams.sfBands.nBands  # per-band enable flags
            if current_block_type == SHORT:
                oh += 7  # grouping mask
            block_overhead.append(oh)

        # Encode the residual signals with original signal for psychoacoustic masking
        codingParams.masking_signals = fullBlockData_
        codingParams.mdct_pred_corrections = corrections
        codingParams.block_overhead = block_overhead
        (scaleFactor, bitAlloc, mantissa, overallScaleFactor) = self.Encode(fullBlockData, codingParams)
        codingParams.masking_signals = None
        codingParams.mdct_pred_corrections = None

        # Encoder-side decode: reconstruct lossy output and store in search buffer so that
        # future prediction searches see the same signal the decoder will have.
        halfN_short = codingParams.nMDCTLines_short
        decoded_channels = []
        for iCh in range(codingParams.nChannels):
            if current_block_type != SHORT:
                mantissa_full = codec.ExpandMantissa(
                    mantissa[iCh], bitAlloc[iCh], codingParams.sfBands, halfN
                )
                scaled_pred = (alpha_qs[iCh] * pred_mdcts[iCh] * enable_masks[iCh]
                               if pred_mdcts[iCh] is not None else None)
                decodedData = codec.Decode(
                    scaleFactor[iCh], bitAlloc[iCh], mantissa_full,
                    overallScaleFactor[iCh], codingParams, mdct_pred=scaled_pred
                )
            else:  # SHORT block (START/STOP use the LONG path above)
                full_mant = [codec.ExpandMantissa(mantissa[iCh][i], bitAlloc[iCh][i], sfBands_short, halfN_short)
                             for i in range(N_SHORT_BLOCKS)]
                decodedData = codec.Decode(
                    scaleFactor[iCh], bitAlloc[iCh], full_mant,
                    overallScaleFactor[iCh], codingParams
                )
            decoded_channels.append(decodedData)

        # Convert M/S decoded data back to L/R before storing in search buffer
        if use_ms and codingParams.nChannels == 2:
            M_hat, S_hat = decoded_channels[0], decoded_channels[1]
            lr_for_buffer = [M_hat + S_hat, M_hat - S_hat]
        else:
            lr_for_buffer = decoded_channels
        for iCh in range(codingParams.nChannels):
            update_search_buffer(codingParams.search_buffer[iCh], lr_for_buffer[iCh], halfN, N)

        # Write bitstream per channel:
        #   [2b block_type | 2b pred_type | 1b ms_flag (ch0 only)]
        #   [1b sign + 8b offset + 3b gain + nBands enable flags (if pred active)]
        #   [7b grouping_mask (if SHORT)]
        #   [nScaleBits ovs | nMantSizeBits+nScaleBits per band | entropy-coded mantissas]
        for iCh in range(codingParams.nChannels):
            entropy_pbs = []
            nBits = 4  # 2 bits block type + 2 bits prediction type
            if iCh == 0 and codingParams.nChannels == 2:
                nBits += 1  # M/S flag
            if ranges[iCh] is not None:
                nBits += 9 + 3  # sign+offset + gain
                if codingParams.blockType != SHORT:
                    nBits += codingParams.sfBands.nBands  # per-band enables for LONG/START/STOP
            if codingParams.blockType == SHORT:
                group_lens = codingParams.group_lens
                nBits += 7  # grouping mask
                sub_idx = 0
                for G in group_lens:
                    # Shared sf/ba written once per group
                    for iBand in range(sfBands_short.nBands):
                        nBits += codingParams.nMantSizeBits + codingParams.nScaleBits
                    # Per sub-block: overallScale + entropy mantissas
                    for g in range(G):
                        entropy_pb = codingParams.entropyCoder_short.encode_block(
                            mantissa[iCh][sub_idx + g], bitAlloc[iCh][sub_idx + g], sfBands_short
                        )
                        entropy_pbs.append(entropy_pb)
                        nBits += codingParams.nScaleBits  # overallScaleFactor
                        nBits += entropy_pb.nBits
                    sub_idx += G
                # Update SHORT entropy compression ratio (EMA)
                _raw_short = sum(
                    bitAlloc[iCh][i][iBand] * sfBands_short.nLines[iBand]
                    for i in range(N_SHORT_BLOCKS)
                    for iBand in range(sfBands_short.nBands)
                    if bitAlloc[iCh][i][iBand] > 0
                )
                _ent_short = sum(ep.nBits for ep in entropy_pbs)
                if _raw_short > 0:
                    _r = _ent_short / _raw_short
                    codingParams._entropy_ratio_short = 0.9 * codingParams._entropy_ratio_short + 0.1 * _r
                    codingParams._entropy_inflation_short = min(2.5, max(0.5, 1.0 / codingParams._entropy_ratio_short))
            else:
                entropy_pb = codingParams.entropyCoder_long.encode_block(
                    mantissa[iCh], bitAlloc[iCh], codingParams.sfBands
                )
                nBits += codingParams.nScaleBits
                for iBand in range(codingParams.sfBands.nBands):
                    nBits += codingParams.nMantSizeBits + codingParams.nScaleBits
                nBits += entropy_pb.nBits
                # Update LONG entropy compression ratio (EMA)
                _raw_long = sum(
                    bitAlloc[iCh][iBand] * codingParams.sfBands.nLines[iBand]
                    for iBand in range(codingParams.sfBands.nBands)
                    if bitAlloc[iCh][iBand] > 0
                )
                if _raw_long > 0:
                    _r = entropy_pb.nBits / _raw_long
                    codingParams._entropy_ratio = 0.9 * codingParams._entropy_ratio + 0.1 * _r
                    codingParams._entropy_inflation = min(2.5, max(0.5, 1.0 / codingParams._entropy_ratio))

            nBytes = (nBits + BYTESIZE - 1) // BYTESIZE
            self.fp.write(pack("<L", int(nBytes)))

            pb = PackedBits()
            pb.Size(nBytes)
            pb.WriteBits(codingParams.blockType, 2)
            pb.WriteBits(PRED_MAP[ranges[iCh]], 2)
            if iCh == 0 and codingParams.nChannels == 2:
                pb.WriteBits(1 if use_ms else 0, 1)
            if ranges[iCh] is not None:
                sign = 1 if offsets[iCh] < 0 else 0
                pb.WriteBits(sign, 1)
                pb.WriteBits(abs(offsets[iCh]), 8)
                pb.WriteBits(alpha_idxs[iCh], 3)
                if codingParams.blockType != SHORT:
                    for iBand in range(codingParams.sfBands.nBands):
                        pb.WriteBits(1 if enable_flags_list[iCh][iBand] else 0, 1)

            if codingParams.blockType == SHORT:
                pb.WriteBits(codingParams.grouping_mask, 7)
                ep_idx = 0
                sub_idx = 0
                for G in group_lens:
                    # Write shared ba/sf once per group (use sub_idx; all in group are identical)
                    for iBand in range(sfBands_short.nBands):
                        ba = bitAlloc[iCh][sub_idx][iBand]
                        if ba: ba -= 1
                        pb.WriteBits(ba, codingParams.nMantSizeBits)
                        pb.WriteBits(scaleFactor[iCh][sub_idx][iBand], codingParams.nScaleBits)
                    # Write per-sub-block overallScale + entropy mantissas
                    for g in range(G):
                        pb.WriteBits(overallScaleFactor[iCh][sub_idx + g], codingParams.nScaleBits)
                        pb.WriteBits(entropy_pbs[ep_idx].buffer, entropy_pbs[ep_idx].nBits)
                        ep_idx += 1
                    sub_idx += G
            else:
                pb.WriteBits(overallScaleFactor[iCh], codingParams.nScaleBits)
                for iBand in range(codingParams.sfBands.nBands):
                    ba = bitAlloc[iCh][iBand]
                    if ba: ba -= 1
                    pb.WriteBits(ba, codingParams.nMantSizeBits)
                    pb.WriteBits(scaleFactor[iCh][iBand], codingParams.nScaleBits)
                pb.WriteBits(entropy_pb.buffer, entropy_pb.nBits)
            self.fp.write(pb.GetPackedData())
        return

    def Close(self,codingParams):
        """
        Flushes the last data block through the encoding process (if encoding)
        and closes the audio file
        """
        # determine if encoding or encoding and, if encoding, do last block
        if self.fp.mode == "wb":  # we are writing to the PACFile, must be encode
            # we are writing the coded file -- pass a block of zeros to move last data block to other side of MDCT block
            data = [ np.zeros(codingParams.nMDCTLines),
                     np.zeros(codingParams.nMDCTLines) ]
            self.WriteDataBlock(data, codingParams)
            total = codingParams._stat_total_blocks
            pred  = codingParams._stat_pred_blocks
            ms    = codingParams._stat_ms_blocks
            avg_band_frac = (codingParams._stat_band_frac_sum / pred * 100) if pred > 0 else 0.0
            print(f"\n--- RMC encoding stats ---")
            print(f"  Blocks with prediction : {pred} / {total} ({100*pred/total:.1f}%)")
            print(f"  Avg bands predicted    : {avg_band_frac:.1f}% of critical bands (when active)")
            if codingParams.nChannels == 2:
                print(f"  Blocks with M/S stereo : {ms} / {total} ({100*ms/total:.1f}%)")
            print(f"  Entropy inflation (LONG) : {codingParams._entropy_inflation:.2f}x (ratio={codingParams._entropy_ratio:.3f})")
            print(f"  Entropy inflation (SHORT): {codingParams._entropy_inflation_short:.2f}x (ratio={codingParams._entropy_ratio_short:.3f})")
        self.fp.close()


    def Encode(self,data,codingParams):
        """
        Encodes multichannel audio data and returns a tuple containing
        the scale factors, mantissa bit allocations, quantized mantissas,
        and the overall scale factor for each channel.
        """
        #Passes encoding logic to the Encode function defined in the codec module
        return codec.Encode(data,codingParams)

    def Decode(self,scaleFactor,bitAlloc,mantissa, overallScaleFactor,codingParams,mdct_pred=None):
        """
        Decodes a single audio channel of data based on the values of its scale factors,
        bit allocations, quantized mantissas, and overall scale factor.
        """
        #Passes decoding logic to the Decode function defined in the codec module
        return codec.Decode(scaleFactor,bitAlloc,mantissa, overallScaleFactor,codingParams,mdct_pred)






#-----------------------------------------------------------------------------

if __name__ == "__main__":
    from prepare_materials import rmc
    import time
    elapsed = time.time()
    rmc("inputs/Brooklyn.wav", "Brooklyn_96.wav", rate_kb=96)
    print(f"\nDone in {time.time() - elapsed:.1f}s")
