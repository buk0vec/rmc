"""
rmcfile.py -- RMC audio codec file format handler.

Extends the baseline PAC codec with block switching, M/S stereo, rhythmic
prediction, context-adaptive entropy coding, and adaptive bit budgeting.
See RMC.md for full technical reference.

Based on pacfile.py © 2009-2026 Marina Bosi & Richard E. Goldberg
"""

import numpy as np  # to allow conversion of data blocks to numpy's array object

import codec  # module where the actual PAC coding functions reside(this module only specifies the PAC file format)
from audiofile import AudioFile, CodingParams  # base class
from bitpack import (  # class for packing data into an array of bytes where each item's number of bits is specified
    BYTESIZE,
    PackedBits,
    calcsize,
    pack,
    unpack,
)
from blockswitching import (
    LONG,
    N_SHORT_BLOCKS,
    SHORT,
    SelectBlockType,
    ShortBlockSFBands,
    WindowForBlockType,
    mask_to_group_lens,
)
from entropy import BlockEntropyCoder, RawMantissaCoder
from features import (
    COMPLEX_PREDICTION,
    ENTROPY_CODING,
    MID_SIDE_CODING,
    PRED_ENABLE_RATIO,
    PRED_ENABLE_RMS,
    PRED_ENABLE_SF,
    PRED_EXTENDED_RANGE,
    PRED_NLINES_THRESH,
    PREDICTION,
    VARIABLE_BIT_RATE,
)
from mdct import MDCT
from psychoac import (  # defines the grouping of MDCT lines into scale factor bands
    AssignMDCTLinesFromFreqLimits,
    ScaleFactorBands,
)
from quantize import ScaleFactor
from search import (
    COMPLEX_GAIN_TABLE,
    GAIN_TABLE,
    PRED_MAP,
    get_best_region,
    phase_idx_to_radians,
    pred_type_to_samples,
    update_search_buffer,
)

MAX16BITS = 32767
PRED_MAP_REV = {v: k for k, v in PRED_MAP.items()}  # int → string for decoder


class RMCFile(AudioFile):
    """
    Handlers for a perceptually coded audio file I am encoding/decoding
    """

    # a file tag to recognize PAC coded files
    tag = b"RMC "

    def ReadFileHeader(self):
        """
        Reads the PAC file header from a just-opened PAC file and uses it to set
        object attributes.  File pointer ends at start of data portion.
        """
        # check file header tag to make sure it is the right kind of file
        tag = self.fp.read(4)
        if tag != self.tag:
            raise RuntimeError("Tried to read a non-PAC file into a PACFile object")
        # use struct.unpack() to load up all the header data
        (sampleRate, nChannels, numSamples, nMDCTLines, nScaleBits, nMantSizeBits) = (
            unpack("<LHLLHH", self.fp.read(calcsize("<LHLLHH")))
        )
        nBands = unpack("<L", self.fp.read(calcsize("<L")))[0]
        nLines = unpack(
            "<" + str(nBands) + "H", self.fp.read(calcsize("<" + str(nBands) + "H"))
        )
        tempo = unpack("<L", self.fp.read(calcsize("<L")))[0]
        sfBands = ScaleFactorBands(nLines)
        # load up a CodingParams object with the header data
        myParams = CodingParams()
        myParams.tempo = tempo
        myParams.sampleRate = sampleRate
        myParams.nChannels = nChannels
        myParams.numSamples = numSamples
        myParams.nMDCTLines = myParams.nSamplesPerBlock = nMDCTLines
        myParams.nScaleBits = nScaleBits
        myParams.nMantSizeBits = nMantSizeBits
        # short block switching additions
        myParams.nMDCTLines_short = nMDCTLines // 8
        myParams.sfBands_short = ShortBlockSFBands(
            myParams.nMDCTLines_short, sampleRate
        )
        myParams.prevBlockType = LONG
        myParams.blockType = LONG
        # add in scale factor band information
        myParams.sfBands = sfBands

        # RMC extras
        myParams.numSamplesQuarterNote = int((60.0 / tempo) * sampleRate)
        myParams.numSamplesHalfBar = int(((60.0 / tempo) * sampleRate) * 2)
        myParams.numSamplesBar = int(((60.0 / tempo) * sampleRate) * 4)
        myParams.numSamples2Bar = int(((60.0 / tempo) * sampleRate) * 8)
        myParams.numSamples4Bar = int(((60.0 / tempo) * sampleRate) * 16)
        myParams.search_range = 1023
        # Buffer sized to hold the furthest lookback (4 bars) plus one block and search margin.
        buf_size = (
            (myParams.numSamples4Bar if PRED_EXTENDED_RANGE else myParams.numSamplesBar)
            + nMDCTLines
            + myParams.search_range
        )
        myParams.search_buffer = [np.zeros(buf_size) for _ in range(myParams.nChannels)]
        myParams.buffer_fill = 0  # samples of real audio accumulated so far
        myParams.prev_use_ms = (
            False  # tracks M/S mode of previous block for OLA transition
        )

        # entropy coders
        myParams.entropyCoder_long = (
            BlockEntropyCoder(14) if ENTROPY_CODING else RawMantissaCoder()
        )
        myParams.entropyCoder_short = (
            BlockEntropyCoder(14) if ENTROPY_CODING else RawMantissaCoder()
        )
        # start w/o all zeroes as data from prior block to overlap-and-add for output
        overlapAndAdd = []
        for _ in range(nChannels):
            overlapAndAdd.append(np.zeros(nMDCTLines, dtype=np.float64))
        myParams.overlapAndAdd = overlapAndAdd
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
                raise RuntimeError("Only read a partial block of coded PACFile data")

            codingParams.blockType = pb.ReadBits(2)
            # Quarter/half/bar, potentially more
            pred_type = pb.ReadBits(3 if PRED_EXTENDED_RANGE else 2)
            # M/S flag: stored only in channel 0 for stereo files
            if iCh == 0 and codingParams.nChannels == 2:
                use_ms = bool(pb.ReadBits(1))

            pred_offset = 0
            pred_alpha_q = 1.0
            # per-band arrays (COMPLEX_PREDICTION only)
            pred_alpha_qs = None
            pred_phase_qs = None
            pred_enable_flags = None
            # If prediction is enabled for this block
            if pred_type != PRED_MAP[None]:
                # Read in the offset
                pred_sign = pb.ReadBits(1)
                pred_offset = pb.ReadBits(10)
                if pred_sign == 1:
                    pred_offset *= -1
                if not COMPLEX_PREDICTION:
                    # Use global gain for predictive region
                    pred_alpha_q = GAIN_TABLE[pb.ReadBits(4)]
                if (
                    codingParams.blockType != SHORT
                ):  # we only predict for for LONG/START/STOP blocks
                    # Read the enable flags
                    pred_enable_flags = [
                        bool(pb.ReadBits(1)) for _ in range(codingParams.sfBands.nBands)
                    ]
                    if COMPLEX_PREDICTION:
                        pred_alpha_qs = np.ones(codingParams.sfBands.nBands)
                        pred_phase_qs = np.zeros(codingParams.sfBands.nBands)
                        # Magnitude and phases per SFB
                        for iBand in range(codingParams.sfBands.nBands):
                            if pred_enable_flags[iBand]:
                                # Translate from bits
                                pred_alpha_qs[iBand] = COMPLEX_GAIN_TABLE[
                                    pb.ReadBits(3)
                                ]
                                pred_phase_qs[iBand] = phase_idx_to_radians(
                                    pb.ReadBits(4)
                                )
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
                        if ba:
                            ba += 1
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
            # LONG/START/STOP blocks
            else:
                overallScaleFactor = pb.ReadBits(codingParams.nScaleBits)
                scaleFactor = []
                bitAlloc = []
                # Read in scale factor and mantissa bit count per band
                for _ in range(codingParams.sfBands.nBands):
                    ba = pb.ReadBits(codingParams.nMantSizeBits)
                    if ba:
                        ba += 1
                    bitAlloc.append(ba)
                    scaleFactor.append(pb.ReadBits(codingParams.nScaleBits))
                # Decode block w/ long block entropy coder to get mantissas
                mantissa = codingParams.entropyCoder_long.decode_block(
                    pb, bitAlloc, codingParams.sfBands, codingParams.nMDCTLines
                )

            # Compute prediction signal from the search buffer.
            # Search buffer always holds L/R; for M/S blocks compute M or S on-the-fly.
            mdct_P = None
            if pred_type != PRED_MAP[None] and codingParams.blockType != SHORT:
                start_offset = pred_type_to_samples(
                    PRED_MAP_REV[pred_type], codingParams
                )
                # Calc M/S from L/R buffer if needed
                if use_ms and codingParams.nChannels == 2:
                    buf_L = codingParams.search_buffer[0]
                    buf_R = codingParams.search_buffer[1]
                    buf = (buf_L + buf_R) * 0.5 if iCh == 0 else (buf_L - buf_R) * 0.5
                else:
                    buf = codingParams.search_buffer[iCh]
                # Calculate start of prediction block w/ halfN offset due to 1/2 overlap
                seg_start = len(buf) - start_offset + pred_offset - halfN
                # Safety check
                if 0 <= seg_start and seg_start + N <= len(buf):
                    # Pull out candidate, window and take MCLT (return_complex=True)
                    candidate = buf[seg_start : seg_start + N].copy()
                    window = WindowForBlockType(codingParams.blockType, N, N_short)
                    mdct_P_raw_cplx = MDCT(
                        window * candidate, halfN, halfN, return_complex=True
                    )[:halfN]
                    # Pull out real MDCT part from MCLT
                    mdct_P_raw = mdct_P_raw_cplx.real
                    if COMPLEX_PREDICTION:
                        # Also pull out the MDST part from MCLT
                        mdst_P_raw = mdct_P_raw_cplx.imag
                        effective_pred = np.zeros(halfN)
                        if (
                            pred_enable_flags is not None
                            and pred_alpha_qs is not None
                            and pred_phase_qs is not None
                        ):
                            for iBand, enabled in enumerate(pred_enable_flags):
                                if enabled:
                                    lo = codingParams.sfBands.lowerLine[iBand]
                                    hi = codingParams.sfBands.upperLine[iBand] + 1
                                    # Real part of complex gain α·e^{jϕ} = α cos ϕ + j α sin ϕ
                                    a_b = pred_alpha_qs[iBand] * np.cos(
                                        pred_phase_qs[iBand]
                                    )
                                    # Imag part of complex gain
                                    b_b = pred_alpha_qs[iBand] * np.sin(
                                        pred_phase_qs[iBand]
                                    )
                                    # Calculate real part of predicted MCLT
                                    # MDCT = Re<(a_b + j b_b)(MDCT + j MDST)> = a_b MDCT - b_b MDS
                                    effective_pred[lo:hi] = (
                                        a_b * mdct_P_raw[lo:hi]
                                        - b_b * mdst_P_raw[lo:hi]
                                    )
                        mdct_P = effective_pred
                    else:
                        effective_pred = pred_alpha_q * mdct_P_raw
                        enable_mask = np.zeros(halfN)
                        for iBand, enabled in enumerate(pred_enable_flags):
                            if enabled:
                                lo = codingParams.sfBands.lowerLine[iBand]
                                hi = codingParams.sfBands.upperLine[iBand] + 1
                                enable_mask[lo:hi] = 1.0
                        mdct_P = effective_pred * enable_mask

            decodedData = self.Decode(
                scaleFactor,
                bitAlloc,
                mantissa,
                overallScaleFactor,
                codingParams,
                mdct_pred=mdct_P,
            )
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
            out = np.maximum(
                -1.0,
                np.minimum(
                    1.0, np.add(codingParams.overlapAndAdd[iCh], decodedData[:halfN])
                ),
            )
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
            update_search_buffer(
                codingParams.search_buffer[iCh], lr_for_buffer[iCh], halfN, N
            )
        codingParams.buffer_fill = min(
            codingParams.buffer_fill + halfN, len(codingParams.search_buffer[0])
        )

        return data

    def WriteFileHeader(self, codingParams):
        """
        Writes the PAC file header for a just-opened PAC file and uses codingParams
        attributes for the header data.  File pointer ends at start of data portion.
        """
        # write a header tag
        self.fp.write(self.tag)
        # make sure that the number of samples in the file is a multiple of the
        # number of MDCT half-blocksize, otherwise zero pad as needed
        if codingParams.numSamples % codingParams.nMDCTLines:
            codingParams.numSamples += (
                codingParams.nMDCTLines
                - codingParams.numSamples % codingParams.nMDCTLines
            )  # zero padding for partial final PCM block
        # also add in the delay block for the second pass w/ the last half-block
        codingParams.numSamples += codingParams.nMDCTLines  # due to the delay in processing the first samples on both sides of the MDCT block
        # write the coded file attributes
        self.fp.write(
            pack(
                "<LHLLHH",
                codingParams.sampleRate,
                codingParams.nChannels,
                codingParams.numSamples,
                codingParams.nMDCTLines,
                codingParams.nScaleBits,
                codingParams.nMantSizeBits,
            )
        )
        # create a ScaleFactorBand object to be used by the encoding process and write its info to header
        sfBands = ScaleFactorBands(
            AssignMDCTLinesFromFreqLimits(
                codingParams.nMDCTLines, codingParams.sampleRate
            )
        )
        codingParams.sfBands = sfBands
        # short block switching additions
        codingParams.nMDCTLines_short = codingParams.nMDCTLines // 8
        codingParams.sfBands_short = ShortBlockSFBands(
            codingParams.nMDCTLines_short, codingParams.sampleRate
        )
        codingParams.blockType = LONG
        codingParams.entropyCoder_long = (
            BlockEntropyCoder(14) if ENTROPY_CODING else RawMantissaCoder()
        )
        codingParams.entropyCoder_short = (
            BlockEntropyCoder(14) if ENTROPY_CODING else RawMantissaCoder()
        )

        # RMC extras
        codingParams.numSamplesQuarterNote = int(
            (60.0 / codingParams.tempo) * codingParams.sampleRate
        )
        codingParams.numSamplesHalfBar = int(
            ((60.0 / codingParams.tempo) * codingParams.sampleRate) * 2
        )
        codingParams.numSamplesBar = int(
            ((60.0 / codingParams.tempo) * codingParams.sampleRate) * 4
        )
        codingParams.numSamples2Bar = int(
            ((60.0 / codingParams.tempo) * codingParams.sampleRate) * 8
        )
        codingParams.numSamples4Bar = int(
            ((60.0 / codingParams.tempo) * codingParams.sampleRate) * 16
        )
        codingParams.search_range = 1023
        buf_size = (
            (
                codingParams.numSamples4Bar
                if PRED_EXTENDED_RANGE
                else codingParams.numSamplesBar
            )
            + codingParams.nMDCTLines
            + codingParams.search_range
        )
        codingParams.search_buffer = [
            np.zeros(buf_size) for _ in range(codingParams.nChannels)
        ]
        codingParams.buffer_fill = 0  # samples of real audio accumulated so far

        self.fp.write(pack("<L", sfBands.nBands))
        self.fp.write(pack("<" + str(sfBands.nBands) + "H", *(sfBands.nLines.tolist())))
        self.fp.write(pack("<L", codingParams.tempo))
        # start w/o all zeroes as prior block of unencoded data for other half of MDCT block
        priorBlock = []
        for iCh in range(codingParams.nChannels):
            priorBlock.append(np.zeros(codingParams.nMDCTLines, dtype=np.float64))
        codingParams.priorBlock = priorBlock
        # initialize prevBlockType
        codingParams.prevBlockType = LONG
        # prediction stats
        codingParams._stat_total_blocks = 0
        codingParams._stat_pred_blocks = 0
        codingParams._stat_band_frac_sum = 0.0
        codingParams._stat_ms_blocks = 0
        codingParams._stat_bits_blocks = (
            0  # non-silent LONG blocks (any bits allocated)
        )
        codingParams._stat_pred_bits_blocks = (
            0  # non-silent LONG blocks with prediction active
        )
        codingParams._stat_pred_alloc_frac_sum = (
            0.0  # pred-enabled / allocated bands (sum)
        )
        codingParams._stat_pred_alloc_count = 0
        codingParams._stat_alpha_sum = 0.0  # average prediction gain magnitude (sum)
        codingParams._stat_alpha_count = 0
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
        block_input = []
        for iCh in range(codingParams.nChannels):
            block_input.append(
                np.concatenate((codingParams.priorBlock[iCh], data[iCh]))
            )
        codingParams.priorBlock = data

        sfBands_short = codingParams.sfBands_short
        halfN = codingParams.nMDCTLines
        N = 2 * halfN
        N_short = 2 * codingParams.nMDCTLines_short

        # Block-level M/S stereo decision: use M/S if side energy < min(L energy, R energy)
        use_ms = False
        if MID_SIDE_CODING and codingParams.nChannels == 2:
            L, R = block_input[0], block_input[1]
            M = (L + R) * 0.5
            S = (L - R) * 0.5
            if np.var(S) < min(np.var(L), np.var(R)):
                use_ms = True
                block_input = [M, S]

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
            search_bufs = [
                codingParams.search_buffer[iCh] for iCh in range(codingParams.nChannels)
            ]

        # Per-channel prediction search and residual computation.
        # Produces three key outputs per channel:
        #   enable_flags_list[iCh] — bool[nBands]: which bands use prediction (written to bitstream)
        #   enable_masks[iCh]      — float[halfN]: same info expanded to MDCT lines (used in encode/decode)
        #   corrections[iCh]       — float[halfN]: cancels prediction in disabled bands so encoder
        #                             sees original signal there (None if no prediction)
        residuals = []
        ranges = []
        offsets = []
        pred_mdcts = []
        pred_mdsts = []
        alpha_idxs = []
        alpha_qs = []
        phase_idxs = []
        enable_flags_list = []
        enable_masks = []
        corrections = []
        for iCh in range(codingParams.nChannels):
            if PREDICTION and current_block_type != SHORT:
                window = WindowForBlockType(current_block_type, N, N_short)
                mdct_X = MDCT(window * block_input[iCh], halfN, halfN)[:halfN]
                # Run predictive search
                (
                    range_type,
                    pcm_residual,
                    rel_offset,
                    mdct_P,
                    mdst_P,
                    alpha_idx,
                    alpha_q,
                    phase_idx,
                ) = get_best_region(
                    mdct_X,
                    block_input[iCh],
                    codingParams,
                    search_bufs[iCh],
                    block_type=current_block_type,
                )
                # If we found a good enough region for prediction
                if range_type is not None:
                    # Reconstruct per-band residual for the enable decision
                    if COMPLEX_PREDICTION:
                        residual_full = np.empty(halfN)
                        for iBand in range(codingParams.sfBands.nBands):
                            lo = codingParams.sfBands.lowerLine[iBand]
                            hi = codingParams.sfBands.upperLine[iBand] + 1
                            a_b = alpha_q[iBand] * np.cos(
                                phase_idx_to_radians(phase_idx[iBand])
                            )
                            b_b = alpha_q[iBand] * np.sin(
                                phase_idx_to_radians(phase_idx[iBand])
                            )
                            residual_full[lo:hi] = (
                                mdct_X[lo:hi]
                                - a_b * mdct_P[lo:hi]
                                + b_b * mdst_P[lo:hi]
                            )
                    else:
                        residual_full = mdct_X - alpha_q * mdct_P
                    # Per-band enable: apply prediction only where it reduces the signal
                    nScaleBits = codingParams.nScaleBits
                    enable_f = np.zeros(codingParams.sfBands.nBands, dtype=bool)
                    enable_m = np.zeros(halfN)
                    for iBand in range(codingParams.sfBands.nBands):
                        lo = codingParams.sfBands.lowerLine[iBand]
                        hi = codingParams.sfBands.upperLine[iBand] + 1
                        orig_band = np.abs(mdct_X[lo:hi])
                        res_band = np.abs(residual_full[lo:hi])
                        orig_rms = (
                            np.sqrt(np.mean(orig_band**2)) if orig_band.size else 0.0
                        )
                        res_rms = (
                            np.sqrt(np.mean(res_band**2)) if res_band.size else 0.0
                        )
                        if COMPLEX_PREDICTION:
                            n_lines = hi - lo
                            if PRED_NLINES_THRESH:
                                # Break-even: n_lines * reduction_dB/6 >= 7 bits of per-band overhead
                                # => RMS threshold = 10^(-42 / (20.0 * n_lines))
                                threshold = 10.0 ** (-42.0 / (20.0 * n_lines))
                            else:
                                threshold = PRED_ENABLE_RATIO
                            enable = (
                                res_rms < threshold * orig_rms
                                if orig_rms > 0
                                else False
                            )
                        elif PRED_ENABLE_RMS:
                            enable = res_rms < orig_rms if orig_rms > 0 else False
                        elif PRED_ENABLE_SF:
                            enable = ScaleFactor(
                                np.amax(res_band), nScaleBits, 0
                            ) > ScaleFactor(np.amax(orig_band), nScaleBits, 0)
                        else:
                            enable = np.amax(res_band) < np.amax(orig_band)
                        if enable:
                            enable_f[iBand] = True
                            enable_m[lo:hi] = 1.0
                    if not np.any(enable_f):
                        range_type, pcm_residual, rel_offset, mdct_P, mdst_P = (
                            None,
                            block_input[iCh],
                            0,
                            None,
                            None,
                        )
                        enable_f = np.zeros(codingParams.sfBands.nBands, dtype=bool)
                        enable_m = np.zeros(halfN)
                        correction = None
                    else:
                        # Per-band MDCT-domain correction: subtract prediction in enabled bands.
                        # pcm_residual == input_pcm, so MDCT(pcm_residual)+correction = per-band residual.
                        if COMPLEX_PREDICTION:
                            correction = np.zeros(halfN)
                            for iBand in range(codingParams.sfBands.nBands):
                                if enable_f[iBand]:
                                    lo = codingParams.sfBands.lowerLine[iBand]
                                    hi = codingParams.sfBands.upperLine[iBand] + 1
                                    a_b = alpha_q[iBand] * np.cos(
                                        phase_idx_to_radians(phase_idx[iBand])
                                    )
                                    b_b = alpha_q[iBand] * np.sin(
                                        phase_idx_to_radians(phase_idx[iBand])
                                    )
                                    correction[lo:hi] = -(
                                        a_b * mdct_P[lo:hi] - b_b * mdst_P[lo:hi]
                                    )
                        else:
                            correction = alpha_q * mdct_P * (1.0 - enable_m)
                else:
                    enable_f = np.zeros(codingParams.sfBands.nBands, dtype=bool)
                    enable_m = np.zeros(halfN)
                    correction = None
            else:  # SHORT block (transients not repetitive) or PREDICTION disabled
                (
                    range_type,
                    pcm_residual,
                    rel_offset,
                    mdct_P,
                    mdst_P,
                    alpha_idx,
                    alpha_q,
                    phase_idx,
                ) = None, block_input[iCh], 0, None, None, 0, 1.0, 128
                enable_f = np.array([], dtype=bool)
                enable_m = np.zeros(halfN)
                correction = None
            residuals.append(pcm_residual)
            ranges.append(range_type)
            offsets.append(rel_offset)
            pred_mdcts.append(mdct_P)
            pred_mdsts.append(mdst_P)
            alpha_idxs.append(alpha_idx)
            alpha_qs.append(alpha_q)
            phase_idxs.append(phase_idx)
            enable_flags_list.append(enable_f)
            enable_masks.append(enable_m)
            corrections.append(correction)

        # Prediction stats
        codingParams._stat_total_blocks += 1
        if use_ms:
            codingParams._stat_ms_blocks += 1
        pred_channels = [
            iCh for iCh in range(codingParams.nChannels) if ranges[iCh] is not None
        ]
        if pred_channels:
            codingParams._stat_pred_blocks += 1
            # SHORT blocks have no per-band enables (full prediction) → count as 100%
            fracs = [
                (
                    np.mean(enable_flags_list[iCh])
                    if len(enable_flags_list[iCh]) > 0
                    else 1.0
                )
                for iCh in pred_channels
            ]
            codingParams._stat_band_frac_sum += np.mean(fracs)

        # Compute per-channel bitstream overhead not accounted for in base bit budget
        _pred_type_bits = 3 if PRED_EXTENDED_RANGE else 2
        block_overhead = []
        for iCh in range(codingParams.nChannels):
            oh = (
                32 + 2 + _pred_type_bits
            )  # nBytes field (4 bytes) + block type + pred type
            if iCh == 0 and codingParams.nChannels == 2:
                oh += 1  # M/S flag
            if ranges[iCh] is not None:
                oh += 1 + 10  # sign + offset
                if not COMPLEX_PREDICTION:
                    oh += 4  # global scalar gain
                if current_block_type != SHORT:
                    oh += codingParams.sfBands.nBands  # per-band enable flags
                    if COMPLEX_PREDICTION:
                        oh += 7 * int(
                            np.sum(enable_flags_list[iCh])
                        )  # 3b gain + 4b phase per enabled band
            if current_block_type == SHORT:
                oh += 7  # grouping mask
            block_overhead.append(oh)

        # Encode the residual signals with original signal for psychoacoustic masking
        codingParams.masking_signals = block_input
        codingParams.mdct_pred_corrections = corrections
        codingParams.block_overhead = block_overhead
        (scaleFactor, bitAlloc, mantissa, overallScaleFactor) = self.Encode(
            residuals, codingParams
        )
        codingParams.masking_signals = None
        codingParams.mdct_pred_corrections = None

        # Stats: prediction vs bit-allocation alignment (LONG/START/STOP only)
        if current_block_type != SHORT:
            block_has_bits = any(
                np.any(np.array(bitAlloc[iCh]) > 0)
                for iCh in range(codingParams.nChannels)
            )
            if block_has_bits:
                codingParams._stat_bits_blocks += 1
                if pred_channels:
                    codingParams._stat_pred_bits_blocks += 1
            for iCh in pred_channels:
                ba = np.array(bitAlloc[iCh])
                ef = enable_flags_list[iCh]
                if len(ef) > 0:
                    alloc_mask = ba > 0
                    if np.any(alloc_mask):
                        codingParams._stat_pred_alloc_frac_sum += float(
                            np.mean(ef[alloc_mask])
                        )
                        codingParams._stat_pred_alloc_count += 1
                    aq = alpha_qs[iCh]
                    if hasattr(aq, "__len__"):
                        enabled_gains = np.array(aq)[ef] if np.any(ef) else np.array([])
                    else:
                        enabled_gains = (
                            np.array([abs(float(aq))]) if np.any(ef) else np.array([])
                        )
                    if len(enabled_gains) > 0:
                        codingParams._stat_alpha_sum += float(np.mean(enabled_gains))
                        codingParams._stat_alpha_count += 1

        # Encoder-side decode: reconstruct lossy output and store in search buffer so that
        # future prediction searches see the same signal the decoder will have.
        halfN_short = codingParams.nMDCTLines_short
        decoded_channels = []
        for iCh in range(codingParams.nChannels):
            if current_block_type != SHORT:
                mantissa_full = codec.ExpandMantissa(
                    mantissa[iCh], bitAlloc[iCh], codingParams.sfBands, halfN
                )
                if pred_mdcts[iCh] is not None:
                    if COMPLEX_PREDICTION:
                        mdst_pred = pred_mdsts[iCh]
                        scaled_pred = np.zeros(halfN)
                        _aq = alpha_qs[iCh]  # per-band gain array
                        _pi = phase_idxs[iCh]  # per-band phase index array
                        for iBand in range(codingParams.sfBands.nBands):
                            if enable_flags_list[iCh][iBand]:
                                lo = codingParams.sfBands.lowerLine[iBand]
                                hi = codingParams.sfBands.upperLine[iBand] + 1
                                a_b = _aq[iBand] * np.cos(
                                    phase_idx_to_radians(_pi[iBand])
                                )
                                b_b = _aq[iBand] * np.sin(
                                    phase_idx_to_radians(_pi[iBand])
                                )
                                scaled_pred[lo:hi] = (
                                    a_b * pred_mdcts[iCh][lo:hi]
                                    - b_b * mdst_pred[lo:hi]
                                )
                    else:
                        scaled_pred = (
                            alpha_qs[iCh] * pred_mdcts[iCh] * enable_masks[iCh]
                        )
                else:
                    scaled_pred = None
                decodedData = codec.Decode(
                    scaleFactor[iCh],
                    bitAlloc[iCh],
                    mantissa_full,
                    overallScaleFactor[iCh],
                    codingParams,
                    mdct_pred=scaled_pred,
                )
            else:  # SHORT block (START/STOP use the LONG path above)
                full_mant = [
                    codec.ExpandMantissa(
                        mantissa[iCh][i], bitAlloc[iCh][i], sfBands_short, halfN_short
                    )
                    for i in range(N_SHORT_BLOCKS)
                ]
                decodedData = codec.Decode(
                    scaleFactor[iCh],
                    bitAlloc[iCh],
                    full_mant,
                    overallScaleFactor[iCh],
                    codingParams,
                )
            decoded_channels.append(decodedData)

        # Convert M/S decoded data back to L/R before storing in search buffer
        if use_ms and codingParams.nChannels == 2:
            M_hat, S_hat = decoded_channels[0], decoded_channels[1]
            lr_for_buffer = [M_hat + S_hat, M_hat - S_hat]
        else:
            lr_for_buffer = decoded_channels
        for iCh in range(codingParams.nChannels):
            update_search_buffer(
                codingParams.search_buffer[iCh], lr_for_buffer[iCh], halfN, N
            )
        codingParams.buffer_fill = min(
            codingParams.buffer_fill + halfN, len(codingParams.search_buffer[0])
        )

        # Write bitstream per channel:
        #   [2b block_type | variable-width pred_type | 1b ms_flag (ch0 only)]
        #   [1b sign + 10b offset + 3b gain + 4b phase per enabled band (if COMPLEX_PREDICTION) + nBands enables (if pred active)]
        #   [7b grouping_mask (if SHORT)]
        #   [nScaleBits ovs | nMantSizeBits+nScaleBits per band | entropy-coded mantissas]
        for iCh in range(codingParams.nChannels):
            entropy_pbs = []
            nBits = 2 + _pred_type_bits  # block type + prediction type
            if iCh == 0 and codingParams.nChannels == 2:
                nBits += 1  # M/S flag
            if ranges[iCh] is not None:
                nBits += 11  # sign + offset
                if not COMPLEX_PREDICTION:
                    nBits += 4  # global scalar gain
                if codingParams.blockType != SHORT:
                    nBits += (
                        codingParams.sfBands.nBands
                    )  # per-band enables for LONG/START/STOP
                    if COMPLEX_PREDICTION:
                        nBits += 7 * int(
                            np.sum(enable_flags_list[iCh])
                        )  # 3b gain + 4b phase per enabled band
            if codingParams.blockType == SHORT and codingParams.group_lens is not None:
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
                            mantissa[iCh][sub_idx + g],
                            bitAlloc[iCh][sub_idx + g],
                            sfBands_short,
                        )
                        entropy_pbs.append(entropy_pb)
                        nBits += codingParams.nScaleBits  # overallScaleFactor
                        nBits += entropy_pb.nBits
                    sub_idx += G
                # Update SHORT entropy compression ratio (EMA)
                if VARIABLE_BIT_RATE:
                    _raw_short = sum(
                        bitAlloc[iCh][i][iBand] * sfBands_short.nLines[iBand]
                        for i in range(N_SHORT_BLOCKS)
                        for iBand in range(sfBands_short.nBands)
                        if bitAlloc[iCh][i][iBand] > 0
                    )
                    _ent_short = sum(ep.nBits for ep in entropy_pbs)
                    if _raw_short > 0:
                        _r = _ent_short / _raw_short
                        codingParams._entropy_ratio_short = (
                            0.9 * codingParams._entropy_ratio_short + 0.1 * _r
                        )
                        codingParams._entropy_inflation_short = min(
                            2.5, max(0.5, 1.0 / codingParams._entropy_ratio_short)
                        )
            else:
                entropy_pb = codingParams.entropyCoder_long.encode_block(
                    mantissa[iCh], bitAlloc[iCh], codingParams.sfBands
                )
                nBits += codingParams.nScaleBits
                for iBand in range(codingParams.sfBands.nBands):
                    nBits += codingParams.nMantSizeBits + codingParams.nScaleBits
                nBits += entropy_pb.nBits
                # Update LONG entropy compression ratio (EMA)
                if VARIABLE_BIT_RATE:
                    _raw_long = sum(
                        bitAlloc[iCh][iBand] * codingParams.sfBands.nLines[iBand]
                        for iBand in range(codingParams.sfBands.nBands)
                        if bitAlloc[iCh][iBand] > 0
                    )
                    if _raw_long > 0:
                        _r = entropy_pb.nBits / _raw_long
                        codingParams._entropy_ratio = (
                            0.9 * codingParams._entropy_ratio + 0.1 * _r
                        )
                        codingParams._entropy_inflation = min(
                            2.5, max(0.5, 1.0 / codingParams._entropy_ratio)
                        )

            nBytes = (nBits + BYTESIZE - 1) // BYTESIZE
            self.fp.write(pack("<L", int(nBytes)))

            pb = PackedBits()
            pb.Size(nBytes)
            pb.WriteBits(codingParams.blockType, 2)
            pb.WriteBits(PRED_MAP[ranges[iCh]], _pred_type_bits)
            if iCh == 0 and codingParams.nChannels == 2:
                pb.WriteBits(1 if use_ms else 0, 1)
            if ranges[iCh] is not None:
                sign = 1 if offsets[iCh] < 0 else 0
                pb.WriteBits(sign, 1)
                pb.WriteBits(abs(offsets[iCh]), 10)
                if not COMPLEX_PREDICTION:
                    pb.WriteBits(alpha_idxs[iCh], 4)
                if codingParams.blockType != SHORT:
                    for iBand in range(codingParams.sfBands.nBands):
                        pb.WriteBits(1 if enable_flags_list[iCh][iBand] else 0, 1)
                    if COMPLEX_PREDICTION:
                        for iBand in range(codingParams.sfBands.nBands):
                            if enable_flags_list[iCh][iBand]:
                                pb.WriteBits(int(alpha_idxs[iCh][iBand]), 3)
                                pb.WriteBits(int(phase_idxs[iCh][iBand]), 4)

            if codingParams.blockType == SHORT:
                pb.WriteBits(codingParams.grouping_mask, 7)
                ep_idx = 0
                sub_idx = 0
                for G in group_lens:
                    # Write shared ba/sf once per group (use sub_idx; all in group are identical)
                    for iBand in range(sfBands_short.nBands):
                        ba = bitAlloc[iCh][sub_idx][iBand]
                        if ba:
                            ba -= 1
                        pb.WriteBits(ba, codingParams.nMantSizeBits)
                        pb.WriteBits(
                            scaleFactor[iCh][sub_idx][iBand], codingParams.nScaleBits
                        )
                    # Write per-sub-block overallScale + entropy mantissas
                    for g in range(G):
                        pb.WriteBits(
                            overallScaleFactor[iCh][sub_idx + g],
                            codingParams.nScaleBits,
                        )
                        pb.WriteBits(
                            entropy_pbs[ep_idx].buffer, entropy_pbs[ep_idx].nBits
                        )
                        ep_idx += 1
                    sub_idx += G
            else:
                pb.WriteBits(overallScaleFactor[iCh], codingParams.nScaleBits)
                for iBand in range(codingParams.sfBands.nBands):
                    ba = bitAlloc[iCh][iBand]
                    if ba:
                        ba -= 1
                    pb.WriteBits(ba, codingParams.nMantSizeBits)
                    pb.WriteBits(scaleFactor[iCh][iBand], codingParams.nScaleBits)
                pb.WriteBits(entropy_pb.buffer, entropy_pb.nBits)
            self.fp.write(pb.GetPackedData())
        return

    def Close(self, codingParams):
        """
        Flushes the last data block through the encoding process (if encoding)
        and closes the audio file
        """
        # determine if encoding or encoding and, if encoding, do last block
        if self.fp.mode == "wb":  # we are writing to the PACFile, must be encode
            # we are writing the coded file -- pass a block of zeros to move last data block to other side of MDCT block
            data = [
                np.zeros(codingParams.nMDCTLines)
                for _ in range(codingParams.nChannels)
            ]
            self.WriteDataBlock(data, codingParams)
            total = codingParams._stat_total_blocks
            pred = codingParams._stat_pred_blocks
            ms = codingParams._stat_ms_blocks
            bits_blocks = codingParams._stat_bits_blocks
            pred_bits_blocks = codingParams._stat_pred_bits_blocks
            avg_band_frac = (
                (codingParams._stat_band_frac_sum / pred * 100) if pred > 0 else 0.0
            )
            avg_alloc_frac = (
                (
                    codingParams._stat_pred_alloc_frac_sum
                    / codingParams._stat_pred_alloc_count
                    * 100
                )
                if codingParams._stat_pred_alloc_count > 0
                else 0.0
            )
            avg_alpha = (
                (codingParams._stat_alpha_sum / codingParams._stat_alpha_count)
                if codingParams._stat_alpha_count > 0
                else 0.0
            )
            print("\n--- RMC encoding stats ---")
            print(
                f"  Blocks with prediction : {pred} / {total} ({100 * pred / total:.1f}%)"
            )
            if bits_blocks > 0:
                print(
                    f"  Pred % of non-silent   : {pred_bits_blocks} / {bits_blocks} ({100 * pred_bits_blocks / bits_blocks:.1f}%)"
                )
            print(
                f"  Avg bands predicted    : {avg_band_frac:.1f}% of all bands (when active)"
            )
            if codingParams._stat_pred_alloc_count > 0:
                print(
                    f"  Pred coverage of alloc : {avg_alloc_frac:.1f}% (pred-enabled / allocated bands)"
                )
            if codingParams._stat_alpha_count > 0:
                print(f"  Avg prediction gain    : {avg_alpha:.3f}")
            if codingParams.nChannels == 2:
                print(
                    f"  Blocks with M/S stereo : {ms} / {total} ({100 * ms / total:.1f}%)"
                )
            print(
                f"  Entropy inflation (LONG) : {codingParams._entropy_inflation:.2f}x (ratio={codingParams._entropy_ratio:.3f})"
            )
            print(
                f"  Entropy inflation (SHORT): {codingParams._entropy_inflation_short:.2f}x (ratio={codingParams._entropy_ratio_short:.3f})"
            )
        self.fp.close()

    def Encode(self, data, codingParams):
        """
        Encodes multichannel audio data and returns a tuple containing
        the scale factors, mantissa bit allocations, quantized mantissas,
        and the overall scale factor for each channel.
        """
        # Passes encoding logic to the Encode function defined in the codec module
        return codec.Encode(data, codingParams)

    def Decode(
        self,
        scaleFactor,
        bitAlloc,
        mantissa,
        overallScaleFactor,
        codingParams,
        mdct_pred=None,
    ):
        """
        Decodes a single audio channel of data based on the values of its scale factors,
        bit allocations, quantized mantissas, and overall scale factor.
        """
        # Passes decoding logic to the Decode function defined in the codec module
        return codec.Decode(
            scaleFactor, bitAlloc, mantissa, overallScaleFactor, codingParams, mdct_pred
        )


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    from prepare_materials import rmc

    elapsed = time.time()
    rmc("inputs/Brooklyn.wav", "Brooklyn_96.wav", rate_kb=96)
    print(f"\nDone in {time.time() - elapsed:.1f}s")
