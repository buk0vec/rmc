"""
rmcfile.py -- RMC audio codec file format handler.

Extends the baseline PAC codec with block switching, rhythmic prediction,
context-adaptive entropy coding, and adaptive bit budgeting.
See RMC.md for full technical reference.

Based on pacfile.py © 2009-2026 Marina Bosi & Richard E. Goldberg
"""

import bisect

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
    K_ATTACK_MAX,
    LONG,
    MEDIUM,
    SHORT,
    START,
    STOP,
    DesignSFBands,
    ShortBlockSFBands,
    TransitionSFBands,
    WindowForBlockType,
    plan_cascade,
)
from entropy import BlockEntropyCoder, RawMantissaCoder
from features import RMCFeatures
from mdct import IMDCT, MDCT
from psychoac import (  # defines the grouping of MDCT lines into scale factor bands
    AssignMDCTLinesFromFreqLimits,
    ScaleFactorBands,
)
from search import (
    COMPLEX_GAIN_TABLE,
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

    def __init__(self, filename, features: RMCFeatures = RMCFeatures()):
        """Object is initialized with its filename"""
        self.filename = filename
        self.features = features

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
        # AC-2A block switching: 128-sample SHORT sub-blocks (nMDCTLines / 16)
        myParams.nMDCTLines_short = nMDCTLines // 16
        myParams.sfBands_short = ShortBlockSFBands(
            myParams.nMDCTLines_short, sampleRate
        )
        _nMDCTLines_trans = (nMDCTLines + nMDCTLines // 16) // 2
        myParams.nMDCTLines_trans = _nMDCTLines_trans
        myParams.sfBands_trans = DesignSFBands(_nMDCTLines_trans, sampleRate)
        myParams.prevBlockType = LONG
        myParams.blockType = LONG
        myParams.block_queue = []
        myParams.cascade_a = nMDCTLines
        myParams.cascade_b = nMDCTLines
        myParams.transientPositions = []  # populated by encoder; decoder doesn't need it
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
        buf_size = myParams.numSamples4Bar + nMDCTLines + myParams.search_range
        myParams.search_buffer = [np.zeros(buf_size) for _ in range(myParams.nChannels)]
        myParams.buffer_fill = 0  # samples of real audio accumulated so far
        myParams.k_attack_for_stop = (
            0  # k_attack from most recent START block; used for AC-2A STOP window
        )

        # entropy coders
        myParams.entropyCoder_long = (
            BlockEntropyCoder(14)
            if self.features.ENTROPY_CODING
            else RawMantissaCoder()
        )
        myParams.entropyCoder_short = (
            BlockEntropyCoder(14)
            if self.features.ENTROPY_CODING
            else RawMantissaCoder()
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
        halfN_short = codingParams.nMDCTLines_short
        N_short = 2 * halfN_short
        sfBands_short = codingParams.sfBands_short

        raw_decoded = []
        for iCh in range(codingParams.nChannels):
            # Read packed header: 1 byte always, then 1 more for SHORT or 3 more for others.
            #   SHORT : 2 bytes = [3b type | 13b nBytes] (big-endian)
            #   others: 4 bytes = [3b type | 29b nBytes] (big-endian)
            hdr1 = self.fp.read(1)
            if not hdr1:
                if codingParams.overlapAndAdd:
                    overlapAndAdd = codingParams.overlapAndAdd
                    codingParams.overlapAndAdd = 0
                    return overlapAndAdd
                else:
                    return
            b0 = hdr1[0]
            block_type_hdr = (b0 >> 5) & 0x7
            if block_type_hdr == SHORT:
                b1 = self.fp.read(1)
                if not b1:
                    raise RuntimeError("Truncated SHORT frame header")
                full = (b0 << 8) | b1[0]
                nBytes = full & 0x1FFF  # low 13 bits
            else:
                rest = self.fp.read(3)
                if len(rest) < 3:
                    raise RuntimeError("Truncated frame header")
                full = (b0 << 24) | (rest[0] << 16) | (rest[1] << 8) | rest[2]
                nBytes = full & 0x1FFFFFFF  # low 29 bits
            pb = PackedBits()
            pb.SetPackedData(self.fp.read(nBytes))
            if pb.nBytes < nBytes:
                raise RuntimeError("Only read a partial block of coded PACFile data")

            codingParams.blockType = block_type_hdr  # from packed header, not pb
            if codingParams.blockType == START:
                k_attack_read = pb.ReadBits(4)
                codingParams.k_attack_for_stop = k_attack_read
                b_start = (15 + k_attack_read) * halfN_short  # recover exact b_start
                medium_lead = plan_cascade(k_attack_read, halfN_short)
                codingParams.cascade_a = halfN
                codingParams.cascade_b = b_start
                # Queue only lead MEDIUMs; STOP is always standard (cascade_a=128)
                codingParams.block_queue = [
                    {"type": MEDIUM, "a": ma, "b": mb} for (ma, mb) in medium_lead
                ]
            elif codingParams.blockType == MEDIUM:
                # Pop cascade dims from queue only on first channel (shared state)
                if iCh == 0 and getattr(codingParams, "block_queue", []):
                    qitem = codingParams.block_queue.pop(0)
                    codingParams.cascade_a = qitem["a"]
                    codingParams.cascade_b = qitem["b"]
                elif iCh > 0:
                    pass  # cascade_a/b already set from iCh==0 pop
                else:
                    # Fallback: infer from OAA tail
                    codingParams.cascade_a = len(codingParams.overlapAndAdd[iCh])
                    codingParams.cascade_b = codingParams.nMDCTLines_short
            if codingParams.blockType == STOP:
                codingParams.cascade_a = halfN_short  # always standard STOP
                codingParams.cascade_b = halfN
            # Per-block sfBands for prediction — must match what the encoder wrote.
            if codingParams.blockType in (START, STOP, MEDIUM):
                _halfN_pred_dec = (codingParams.cascade_a + codingParams.cascade_b) // 2
                sfBands_pred_dec = DesignSFBands(
                    _halfN_pred_dec, codingParams.sampleRate
                )
            else:
                sfBands_pred_dec = codingParams.sfBands
            # Quarter/half/bar/2bar/4bar — always 3 bits
            pred_type = pb.ReadBits(3)

            pred_offset = 0
            pred_alpha_qs = None
            pred_phase_qs = None
            pred_enable_flags = None
            # If prediction is enabled for this block
            if pred_type != PRED_MAP[None]:
                # Read the offset: sign (1 bit) + magnitude (10 bits)
                pred_sign = pb.ReadBits(1)
                pred_offset = pb.ReadBits(10)
                if pred_sign == 1:
                    pred_offset *= -1
                if (
                    codingParams.blockType != SHORT
                ):  # we only predict for LONG/START/STOP/MEDIUM blocks
                    # Read the enable flags — count matches sfBands_pred_dec.nBands (per-block)
                    pred_enable_flags = [
                        bool(pb.ReadBits(1)) for _ in range(sfBands_pred_dec.nBands)
                    ]
                    pred_alpha_qs = np.ones(sfBands_pred_dec.nBands)
                    pred_phase_qs = np.zeros(sfBands_pred_dec.nBands)
                    # Magnitude and phases per SFB
                    for iBand in range(sfBands_pred_dec.nBands):
                        if pred_enable_flags[iBand]:
                            pred_alpha_qs[iBand] = COMPLEX_GAIN_TABLE[pb.ReadBits(3)]
                            pred_phase_qs[iBand] = phase_idx_to_radians(pb.ReadBits(4))
            if codingParams.blockType == SHORT:
                # AC-2A single SHORT: no grouping mask
                overallScaleFactor = pb.ReadBits(codingParams.nScaleBits)
                scaleFactor = []
                bitAlloc = []
                for _ in range(sfBands_short.nBands):
                    ba = pb.ReadBits(codingParams.nMantSizeBits)
                    if ba:
                        ba += 1
                    bitAlloc.append(ba)
                    scaleFactor.append(pb.ReadBits(codingParams.nScaleBits))
                mantissa = codingParams.entropyCoder_short.decode_block(
                    pb, bitAlloc, sfBands_short, codingParams.nMDCTLines_short
                )
            elif codingParams.blockType == MEDIUM:
                # Cascade intermediate block
                ca = codingParams.cascade_a
                cb = codingParams.cascade_b
                halfN_med = (ca + cb) // 2
                sfBands_med = DesignSFBands(halfN_med, codingParams.sampleRate)
                overallScaleFactor = pb.ReadBits(codingParams.nScaleBits)
                scaleFactor = []
                bitAlloc = []
                for _ in range(sfBands_med.nBands):
                    ba = pb.ReadBits(codingParams.nMantSizeBits)
                    if ba:
                        ba += 1
                    bitAlloc.append(ba)
                    scaleFactor.append(pb.ReadBits(codingParams.nScaleBits))
                mantissa = codingParams.entropyCoder_long.decode_block(
                    pb, bitAlloc, sfBands_med, halfN_med
                )
            # LONG/START/STOP blocks
            else:
                overallScaleFactor = pb.ReadBits(codingParams.nScaleBits)
                scaleFactor = []
                bitAlloc = []
                if codingParams.blockType in (START, STOP):
                    ca = getattr(codingParams, "cascade_a", halfN)
                    cb = getattr(
                        codingParams, "cascade_b", codingParams.nMDCTLines_short
                    )
                    _halfN_dec = (ca + cb) // 2
                    if _halfN_dec != codingParams.nMDCTLines_trans:
                        _sfb_dec = DesignSFBands(_halfN_dec, codingParams.sampleRate)
                    else:
                        _sfb_dec = codingParams.sfBands_trans
                else:
                    _sfb_dec = codingParams.sfBands
                    _halfN_dec = codingParams.nMDCTLines
                # Read in scale factor and mantissa bit count per band
                for _ in range(_sfb_dec.nBands):
                    ba = pb.ReadBits(codingParams.nMantSizeBits)
                    if ba:
                        ba += 1
                    bitAlloc.append(ba)
                    scaleFactor.append(pb.ReadBits(codingParams.nScaleBits))
                # Decode block w/ long block entropy coder to get mantissas
                mantissa = codingParams.entropyCoder_long.decode_block(
                    pb, bitAlloc, _sfb_dec, _halfN_dec
                )

            # Compute prediction signal from the search buffer.
            mdct_P = None
            if pred_type != PRED_MAP[None] and codingParams.blockType != SHORT:
                start_offset = pred_type_to_samples(
                    PRED_MAP_REV[pred_type], codingParams
                )
                # Determine block-specific dimensions for prediction reconstruction
                if codingParams.blockType in (START, STOP, MEDIUM):
                    ca_dec = codingParams.cascade_a
                    cb_dec = codingParams.cascade_b
                    N_t_dec = ca_dec + cb_dec
                    halfN_t_dec = N_t_dec // 2
                    _cascade_a_dec = (
                        ca_dec if codingParams.blockType in (STOP, MEDIUM) else None
                    )
                    _cascade_b_dec = (
                        cb_dec if codingParams.blockType in (START, MEDIUM) else None
                    )
                else:
                    ca_dec = cb_dec = halfN
                    N_t_dec = N
                    halfN_t_dec = halfN
                    _cascade_a_dec = _cascade_b_dec = None
                buf = codingParams.search_buffer[iCh]
                # Calculate start of prediction block using halfN_t_dec offset
                seg_start = len(buf) - start_offset + pred_offset - halfN_t_dec
                # Safety check
                if 0 <= seg_start and seg_start + N_t_dec <= len(buf):
                    # Pull out candidate, window and take MCLT (return_complex=True)
                    candidate = buf[seg_start : seg_start + N_t_dec].copy()
                    window = WindowForBlockType(
                        codingParams.blockType,
                        N,
                        N_short,
                        cascade_a=_cascade_a_dec,
                        cascade_b=_cascade_b_dec,
                    )
                    mdct_P_raw_cplx = MDCT(
                        window * candidate, ca_dec, cb_dec, return_complex=True
                    )[:halfN_t_dec]
                    mdct_P_raw = mdct_P_raw_cplx.real
                    mdst_P_raw = mdct_P_raw_cplx.imag
                    effective_pred = np.zeros(halfN_t_dec)
                    if (
                        pred_enable_flags is not None
                        and pred_alpha_qs is not None
                        and pred_phase_qs is not None
                    ):
                        for iBand, enabled in enumerate(pred_enable_flags):
                            if enabled:
                                lo = sfBands_pred_dec.lowerLine[iBand]
                                hi = sfBands_pred_dec.upperLine[iBand] + 1
                                a_b = pred_alpha_qs[iBand] * np.cos(
                                    pred_phase_qs[iBand]
                                )
                                b_b = pred_alpha_qs[iBand] * np.sin(
                                    pred_phase_qs[iBand]
                                )
                                effective_pred[lo:hi] = (
                                    a_b * mdct_P_raw[lo:hi] - b_b * mdst_P_raw[lo:hi]
                                )
                    mdct_P = effective_pred

            decodedData = self.Decode(
                scaleFactor,
                bitAlloc,
                mantissa,
                overallScaleFactor,
                codingParams,
                mdct_pred=mdct_P,
            )

            raw_decoded.append(decodedData)

        # Overlap-and-add
        data = []
        for iCh in range(codingParams.nChannels):
            decodedData = raw_decoded[iCh]
            ola_size = len(codingParams.overlapAndAdd[iCh])
            out = np.maximum(
                -1.0,
                np.minimum(
                    1.0, np.add(codingParams.overlapAndAdd[iCh], decodedData[:ola_size])
                ),
            )
            data.append(out)
            codingParams.overlapAndAdd[iCh] = decodedData[ola_size:]

        # Buffer update: search buffer stores L/R reconstructed audio
        if codingParams.blockType == SHORT:
            _halfN_buf = codingParams.nMDCTLines_short
        elif codingParams.blockType in (START, STOP, MEDIUM):
            ca = getattr(codingParams, "cascade_a", halfN)
            cb = getattr(codingParams, "cascade_b", codingParams.nMDCTLines_short)
            _halfN_buf = (ca + cb) // 2
        else:
            _halfN_buf = halfN
        _N_buf = 2 * _halfN_buf
        for iCh in range(codingParams.nChannels):
            update_search_buffer(
                codingParams.search_buffer[iCh], raw_decoded[iCh], _halfN_buf, _N_buf
            )
        codingParams.buffer_fill = min(
            codingParams.buffer_fill + _halfN_buf, len(codingParams.search_buffer[0])
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
        # AC-2A block switching: 128-sample SHORT sub-blocks (nMDCTLines / 16)
        codingParams.nMDCTLines_short = codingParams.nMDCTLines // 16
        codingParams.sfBands_short = ShortBlockSFBands(
            codingParams.nMDCTLines_short, codingParams.sampleRate
        )
        _nMDCTLines_trans = (
            codingParams.nMDCTLines + codingParams.nMDCTLines_short
        ) // 2
        codingParams.nMDCTLines_trans = _nMDCTLines_trans
        codingParams.sfBands_trans = DesignSFBands(
            _nMDCTLines_trans, codingParams.sampleRate
        )
        codingParams.currentSamplePos = 0
        codingParams.short_blocks_remaining = 0
        # Pre-set nSamplesPerBlock for the first PCM read
        _positions0 = getattr(codingParams, "transientPositions", [])
        if _positions0:
            _raw0 = _positions0[0]
            if _raw0 < 2 * codingParams.nMDCTLines:
                _k0 = max(
                    0,
                    min(
                        K_ATTACK_MAX,
                        (_raw0 - codingParams.nMDCTLines)
                        // codingParams.nMDCTLines_short,
                    ),
                )
                _bs0 = (15 + _k0) * codingParams.nMDCTLines_short
                codingParams.nSamplesPerBlock = _bs0
        codingParams.blockType = LONG
        codingParams.block_queue = []
        codingParams.cascade_a = codingParams.nMDCTLines
        codingParams.cascade_b = codingParams.nMDCTLines
        codingParams.entropyCoder_long = (
            BlockEntropyCoder(14)
            if self.features.ENTROPY_CODING
            else RawMantissaCoder()
        )
        codingParams.entropyCoder_short = (
            BlockEntropyCoder(14)
            if self.features.ENTROPY_CODING
            else RawMantissaCoder()
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
            codingParams.numSamples4Bar
            + codingParams.nMDCTLines
            + codingParams.search_range
        )
        codingParams.search_buffer = [
            np.zeros(buf_size) for _ in range(codingParams.nChannels)
        ]
        codingParams.buffer_fill = 0  # samples of real audio accumulated so far
        # Coding pool (bit reservoir): per-channel accumulated surplus
        codingParams.bit_pool = [0] * codingParams.nChannels
        codingParams._pool_draws = [0] * codingParams.nChannels

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
        codingParams.k_attack_for_stop = 0
        # prediction stats
        codingParams._stat_total_blocks = 0
        codingParams._stat_pred_blocks = 0
        codingParams._stat_band_frac_sum = 0.0
        # per-block-type counters: {block_type_int: count}
        codingParams._stat_total_by_type = {}
        codingParams._stat_pred_by_type = {}
        # per-block-type band enable: {block_type_int: (sum_of_fracs, count)}
        codingParams._stat_band_frac_by_type = {}
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
        # Band enable rate stats for ranges 1-5, 1-10, 1-15, 1-20
        codingParams._stat_band_1_5_sum = 0.0
        codingParams._stat_band_1_5_count = 0
        codingParams._stat_band_1_10_sum = 0.0
        codingParams._stat_band_1_10_count = 0
        codingParams._stat_band_1_15_sum = 0.0
        codingParams._stat_band_1_15_count = 0
        codingParams._stat_band_1_20_sum = 0.0
        codingParams._stat_band_1_20_count = 0
        # Defaults for per-encode-pass state (set properly before second encode)
        codingParams.masking_signals = None
        codingParams.mdct_pred_corrections = None
        codingParams.block_overhead = None
        return

    def WriteDataBlock(self, data, codingParams):
        """
        Writes a block of signed-fraction data to a PACFile object that has
        already executed OpenForWriting()"""

        sfBands_short = codingParams.sfBands_short
        halfN = codingParams.nMDCTLines
        N = 2 * halfN
        halfN_short = codingParams.nMDCTLines_short
        N_short = 2 * halfN_short
        N_trans = halfN + halfN_short  # 1152 for AC-2A transitions

        # AC-2A: determine block type and cascade dimensions before building analysis windows
        # Find the next transient within 2 * halfN samples using exact positions
        _positions = getattr(codingParams, "transientPositions", [])
        _pos = codingParams.currentSamplePos
        _tidx = bisect.bisect_left(_positions, _pos)
        if _tidx < len(_positions) and _positions[_tidx] < _pos + 2 * halfN:
            _raw_offset = _positions[_tidx] - _pos
            _k_encoded = max(0, min(K_ATTACK_MAX, (_raw_offset - halfN) // halfN_short))
            _b_start = (15 + _k_encoded) * halfN_short
            k_attack = _k_encoded
        else:
            _b_start = halfN_short
            k_attack = -1

        _prev = codingParams.prevBlockType
        _rem = codingParams.short_blocks_remaining

        if codingParams.block_queue:
            # Pop next planned MEDIUM or STOP block from cascade queue
            qitem = codingParams.block_queue.pop(0)
            _bt = qitem["type"]
            codingParams.cascade_a = qitem["a"]
            codingParams.cascade_b = qitem["b"]
        elif _prev in (LONG, STOP):
            if k_attack >= 0:
                _bt = START
                medium_lead = plan_cascade(_k_encoded, halfN_short)
                codingParams.cascade_a = halfN
                codingParams.cascade_b = _b_start
                for ma, mb in medium_lead:
                    codingParams.block_queue.append({"type": MEDIUM, "a": ma, "b": mb})
                codingParams.short_blocks_remaining = 1
                codingParams.k_attack_for_stop = _k_encoded
            else:
                _bt = LONG
                codingParams.cascade_a = halfN
                codingParams.cascade_b = halfN
        elif _prev in (START, MEDIUM):
            _bt = SHORT
            codingParams.cascade_a = halfN_short
            codingParams.cascade_b = halfN_short
        elif _prev == SHORT:
            # STOP is always queued when short_blocks_remaining hits 0,
            # so this branch is only reached when _rem > 0 (more SHORTs).
            _bt = SHORT
            codingParams.cascade_a = halfN_short
            codingParams.cascade_b = halfN_short
        else:  # STOP → LONG
            _bt = LONG
            codingParams.cascade_a = halfN
            codingParams.cascade_b = halfN

        codingParams.blockType = _bt
        codingParams.prevBlockType = _bt
        if _bt == SHORT:
            codingParams.short_blocks_remaining -= 1
            if codingParams.short_blocks_remaining == 0:
                # No tail mediums — STOP_std directly after SHORT
                codingParams.block_queue.append(
                    {"type": STOP, "a": halfN_short, "b": halfN}
                )
        codingParams.group_lens = None
        codingParams.grouping_mask = 0

        # Build full analysis windows per channel (rolling 1024-sample priorBlock for AC-2A)
        fullBlockData_ = []
        maskingData_ = []
        new_prior = []
        for iCh in range(codingParams.nChannels):
            all_samples = np.concatenate((codingParams.priorBlock[iCh], data[iCh]))
            # After START, keep b_start samples so MEDIUM's left overlap is fully covered
            _keep = (
                codingParams.cascade_b
                if codingParams.blockType == START and codingParams.cascade_b > halfN
                else halfN
            )
            new_prior.append(all_samples[-_keep:])
            _bt = codingParams.blockType
            if _bt == LONG:
                fullBlockData_.append(all_samples)
                maskingData_.append(all_samples)
            else:
                # MDCT uses truncated window; masking uses full all_samples for
                # better FFT resolution (1152 samples → 38 Hz/bin vs 172 Hz/bin for SHORT)
                N_block = codingParams.cascade_a + codingParams.cascade_b
                fullBlockData_.append(all_samples[-N_block:])
                maskingData_.append(all_samples)
        codingParams.priorBlock = new_prior

        current_block_type = codingParams.blockType

        # Per-block sfBands for prediction — sized to halfN_t, not LONG halfN.
        if self.features.PREDICTION and current_block_type != LONG:
            _halfN_t_pred = (codingParams.cascade_a + codingParams.cascade_b) // 2
            sfBands_pred = DesignSFBands(_halfN_t_pred, codingParams.sampleRate)
        else:
            sfBands_pred = codingParams.sfBands

        # Per-channel prediction search and residual computation.
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
            pred_signal = fullBlockData_[iCh]
            _pred_active = current_block_type == LONG or current_block_type in (
                START,
                STOP,
                MEDIUM,
            )
            if self.features.PREDICTION and _pred_active:
                # Determine block-specific MDCT dimensions
                if current_block_type != LONG:
                    ca_pred = codingParams.cascade_a
                    cb_pred = codingParams.cascade_b
                    _cascade_a_arg = (
                        ca_pred if current_block_type in (STOP, MEDIUM) else None
                    )
                    _cascade_b_arg = (
                        cb_pred if current_block_type in (START, MEDIUM) else None
                    )
                else:
                    ca_pred = cb_pred = halfN
                    _cascade_a_arg = _cascade_b_arg = None
                halfN_t = (ca_pred + cb_pred) // 2

                _window = WindowForBlockType(
                    current_block_type,
                    N,
                    N_short,
                    cascade_a=_cascade_a_arg,
                    cascade_b=_cascade_b_arg,
                )
                mdct_X = MDCT(_window * pred_signal, ca_pred, cb_pred)[:halfN_t]
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
                    pred_signal,
                    codingParams,
                    codingParams.search_buffer[iCh],
                    block_type=current_block_type,
                    cascade_a=ca_pred if current_block_type != LONG else None,
                    cascade_b=cb_pred if current_block_type != LONG else None,
                    sfBands=sfBands_pred,
                )
                # If we found a good enough region for prediction
                if range_type is not None:
                    # Per-band complex residual (vectorized over all bands)
                    _phases = phase_idx * (np.pi / 8.0) - np.pi
                    _a_vec = alpha_q * np.cos(_phases)
                    _b_vec = alpha_q * np.sin(_phases)
                    _a_line = np.repeat(_a_vec, sfBands_pred.nLines)
                    _b_line = np.repeat(_b_vec, sfBands_pred.nLines)
                    residual_full = mdct_X - _a_line * mdct_P + _b_line * mdst_P
                    # Per-band enable: apply prediction only where it reduces the signal
                    enable_f = np.zeros(sfBands_pred.nBands, dtype=bool)
                    enable_m = np.zeros(halfN_t)
                    for iBand in range(sfBands_pred.nBands):
                        if (
                            self.features.PRED_MAX_SFB is not None
                            and iBand >= self.features.PRED_MAX_SFB
                        ):
                            continue
                        lo = sfBands_pred.lowerLine[iBand]
                        hi = sfBands_pred.upperLine[iBand] + 1
                        orig_band = np.abs(mdct_X[lo:hi])
                        res_band = np.abs(residual_full[lo:hi])
                        orig_rms = (
                            np.sqrt(np.mean(orig_band**2)) if orig_band.size else 0.0
                        )
                        res_rms = (
                            np.sqrt(np.mean(res_band**2)) if res_band.size else 0.0
                        )
                        n_lines = hi - lo
                        if self.features.PRED_NLINES_THRESH:
                            threshold = 10.0 ** (-42.0 / (20.0 * n_lines))
                        else:
                            threshold = self.features.PRED_ENABLE_RATIO
                        enable = (
                            res_rms < threshold * orig_rms if orig_rms > 0 else False
                        )
                        if enable:
                            enable_f[iBand] = True
                            enable_m[lo:hi] = 1.0
                    if not np.any(enable_f):
                        (
                            range_type,
                            pcm_residual,
                            rel_offset,
                            mdct_P,
                            mdst_P,
                            alpha_idx,
                            alpha_q,
                            phase_idx,
                        ) = (None, pred_signal, 0, None, None, 0, 1.0, 128)
                        enable_f = np.zeros(sfBands_pred.nBands, dtype=bool)
                        enable_m = np.zeros(halfN_t)
                        correction = None
                    else:
                        # Per-band MDCT-domain correction: subtract prediction in enabled bands.
                        _a_vec_e = alpha_q * np.cos(_phases) * enable_f
                        _b_vec_e = alpha_q * np.sin(_phases) * enable_f
                        _a_line_e = np.repeat(_a_vec_e, sfBands_pred.nLines)
                        _b_line_e = np.repeat(_b_vec_e, sfBands_pred.nLines)
                        correction = -(_a_line_e * mdct_P - _b_line_e * mdst_P)
                else:
                    enable_f = np.zeros(sfBands_pred.nBands, dtype=bool)
                    enable_m = np.zeros(halfN_t)
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
                ) = (None, pred_signal, 0, None, None, 0, 1.0, 128)
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
        bt = current_block_type
        codingParams._stat_total_by_type[bt] = (
            codingParams._stat_total_by_type.get(bt, 0) + 1
        )
        pred_channels = [
            iCh for iCh in range(codingParams.nChannels) if ranges[iCh] is not None
        ]
        if pred_channels:
            codingParams._stat_pred_blocks += 1
            codingParams._stat_pred_by_type[bt] = (
                codingParams._stat_pred_by_type.get(bt, 0) + 1
            )
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

            # Band range enable rates (for channels with prediction)
            for iCh in pred_channels:
                ef = enable_flags_list[iCh]
                if len(ef) > 0:
                    s, c = codingParams._stat_band_frac_by_type.get(bt, (0.0, 0))
                    codingParams._stat_band_frac_by_type[bt] = (
                        s + float(np.mean(ef)),
                        c + 1,
                    )
                    # Bands 1-5 (indices 0-4)
                    max_idx = min(5, len(ef))
                    if max_idx > 0:
                        codingParams._stat_band_1_5_sum += float(np.mean(ef[:max_idx]))
                        codingParams._stat_band_1_5_count += 1
                    # Bands 1-10 (indices 0-9)
                    max_idx = min(10, len(ef))
                    if max_idx > 0:
                        codingParams._stat_band_1_10_sum += float(np.mean(ef[:max_idx]))
                        codingParams._stat_band_1_10_count += 1
                    # Bands 1-15 (indices 0-14)
                    max_idx = min(15, len(ef))
                    if max_idx > 0:
                        codingParams._stat_band_1_15_sum += float(np.mean(ef[:max_idx]))
                        codingParams._stat_band_1_15_count += 1
                    # Bands 1-20 (indices 0-19)
                    max_idx = min(20, len(ef))
                    if max_idx > 0:
                        codingParams._stat_band_1_20_sum += float(np.mean(ef[:max_idx]))
                        codingParams._stat_band_1_20_count += 1

        # Compute per-channel bitstream overhead not accounted for in base bit budget.
        # Frame header format: packed type+nBytes field (block_type NOT written inside pb).
        #   SHORT : 2 bytes = [3b type | 13b nBytes]  (max 8191 bytes; pool draws can push SHORT large)
        #   others: 4 bytes = [3b type | 29b nBytes]
        block_overhead = []
        for iCh in range(codingParams.nChannels):
            if current_block_type == SHORT:
                oh = 16 + 3  # 2-byte packed header + 3b pred type (type not in pb)
            else:
                oh = 32 + 3  # 4-byte packed header + 3b pred type (type not in pb)
            if current_block_type == START:
                oh += 4  # k_attack field
            if ranges[iCh] is not None:
                oh += 1 + 10  # sign + offset
                if current_block_type != SHORT:
                    oh += sfBands_pred.nBands  # per-band enable flags (per-block count)
                    oh += 7 * int(
                        np.sum(enable_flags_list[iCh])
                    )  # 3b gain + 4b phase per enabled band
            block_overhead.append(oh)

        # Encode the residual signals with original signal for psychoacoustic masking
        codingParams.masking_signals = maskingData_
        codingParams.mdct_pred_corrections = corrections
        codingParams.block_overhead = block_overhead
        # Pool: effective halfN for this block type and per-channel draw snapshot
        if current_block_type == SHORT:
            _pool_eff_halfN = codingParams.nMDCTLines_short
        elif current_block_type in (START, STOP, MEDIUM):
            _pool_eff_halfN = (codingParams.cascade_a + codingParams.cascade_b) // 2
        else:
            _pool_eff_halfN = halfN
        # All block types draw from the pool
        codingParams._pool_draws = [
            codingParams.bit_pool[iCh] for iCh in range(codingParams.nChannels)
        ]
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
                if len(ef) > 0 and len(ba) == len(ef):
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
        halfN_trans = getattr(
            codingParams, "nMDCTLines_trans", (halfN + halfN_short) // 2
        )
        decoded_channels = []
        for iCh in range(codingParams.nChannels):
            if current_block_type == SHORT:
                mantissa_full = codec.ExpandMantissa(
                    mantissa[iCh], bitAlloc[iCh], sfBands_short, halfN_short
                )
                decodedData = codec.Decode(
                    scaleFactor[iCh],
                    bitAlloc[iCh],
                    mantissa_full,
                    overallScaleFactor[iCh],
                    codingParams,
                )
            elif current_block_type == MEDIUM:
                ca = codingParams.cascade_a
                cb = codingParams.cascade_b
                halfN_med = (ca + cb) // 2
                sfBands_med = DesignSFBands(halfN_med, codingParams.sampleRate)
                mantissa_full = codec.ExpandMantissa(
                    mantissa[iCh], bitAlloc[iCh], sfBands_med, halfN_med
                )
                if pred_mdcts[iCh] is not None:
                    _mdst_pred = pred_mdsts[iCh]
                    _aq = alpha_qs[iCh]
                    _pi = phase_idxs[iCh]
                    _ef = enable_flags_list[iCh]
                    _ph = _pi * (np.pi / 8.0) - np.pi
                    _av = _aq * np.cos(_ph) * _ef
                    _bv = _aq * np.sin(_ph) * _ef
                    scaled_pred = (
                        np.repeat(_av, sfBands_pred.nLines) * pred_mdcts[iCh]
                        - np.repeat(_bv, sfBands_pred.nLines) * _mdst_pred
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
            elif current_block_type in (START, STOP):
                ca = codingParams.cascade_a
                cb = codingParams.cascade_b
                halfN_used = (ca + cb) // 2
                sfBands_used = (
                    DesignSFBands(halfN_used, codingParams.sampleRate)
                    if halfN_used != halfN_trans
                    else codingParams.sfBands_trans
                )
                mantissa_full = codec.ExpandMantissa(
                    mantissa[iCh], bitAlloc[iCh], sfBands_used, halfN_used
                )
                if pred_mdcts[iCh] is not None:
                    _mdst_pred = pred_mdsts[iCh]
                    _aq = alpha_qs[iCh]
                    _pi = phase_idxs[iCh]
                    _ef = enable_flags_list[iCh]
                    _ph = _pi * (np.pi / 8.0) - np.pi
                    _av = _aq * np.cos(_ph) * _ef
                    _bv = _aq * np.sin(_ph) * _ef
                    scaled_pred = (
                        np.repeat(_av, sfBands_pred.nLines) * pred_mdcts[iCh]
                        - np.repeat(_bv, sfBands_pred.nLines) * _mdst_pred
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
            else:  # LONG
                mantissa_full = codec.ExpandMantissa(
                    mantissa[iCh], bitAlloc[iCh], codingParams.sfBands, halfN
                )
                if pred_mdcts[iCh] is not None:
                    mdst_pred = pred_mdsts[iCh]
                    _aq = alpha_qs[iCh]
                    _pi = phase_idxs[iCh]
                    _ef = enable_flags_list[iCh]
                    _ph = _pi * (np.pi / 8.0) - np.pi
                    _av = _aq * np.cos(_ph) * _ef
                    _bv = _aq * np.sin(_ph) * _ef
                    scaled_pred = (
                        np.repeat(_av, sfBands_pred.nLines) * pred_mdcts[iCh]
                        - np.repeat(_bv, sfBands_pred.nLines) * mdst_pred
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
            decoded_channels.append(decodedData)

        if current_block_type == SHORT:
            _halfN_buf = halfN_short
        elif current_block_type in (START, STOP, MEDIUM):
            _halfN_buf = (codingParams.cascade_a + codingParams.cascade_b) // 2
        else:
            _halfN_buf = halfN
        _N_buf = 2 * _halfN_buf
        for iCh in range(codingParams.nChannels):
            update_search_buffer(
                codingParams.search_buffer[iCh],
                decoded_channels[iCh],
                _halfN_buf,
                _N_buf,
            )
        codingParams.buffer_fill = min(
            codingParams.buffer_fill + _halfN_buf, len(codingParams.search_buffer[0])
        )

        # Write bitstream per channel:
        #   [3b block_type | 3b pred_type]
        #   [4b k_attack (START only)]
        #   [1b sign + 10b offset + nBands enables + 3b gain + 4b phase per enabled band (if pred active)]
        #   [nScaleBits ovs | nMantSizeBits+nScaleBits per band | entropy-coded mantissas]
        for iCh in range(codingParams.nChannels):
            nBits = 3 + 3  # block type + prediction type
            if codingParams.blockType == START:
                nBits += 4  # k_attack field
            if ranges[iCh] is not None:
                nBits += 1 + 10  # sign + offset
                if codingParams.blockType != SHORT:
                    nBits += sfBands_pred.nBands  # per-band enables (per-block count)
                    nBits += 7 * int(
                        np.sum(enable_flags_list[iCh])
                    )  # 3b gain + 4b phase per enabled band
            if codingParams.blockType == SHORT:
                # AC-2A single SHORT: overallScaleFactor + per-band ba/sf + entropy mantissa
                entropy_pb_short = codingParams.entropyCoder_short.encode_block(
                    mantissa[iCh], bitAlloc[iCh], sfBands_short
                )
                nBits += codingParams.nScaleBits  # overallScaleFactor
                for iBand in range(sfBands_short.nBands):
                    nBits += codingParams.nMantSizeBits + codingParams.nScaleBits
                nBits += entropy_pb_short.nBits
            elif codingParams.blockType == MEDIUM:
                # MEDIUM cascade block: same format as START/STOP (ovs + per-band sf/ba + mantissas)
                ca = codingParams.cascade_a
                cb = codingParams.cascade_b
                halfN_med = (ca + cb) // 2
                _sfb_med = DesignSFBands(halfN_med, codingParams.sampleRate)
                entropy_pb = codingParams.entropyCoder_long.encode_block(
                    mantissa[iCh], bitAlloc[iCh], _sfb_med
                )
                nBits += codingParams.nScaleBits
                for iBand in range(_sfb_med.nBands):
                    nBits += codingParams.nMantSizeBits + codingParams.nScaleBits
                nBits += entropy_pb.nBits
            else:
                # LONG and START/STOP (AC-2A)
                ca = getattr(codingParams, "cascade_a", halfN)
                cb = getattr(codingParams, "cascade_b", halfN)
                halfN_enc = (ca + cb) // 2
                _sfb_enc = (
                    DesignSFBands(halfN_enc, codingParams.sampleRate)
                    if codingParams.blockType in (START, STOP)
                    and halfN_enc != halfN_trans
                    else (
                        codingParams.sfBands_trans
                        if codingParams.blockType in (START, STOP)
                        else codingParams.sfBands
                    )
                )
                entropy_pb = codingParams.entropyCoder_long.encode_block(
                    mantissa[iCh], bitAlloc[iCh], _sfb_enc
                )
                nBits += codingParams.nScaleBits
                for iBand in range(_sfb_enc.nBands):
                    nBits += codingParams.nMantSizeBits + codingParams.nScaleBits
                nBits += entropy_pb.nBits

            # Pool update: recycle coding surplus into next frame's budget.
            # actual_ch accounts for packed header size (block_type lives in header, not pb).
            nominal_ch = int(codingParams.targetBitsPerSample * _pool_eff_halfN)
            if codingParams.blockType == SHORT:
                actual_ch = nBits + 16  # 2-byte packed [3b type | 13b nBytes]
            else:
                actual_ch = nBits + 32  # 4-byte packed [3b type | 29b nBytes]
            codingParams.bit_pool[iCh] = max(
                0, codingParams.bit_pool[iCh] + nominal_ch - actual_ch
            )
            nBytes = (nBits + BYTESIZE - 1) // BYTESIZE
            if codingParams.blockType == SHORT:
                # 2-byte header: [3b block_type | 13b nBytes] big-endian
                assert nBytes <= 8191, (
                    f"SHORT block nBytes={nBytes} exceeds 13-bit limit"
                )
                self.fp.write(pack(">H", (codingParams.blockType << 13) | nBytes))
            else:
                # 4-byte header: [3b block_type | 29b nBytes] big-endian
                self.fp.write(pack(">L", (codingParams.blockType << 29) | nBytes))

            pb = PackedBits()
            pb.Size(nBytes)
            # block_type is in the packed header — NOT written into pb
            if codingParams.blockType == START:
                pb.WriteBits(codingParams.k_attack_for_stop, 4)
            pb.WriteBits(PRED_MAP[ranges[iCh]], 3)
            if ranges[iCh] is not None:
                sign = 1 if offsets[iCh] < 0 else 0
                pb.WriteBits(sign, 1)
                pb.WriteBits(abs(offsets[iCh]), 10)
                if codingParams.blockType != SHORT:
                    for iBand in range(sfBands_pred.nBands):
                        pb.WriteBits(1 if enable_flags_list[iCh][iBand] else 0, 1)
                    for iBand in range(sfBands_pred.nBands):
                        if enable_flags_list[iCh][iBand]:
                            pb.WriteBits(int(alpha_idxs[iCh][iBand]), 3)
                            pb.WriteBits(int(phase_idxs[iCh][iBand]), 4)

            if codingParams.blockType == SHORT:
                # AC-2A single SHORT: overallScaleFactor + per-band ba/sf + entropy mantissa
                pb.WriteBits(overallScaleFactor[iCh], codingParams.nScaleBits)
                for iBand in range(sfBands_short.nBands):
                    ba = bitAlloc[iCh][iBand]
                    if ba:
                        ba -= 1
                    pb.WriteBits(ba, codingParams.nMantSizeBits)
                    pb.WriteBits(scaleFactor[iCh][iBand], codingParams.nScaleBits)
                pb.WriteBits(entropy_pb_short.buffer, entropy_pb_short.nBits)
            elif codingParams.blockType == MEDIUM:
                pb.WriteBits(overallScaleFactor[iCh], codingParams.nScaleBits)
                for iBand in range(_sfb_med.nBands):
                    ba = bitAlloc[iCh][iBand]
                    if ba:
                        ba -= 1
                    pb.WriteBits(ba, codingParams.nMantSizeBits)
                    pb.WriteBits(scaleFactor[iCh][iBand], codingParams.nScaleBits)
                pb.WriteBits(entropy_pb.buffer, entropy_pb.nBits)
            else:
                # LONG and START/STOP (AC-2A) — _sfb_enc already computed above
                pb.WriteBits(overallScaleFactor[iCh], codingParams.nScaleBits)
                for iBand in range(_sfb_enc.nBands):
                    ba = bitAlloc[iCh][iBand]
                    if ba:
                        ba -= 1
                    pb.WriteBits(ba, codingParams.nMantSizeBits)
                    pb.WriteBits(scaleFactor[iCh][iBand], codingParams.nScaleBits)
                pb.WriteBits(entropy_pb.buffer, entropy_pb.nBits)
            self.fp.write(pb.GetPackedData())

        # AC-2A: advance sample counter and pre-set nSamplesPerBlock for the next PCM read
        codingParams.currentSamplePos += len(data[0])
        _prev_now = codingParams.prevBlockType

        # Look ahead from new position for the next transient
        _pos_next = codingParams.currentSamplePos
        _positions = getattr(codingParams, "transientPositions", [])
        _tidx_next = bisect.bisect_left(_positions, _pos_next)
        if (
            _tidx_next < len(_positions)
            and _positions[_tidx_next] < _pos_next + 2 * halfN
        ):
            _raw_next = _positions[_tidx_next] - _pos_next
            _k_next = max(0, min(K_ATTACK_MAX, (_raw_next - halfN) // halfN_short))
            _b_next = (15 + _k_next) * halfN_short
            _k_next_valid = True
        else:
            _b_next = halfN
            _k_next_valid = False

        if codingParams.block_queue:
            # Next block is queued (MEDIUM or STOP) — its nSPB = b of that block
            codingParams.nSamplesPerBlock = codingParams.block_queue[0]["b"]
        elif _prev_now in (LONG, STOP):
            codingParams.nSamplesPerBlock = _b_next if _k_next_valid else halfN
        elif _prev_now in (START, MEDIUM):
            codingParams.nSamplesPerBlock = halfN_short
        elif _prev_now == SHORT:
            # STOP was just queued; block_queue is non-empty so this branch
            # is only reached if more SHORTs remain.
            codingParams.nSamplesPerBlock = halfN_short
        else:  # STOP → LONG
            codingParams.nSamplesPerBlock = halfN

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
                np.zeros(codingParams.nMDCTLines) for _ in range(codingParams.nChannels)
            ]
            self.WriteDataBlock(data, codingParams)
            total = codingParams._stat_total_blocks
            pred = codingParams._stat_pred_blocks
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
            _BT_NAMES = {
                LONG: "LONG",
                SHORT: "SHORT",
                START: "START",
                STOP: "STOP",
                MEDIUM: "MED",
            }
            print("\n--- RMC encoding stats ---")
            print(
                f"  Blocks with prediction : {pred} / {total} ({100 * pred / total:.1f}%)"
            )
            # Per-block-type: block enable rate
            for bt_val in (LONG, START, STOP, MEDIUM, SHORT):
                n_bt = codingParams._stat_total_by_type.get(bt_val, 0)
                if n_bt > 0:
                    p_bt = codingParams._stat_pred_by_type.get(bt_val, 0)
                    print(
                        f"    {_BT_NAMES[bt_val]:5s} block enable rate: {p_bt} / {n_bt} ({100 * p_bt / n_bt:.1f}%)"
                    )
            if bits_blocks > 0:
                print(
                    f"  Pred % of non-silent   : {pred_bits_blocks} / {bits_blocks} ({100 * pred_bits_blocks / bits_blocks:.1f}%)"
                )
            print(
                f"  Avg bands predicted    : {avg_band_frac:.1f}% of all bands (when active)"
            )
            # Per-block-type: band enable rate
            for bt_val in (LONG, START, STOP, MEDIUM):
                s_bt, c_bt = codingParams._stat_band_frac_by_type.get(bt_val, (0.0, 0))
                if c_bt > 0:
                    print(
                        f"    {_BT_NAMES[bt_val]:5s} avg band enable    : {100 * s_bt / c_bt:.1f}% (when active)"
                    )
            # Band range enable rates
            if codingParams._stat_band_1_5_count > 0:
                print(
                    f"  Enable rate bands 1-5  : {100 * codingParams._stat_band_1_5_sum / codingParams._stat_band_1_5_count:.1f}%"
                )
            if codingParams._stat_band_1_10_count > 0:
                print(
                    f"  Enable rate bands 1-10 : {100 * codingParams._stat_band_1_10_sum / codingParams._stat_band_1_10_count:.1f}%"
                )
            if codingParams._stat_band_1_15_count > 0:
                print(
                    f"  Enable rate bands 1-15 : {100 * codingParams._stat_band_1_15_sum / codingParams._stat_band_1_15_count:.1f}%"
                )
            if codingParams._stat_band_1_20_count > 0:
                print(
                    f"  Enable rate bands 1-20 : {100 * codingParams._stat_band_1_20_sum / codingParams._stat_band_1_20_count:.1f}%"
                )
            if codingParams._stat_pred_alloc_count > 0:
                print(
                    f"  Pred coverage of alloc : {avg_alloc_frac:.1f}% (pred-enabled / allocated bands)"
                )
            if codingParams._stat_alpha_count > 0:
                print(f"  Avg prediction gain    : {avg_alpha:.3f}")
        self.fp.close()

    def Encode(self, data, codingParams):
        """
        Encodes multichannel audio data and returns a tuple containing
        the scale factors, mantissa bit allocations, quantized mantissas,
        and the overall scale factor for each channel.
        """
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
        return codec.Decode(
            scaleFactor,
            bitAlloc,
            mantissa,
            overallScaleFactor,
            codingParams,
            mdct_pred,
        )


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    from prepare_materials import rmc

    elapsed = time.time()
    rmc("Van_124.wav", "VAN_96_onlyBS.wav", rate_kb=96)
    print(f"\nDone in {time.time() - elapsed:.1f}s")
