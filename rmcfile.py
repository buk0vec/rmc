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
from entropy import BlockEntropyCoder, RawMantissaCoder
from features import ENTROPY_CODING, VARIABLE_BIT_RATE, MID_SIDE_CODING, PREDICTION, SUBBASS_HYBRID, AC2A_BLOCK_SWITCHING, ADAPTIVE_CASCADE, K_ATTACK_MAX
import bisect
from blockswitching import LONG, START, SHORT, STOP, MEDIUM, N_SHORT_BLOCKS, ShortBlockSFBands, TransitionSFBands, WindowForBlockType, SelectBlockType, mask_to_group_lens, plan_cascade, DesignSFBands
from mdct import MDCT, IMDCT
from search import get_best_region, PRED_MAP, GAIN_TABLE, pred_type_to_samples, update_search_buffer
import subbass

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
        myParams.nMDCTLines_short = nMDCTLines // 16
        myParams.sfBands_short = ShortBlockSFBands(myParams.nMDCTLines_short, sampleRate)
        if AC2A_BLOCK_SWITCHING:
            _nMDCTLines_trans = (nMDCTLines + nMDCTLines // 16) // 2
            myParams.nMDCTLines_trans = _nMDCTLines_trans
            myParams.sfBands_trans = TransitionSFBands(_nMDCTLines_trans, sampleRate)
        myParams.prevBlockType = LONG
        myParams.blockType = LONG
        myParams.block_queue = []
        myParams.cascade_a = nMDCTLines
        myParams.cascade_b = nMDCTLines
        myParams.transientPositions = []  # populated by encoder; decoder doesn't need it
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
        myParams.k_attack_for_stop = 0  # k_attack from most recent START block; used for AC-2A SHORT pad and STOP window

        # entropy coders
        myParams.entropyCoder_long = BlockEntropyCoder(14) if ENTROPY_CODING else RawMantissaCoder()
        myParams.entropyCoder_short = BlockEntropyCoder(14) if ENTROPY_CODING else RawMantissaCoder()
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
        halfN_short = codingParams.nMDCTLines_short
        N_short = 2 * halfN_short
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

            codingParams.blockType = pb.ReadBits(3)
            if AC2A_BLOCK_SWITCHING and codingParams.blockType == START:
                k_attack_read = pb.ReadBits(4)
                codingParams.k_attack_for_stop = k_attack_read
                b_start = (15 + k_attack_read) * halfN_short  # recover exact b_start
                medium_lead = plan_cascade(k_attack_read, halfN_short)
                codingParams.cascade_a = halfN
                codingParams.cascade_b = b_start
                # Queue only lead MEDIUMs; STOP is always standard (cascade_a=128)
                codingParams.block_queue = [
                    {'type': MEDIUM, 'a': ma, 'b': mb} for (ma, mb) in medium_lead
                ]
            elif AC2A_BLOCK_SWITCHING and codingParams.blockType == MEDIUM:
                # Pop cascade dims from queue only on first channel (shared state)
                if iCh == 0 and getattr(codingParams, 'block_queue', []):
                    qitem = codingParams.block_queue.pop(0)
                    codingParams.cascade_a = qitem['a']
                    codingParams.cascade_b = qitem['b']
                elif iCh > 0:
                    pass  # cascade_a/b already set from iCh==0 pop
                else:
                    # Fallback: infer from OAA tail
                    codingParams.cascade_a = len(codingParams.overlapAndAdd[iCh])
                    codingParams.cascade_b = codingParams.nMDCTLines_short
            if AC2A_BLOCK_SWITCHING and codingParams.blockType == STOP:
                codingParams.cascade_a = halfN_short  # always standard STOP
                codingParams.cascade_b = halfN
            if SUBBASS_HYBRID and codingParams.blockType == LONG:
                has_subbass = bool(pb.ReadBits(1))
            elif SUBBASS_HYBRID:
                has_subbass = True  # non-LONG blocks always carry subbass side channel
            else:
                has_subbass = False
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

            if AC2A_BLOCK_SWITCHING and codingParams.blockType == SHORT:
                # AC-2A single SHORT: no grouping mask
                overallScaleFactor = pb.ReadBits(codingParams.nScaleBits)
                scaleFactor = []
                bitAlloc = []
                for _ in range(sfBands_short.nBands):
                    ba = pb.ReadBits(codingParams.nMantSizeBits)
                    if ba: ba += 1
                    bitAlloc.append(ba)
                    scaleFactor.append(pb.ReadBits(codingParams.nScaleBits))
                mantissa = codingParams.entropyCoder_short.decode_block(
                    pb, bitAlloc, sfBands_short, codingParams.nMDCTLines_short
                )
            elif AC2A_BLOCK_SWITCHING and codingParams.blockType == MEDIUM:
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
                    if ba: ba += 1
                    bitAlloc.append(ba)
                    scaleFactor.append(pb.ReadBits(codingParams.nScaleBits))
                mantissa = codingParams.entropyCoder_long.decode_block(
                    pb, bitAlloc, sfBands_med, halfN_med
                )
            elif codingParams.blockType == SHORT:
                # Edler SHORT: grouping mask + groups
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
                # LONG and START/STOP (AC-2A or Edler)
                overallScaleFactor = pb.ReadBits(codingParams.nScaleBits)
                scaleFactor = []
                bitAlloc = []
                if AC2A_BLOCK_SWITCHING and codingParams.blockType in (START, STOP):
                    ca = getattr(codingParams, 'cascade_a', halfN)
                    cb = getattr(codingParams, 'cascade_b', codingParams.nMDCTLines_short)
                    _halfN_dec = (ca + cb) // 2
                    if _halfN_dec != codingParams.nMDCTLines_trans:
                        _sfb_dec = DesignSFBands(_halfN_dec, codingParams.sampleRate)
                    else:
                        _sfb_dec = codingParams.sfBands_trans
                else:
                    _sfb_dec = codingParams.sfBands
                    _halfN_dec = codingParams.nMDCTLines
                for _ in range(_sfb_dec.nBands):
                    ba = pb.ReadBits(codingParams.nMantSizeBits)
                    if ba: ba += 1
                    bitAlloc.append(ba)
                    scaleFactor.append(pb.ReadBits(codingParams.nScaleBits))
                mantissa = codingParams.entropyCoder_long.decode_block(
                    pb, bitAlloc, _sfb_dec, _halfN_dec
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
                    _k = (codingParams.k_attack_for_stop if AC2A_BLOCK_SWITCHING else None)
                    window = WindowForBlockType(codingParams.blockType, N, N_short, k_attack=_k)
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

            # Sub-bass hybrid: read K low-band bins and add LONG IMDCT reconstruction
            if has_subbass:
                from quantize import vDequantize as _vDQ2
                ovs  = pb.ReadBits(codingParams.nScaleBits)
                sf   = pb.ReadBits(codingParams.nScaleBits)
                mant_low = np.array([pb.ReadBits(subbass.LOW_MANT_BITS)
                                     for _ in range(subbass.K_BINS)], dtype=np.int32)
                low_bins = _vDQ2(sf, mant_low, codingParams.nScaleBits, subbass.LOW_MANT_BITS)
                low_bins /= (1 << ovs)
                low_padded = np.zeros(halfN)
                low_padded[:subbass.K_BINS] = low_bins
                data_low = WindowForBlockType(LONG, N, N_short) * IMDCT(low_padded, halfN, halfN)
                decodedData = decodedData + data_low

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
        # ola_size is self-sizing: LONG/STOP leave 1024-sample tail; START/SHORT leave 128-sample tail
        data_current_domain = []
        for iCh in range(codingParams.nChannels):
            decodedData = raw_decoded[iCh]
            ola_size = len(codingParams.overlapAndAdd[iCh])
            out = np.maximum(-1., np.minimum(1.,
                np.add(codingParams.overlapAndAdd[iCh], decodedData[:ola_size])
            ))
            data_current_domain.append(out)
            codingParams.overlapAndAdd[iCh] = decodedData[ola_size:]

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
        if AC2A_BLOCK_SWITCHING and codingParams.blockType == SHORT:
            _halfN_buf = codingParams.nMDCTLines_short
        elif AC2A_BLOCK_SWITCHING and codingParams.blockType in (START, STOP, MEDIUM):
            ca = getattr(codingParams, 'cascade_a', halfN)
            cb = getattr(codingParams, 'cascade_b', codingParams.nMDCTLines_short)
            _halfN_buf = (ca + cb) // 2
        else:
            _halfN_buf = halfN
        _N_buf = 2 * _halfN_buf
        for iCh in range(codingParams.nChannels):
            update_search_buffer(codingParams.search_buffer[iCh], lr_for_buffer[iCh], _halfN_buf, _N_buf)

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
        codingParams.nMDCTLines_short = codingParams.nMDCTLines // 16
        codingParams.sfBands_short = ShortBlockSFBands(codingParams.nMDCTLines_short, codingParams.sampleRate)
        if AC2A_BLOCK_SWITCHING:
            _nMDCTLines_trans = (codingParams.nMDCTLines + codingParams.nMDCTLines_short) // 2
            codingParams.nMDCTLines_trans = _nMDCTLines_trans
            codingParams.sfBands_trans = TransitionSFBands(_nMDCTLines_trans, codingParams.sampleRate)
            codingParams.currentSamplePos = 0
            codingParams.short_blocks_remaining = 0
            # Pre-set nSamplesPerBlock for the first PCM read
            _positions0 = getattr(codingParams, 'transientPositions', [])
            if _positions0:
                _raw0 = _positions0[0]
                if _raw0 < 2 * codingParams.nMDCTLines:
                    _k0 = max(0, min(K_ATTACK_MAX, (_raw0 - codingParams.nMDCTLines) // codingParams.nMDCTLines_short))
                    _bs0 = (15 + _k0) * codingParams.nMDCTLines_short
                    codingParams.nSamplesPerBlock = _bs0
        codingParams.blockType = LONG
        codingParams.block_queue = []
        codingParams.cascade_a = codingParams.nMDCTLines
        codingParams.cascade_b = codingParams.nMDCTLines
        codingParams.entropyCoder_long = BlockEntropyCoder(14) if ENTROPY_CODING else RawMantissaCoder()
        codingParams.entropyCoder_short = BlockEntropyCoder(14) if ENTROPY_CODING else RawMantissaCoder()

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
        codingParams.k_attack_for_stop = 0
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
        # Sub-bass hybrid: per-channel LPF state and delay buffer (kept warm across all blocks)
        if SUBBASS_HYBRID:
            codingParams.lpf_state     = [subbass.make_state()     for _ in range(codingParams.nChannels)]
            codingParams.lpf_delay_buf = [subbass.make_delay_buf() for _ in range(codingParams.nChannels)]
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
        if AC2A_BLOCK_SWITCHING:
            # Find the next transient within 2 * halfN samples using exact positions
            _positions = getattr(codingParams, 'transientPositions', [])
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
            _rem  = codingParams.short_blocks_remaining

            if codingParams.block_queue:
                # Pop next planned MEDIUM or STOP block from cascade queue
                qitem = codingParams.block_queue.pop(0)
                _bt = qitem['type']
                codingParams.cascade_a = qitem['a']
                codingParams.cascade_b = qitem['b']
            elif _prev in (LONG, STOP):
                if k_attack >= 0:
                    _bt = START
                    medium_lead = plan_cascade(_k_encoded, halfN_short)
                    codingParams.cascade_a = halfN
                    codingParams.cascade_b = _b_start
                    for (ma, mb) in medium_lead:
                        codingParams.block_queue.append({'type': MEDIUM, 'a': ma, 'b': mb})
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
                    codingParams.block_queue.append({'type': STOP, 'a': halfN_short, 'b': halfN})
            codingParams.group_lens = None
            codingParams.grouping_mask = 0

        # Build full analysis windows per channel (rolling 1024-sample priorBlock for AC-2A)
        fullBlockData_ = []
        maskingData_ = []
        new_prior = []
        for iCh in range(codingParams.nChannels):
            all_samples = np.concatenate((codingParams.priorBlock[iCh], data[iCh]))
            # After START, keep b_start samples so MEDIUM's left overlap is fully covered
            _keep = (codingParams.cascade_b
                     if AC2A_BLOCK_SWITCHING and codingParams.blockType == START
                        and codingParams.cascade_b > halfN
                     else halfN)
            new_prior.append(all_samples[-_keep:])
            if AC2A_BLOCK_SWITCHING:
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
            else:
                fullBlockData_.append(all_samples)                # always 2048 for Edler
                maskingData_.append(all_samples)
        codingParams.priorBlock = new_prior

        # Block-level M/S stereo decision: use M/S if side energy < min(L energy, R energy)
        use_ms = False
        if MID_SIDE_CODING and codingParams.nChannels == 2:
            L, R = fullBlockData_[0], fullBlockData_[1]
            M = (L + R) * 0.5
            S = (L - R) * 0.5
            if np.var(S) < min(np.var(L), np.var(R)):
                use_ms = True
                fullBlockData_ = [M, S]

        # Block type state machine (Edler / non-AC-2A path only)
        prev_block_type_for_adj = codingParams.prevBlockType
        if not AC2A_BLOCK_SWITCHING:
            k_attack = codingParams.transientBlocks.get(codingParams.blockIndex, -1)
            codingParams.blockType = SelectBlockType(k_attack, codingParams.prevBlockType)
            codingParams.prevBlockType = codingParams.blockType
            codingParams.blockIndex += 1
            if codingParams.blockType == SHORT:
                grouping_mask = (1 << (N_SHORT_BLOCKS - 1)) - 1  # 0b1111111 = all singletons
                codingParams.group_lens = mask_to_group_lens(grouping_mask)
                codingParams.grouping_mask = grouping_mask
            else:
                codingParams.group_lens = None
                codingParams.grouping_mask = 0
        current_block_type = codingParams.blockType

        # Adjacent LONG blocks (immediately before START or immediately after STOP) must encode
        # y_high + K_BINS side channel, same as transient blocks, so the MDCT overlap region
        # sees the same signal on both sides of the LONG↔START and STOP↔LONG boundaries (TDAC fix).
        is_adjacent_long = False
        if SUBBASS_HYBRID and current_block_type == LONG:
            is_adjacent_long = (
                prev_block_type_for_adj == STOP or            # post-STOP LONG
                codingParams.blockIndex in codingParams.transientBlocks  # pre-START LONG (blockIndex already incremented)
            )

        # Sub-bass hybrid split: run LPF every block to keep state warm.
        # Transient blocks + adjacent LONG: split into y_low (K bins) and y_high (encoding path).
        # Non-adjacent LONG blocks: encode x_delayed.
        lpf_low_k_bins = {}  # iCh -> K low MDCT bins (transient + adjacent LONG blocks)
        lpf_y_high     = {}  # iCh -> y_high 2048-sample residual (transient + adjacent LONG blocks)
        if SUBBASS_HYBRID:
            for iCh in range(codingParams.nChannels):
                x = fullBlockData_[iCh]
                if current_block_type != LONG or is_adjacent_long:
                    y_low, y_high, new_state, new_delay = subbass.compute_split(
                        x, codingParams.lpf_state[iCh], codingParams.lpf_delay_buf[iCh],
                        codingParams.sampleRate
                    )
                    w_long = WindowForBlockType(LONG, N, N_short)
                    low_mdct = MDCT(w_long * y_low, halfN, halfN)[:halfN]
                    lpf_low_k_bins[iCh] = low_mdct[:subbass.K_BINS]
                    lpf_y_high[iCh]     = y_high
                    # fullBlockData_ stays as x (used for masking); y_high used via lpf_y_high
                else:
                    new_state, new_delay, x_delayed = subbass.update_state_only(
                        x, codingParams.lpf_state[iCh], codingParams.lpf_delay_buf[iCh],
                        codingParams.sampleRate
                    )
                    fullBlockData_[iCh] = x_delayed
                codingParams.lpf_state[iCh]     = new_state
                codingParams.lpf_delay_buf[iCh] = new_delay

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
            # For SUBBASS_HYBRID transient blocks, prediction runs on y_high (the Edler signal).
            # Per-band enable naturally skips low bands where y_high ≈ 0, so the template
            # subtraction only applies where it helps (mid/high frequencies).
            pred_signal = (lpf_y_high[iCh]
                           if SUBBASS_HYBRID and (current_block_type != LONG or is_adjacent_long) and iCh in lpf_y_high
                           else fullBlockData_[iCh])
            run_prediction = PREDICTION and current_block_type != SHORT
            if run_prediction:
                _k = (codingParams.k_attack_for_stop if AC2A_BLOCK_SWITCHING else None)
                window = WindowForBlockType(current_block_type, N, N_short, k_attack=_k)
                mdct_X = MDCT(window * pred_signal, halfN, halfN)[:halfN]
                range_type, pcm_residual, rel_offset, mdct_P, alpha_idx, alpha_q = get_best_region(
                    mdct_X, pred_signal, codingParams,
                    search_bufs[iCh], block_type=current_block_type
                )
                if range_type is not None:
                    # Per-band enable: apply prediction only where it reduces peak magnitude
                    residual_full = mdct_X - alpha_q * mdct_P
                    enable_f = np.zeros(codingParams.sfBands.nBands, dtype=bool)
                    enable_m = np.zeros(halfN)
                    for iBand in range(codingParams.sfBands.nBands):
                        lo = codingParams.sfBands.lowerLine[iBand]
                        hi = codingParams.sfBands.upperLine[iBand] + 1
                        if np.amax(np.abs(residual_full[lo:hi])) < np.amax(np.abs(mdct_X[lo:hi])):
                            enable_f[iBand] = True
                            enable_m[lo:hi] = 1.0
                    if not np.any(enable_f):
                        range_type, pcm_residual, rel_offset, mdct_P, alpha_idx, alpha_q = \
                            None, pred_signal, 0, None, 0, 1.0
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
            else:  # SHORT or PREDICTION disabled
                range_type, pcm_residual, rel_offset, mdct_P, alpha_idx, alpha_q = \
                    None, pred_signal, 0, None, 0, 1.0
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
            oh = 32 + 3 + 2  # nBytes field (4 bytes) + 3-bit block type + pred type
            if AC2A_BLOCK_SWITCHING and current_block_type == START:
                oh += 4  # k_attack field (4 bits: k range 0-15)
            if iCh == 0 and codingParams.nChannels == 2:
                oh += 1  # M/S flag
            if ranges[iCh] is not None:
                oh += 1 + 8 + 3  # sign + offset + gain
                if current_block_type != SHORT:
                    oh += codingParams.sfBands.nBands  # per-band enable flags
            if current_block_type == SHORT:
                if not AC2A_BLOCK_SWITCHING:
                    oh += 7  # grouping mask (Edler only)
            if SUBBASS_HYBRID:
                if current_block_type == LONG:
                    oh += 1  # has_subbass flag bit
                    if is_adjacent_long:
                        oh += codingParams.nScaleBits * 2 + subbass.K_BINS * subbass.LOW_MANT_BITS
                else:
                    oh += codingParams.nScaleBits * 2 + subbass.K_BINS * subbass.LOW_MANT_BITS
            block_overhead.append(oh)

        # Encode the residual signals with original signal for psychoacoustic masking
        codingParams.masking_signals = maskingData_
        codingParams.mdct_pred_corrections = corrections
        codingParams.block_overhead = block_overhead
        (scaleFactor, bitAlloc, mantissa, overallScaleFactor) = self.Encode(fullBlockData, codingParams)
        codingParams.masking_signals = None
        codingParams.mdct_pred_corrections = None

        # Encoder-side decode: reconstruct lossy output and store in search buffer so that
        # future prediction searches see the same signal the decoder will have.
        halfN_trans = getattr(codingParams, 'nMDCTLines_trans', (halfN + halfN_short) // 2)
        decoded_channels = []
        for iCh in range(codingParams.nChannels):
            if AC2A_BLOCK_SWITCHING and current_block_type == SHORT:
                mantissa_full = codec.ExpandMantissa(
                    mantissa[iCh], bitAlloc[iCh], sfBands_short, halfN_short
                )
                decodedData = codec.Decode(
                    scaleFactor[iCh], bitAlloc[iCh], mantissa_full,
                    overallScaleFactor[iCh], codingParams
                )
            elif AC2A_BLOCK_SWITCHING and current_block_type == MEDIUM:
                ca = codingParams.cascade_a
                cb = codingParams.cascade_b
                halfN_med = (ca + cb) // 2
                sfBands_med = DesignSFBands(halfN_med, codingParams.sampleRate)
                mantissa_full = codec.ExpandMantissa(
                    mantissa[iCh], bitAlloc[iCh], sfBands_med, halfN_med
                )
                decodedData = codec.Decode(
                    scaleFactor[iCh], bitAlloc[iCh], mantissa_full,
                    overallScaleFactor[iCh], codingParams
                )
            elif AC2A_BLOCK_SWITCHING and current_block_type in (START, STOP):
                ca = codingParams.cascade_a
                cb = codingParams.cascade_b
                halfN_used = (ca + cb) // 2
                sfBands_used = DesignSFBands(halfN_used, codingParams.sampleRate) \
                    if halfN_used != halfN_trans else codingParams.sfBands_trans
                mantissa_full = codec.ExpandMantissa(
                    mantissa[iCh], bitAlloc[iCh], sfBands_used, halfN_used
                )
                scaled_pred = (alpha_qs[iCh] * pred_mdcts[iCh] * enable_masks[iCh]
                               if pred_mdcts[iCh] is not None else None)
                decodedData = codec.Decode(
                    scaleFactor[iCh], bitAlloc[iCh], mantissa_full,
                    overallScaleFactor[iCh], codingParams, mdct_pred=scaled_pred
                )
            elif current_block_type != SHORT:
                mantissa_full = codec.ExpandMantissa(
                    mantissa[iCh], bitAlloc[iCh], codingParams.sfBands, halfN
                )
                scaled_pred = (alpha_qs[iCh] * pred_mdcts[iCh] * enable_masks[iCh]
                               if pred_mdcts[iCh] is not None else None)
                decodedData = codec.Decode(
                    scaleFactor[iCh], bitAlloc[iCh], mantissa_full,
                    overallScaleFactor[iCh], codingParams, mdct_pred=scaled_pred
                )
            else:  # Edler SHORT
                full_mant = [codec.ExpandMantissa(mantissa[iCh][i], bitAlloc[iCh][i], sfBands_short, halfN_short)
                             for i in range(N_SHORT_BLOCKS)]
                decodedData = codec.Decode(
                    scaleFactor[iCh], bitAlloc[iCh], full_mant,
                    overallScaleFactor[iCh], codingParams
                )
            # Sub-bass hybrid: add quantized y_low LONG IMDCT so search buffer matches decoder.
            # Every non-LONG block + adjacent LONG: decoder outputs y_high + y_low; buffer must match.
            if SUBBASS_HYBRID and (current_block_type != LONG or is_adjacent_long) and iCh in lpf_low_k_bins:
                from quantize import ScaleFactor as _SF, vMantissa as _vM, vDequantize as _vDQ
                low_bins = lpf_low_k_bins[iCh]
                max_abs  = np.max(np.abs(low_bins)) if np.any(low_bins != 0) else 1e-10
                ovs = _SF(max_abs, codingParams.nScaleBits)
                scaled = low_bins * (1 << ovs)
                max_sc = np.max(np.abs(scaled)) if np.any(scaled != 0) else 1e-10
                sf  = _SF(max_sc, codingParams.nScaleBits, subbass.LOW_MANT_BITS)
                mant_q = _vM(scaled, sf, codingParams.nScaleBits, subbass.LOW_MANT_BITS)
                low_bins_q = _vDQ(sf, mant_q, codingParams.nScaleBits, subbass.LOW_MANT_BITS)
                low_bins_q /= (1 << ovs)
                low_padded = np.zeros(halfN)
                low_padded[:subbass.K_BINS] = low_bins_q
                data_low = WindowForBlockType(LONG, N, N_short) * IMDCT(low_padded, halfN, halfN)
                decodedData = decodedData + data_low
            decoded_channels.append(decodedData)

        # Convert M/S decoded data back to L/R before storing in search buffer
        if use_ms and codingParams.nChannels == 2:
            M_hat, S_hat = decoded_channels[0], decoded_channels[1]
            lr_for_buffer = [M_hat + S_hat, M_hat - S_hat]
        else:
            lr_for_buffer = decoded_channels
        if AC2A_BLOCK_SWITCHING and current_block_type == SHORT:
            _halfN_buf = halfN_short
        elif AC2A_BLOCK_SWITCHING and current_block_type in (START, STOP, MEDIUM):
            _halfN_buf = (codingParams.cascade_a + codingParams.cascade_b) // 2
        else:
            _halfN_buf = halfN
        _N_buf = 2 * _halfN_buf
        for iCh in range(codingParams.nChannels):
            update_search_buffer(codingParams.search_buffer[iCh], lr_for_buffer[iCh], _halfN_buf, _N_buf)

        # Write bitstream per channel:
        #   [2b block_type | 2b pred_type | 1b ms_flag (ch0 only)]
        #   [1b sign + 8b offset + 3b gain + nBands enable flags (if pred active)]
        #   [7b grouping_mask (if SHORT)]
        #   [nScaleBits ovs | nMantSizeBits+nScaleBits per band | entropy-coded mantissas]
        for iCh in range(codingParams.nChannels):
            entropy_pbs = []
            nBits = 5  # 3 bits block type + 2 bits prediction type
            if AC2A_BLOCK_SWITCHING and codingParams.blockType == START:
                nBits += 4  # k_attack field
            if iCh == 0 and codingParams.nChannels == 2:
                nBits += 1  # M/S flag
            if ranges[iCh] is not None:
                nBits += 9 + 3  # sign+offset + gain
                if codingParams.blockType != SHORT:
                    nBits += codingParams.sfBands.nBands  # per-band enables for LONG/START/STOP
            if AC2A_BLOCK_SWITCHING and codingParams.blockType == SHORT:
                # AC-2A single SHORT: overallScaleFactor + per-band ba/sf + entropy mantissa
                entropy_pb_short = codingParams.entropyCoder_short.encode_block(
                    mantissa[iCh], bitAlloc[iCh], sfBands_short
                )
                nBits += codingParams.nScaleBits  # overallScaleFactor
                for iBand in range(sfBands_short.nBands):
                    nBits += codingParams.nMantSizeBits + codingParams.nScaleBits
                nBits += entropy_pb_short.nBits
                if VARIABLE_BIT_RATE:
                    _raw_short = sum(
                        bitAlloc[iCh][iBand] * sfBands_short.nLines[iBand]
                        for iBand in range(sfBands_short.nBands) if bitAlloc[iCh][iBand] > 0
                    )
                    if _raw_short > 0:
                        _r = entropy_pb_short.nBits / _raw_short
                        codingParams._entropy_ratio_short = 0.9 * codingParams._entropy_ratio_short + 0.1 * _r
                        codingParams._entropy_inflation_short = min(2.5, max(0.5, 1.0 / codingParams._entropy_ratio_short))
            elif codingParams.blockType == SHORT:
                # Edler SHORT: grouping mask + groups
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
                        codingParams._entropy_ratio_short = 0.9 * codingParams._entropy_ratio_short + 0.1 * _r
                        codingParams._entropy_inflation_short = min(2.5, max(0.5, 1.0 / codingParams._entropy_ratio_short))
            elif AC2A_BLOCK_SWITCHING and codingParams.blockType == MEDIUM:
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
                # LONG and START/STOP (AC-2A or Edler)
                ca = getattr(codingParams, 'cascade_a', halfN)
                cb = getattr(codingParams, 'cascade_b', halfN)
                halfN_enc = (ca + cb) // 2
                _sfb_enc = (
                    DesignSFBands(halfN_enc, codingParams.sampleRate)
                    if AC2A_BLOCK_SWITCHING and codingParams.blockType in (START, STOP) and halfN_enc != halfN_trans
                    else (codingParams.sfBands_trans
                          if AC2A_BLOCK_SWITCHING and codingParams.blockType in (START, STOP)
                          else codingParams.sfBands)
                )
                entropy_pb = codingParams.entropyCoder_long.encode_block(
                    mantissa[iCh], bitAlloc[iCh], _sfb_enc
                )
                nBits += codingParams.nScaleBits
                for iBand in range(_sfb_enc.nBands):
                    nBits += codingParams.nMantSizeBits + codingParams.nScaleBits
                nBits += entropy_pb.nBits
                # Update LONG entropy compression ratio (EMA)
                if VARIABLE_BIT_RATE:
                    _raw_long = sum(
                        bitAlloc[iCh][iBand] * _sfb_enc.nLines[iBand]
                        for iBand in range(_sfb_enc.nBands)
                        if bitAlloc[iCh][iBand] > 0
                    )
                    if _raw_long > 0:
                        _r = entropy_pb.nBits / _raw_long
                        codingParams._entropy_ratio = 0.9 * codingParams._entropy_ratio + 0.1 * _r
                        codingParams._entropy_inflation = min(2.5, max(0.5, 1.0 / codingParams._entropy_ratio))

            if SUBBASS_HYBRID:
                if codingParams.blockType == LONG:
                    nBits += 1  # has_subbass flag
                    if is_adjacent_long:
                        nBits += codingParams.nScaleBits * 2 + subbass.K_BINS * subbass.LOW_MANT_BITS
                else:
                    nBits += codingParams.nScaleBits * 2 + subbass.K_BINS * subbass.LOW_MANT_BITS
            nBytes = (nBits + BYTESIZE - 1) // BYTESIZE
            self.fp.write(pack("<L", int(nBytes)))

            pb = PackedBits()
            pb.Size(nBytes)
            pb.WriteBits(codingParams.blockType, 3)
            if AC2A_BLOCK_SWITCHING and codingParams.blockType == START:
                pb.WriteBits(codingParams.k_attack_for_stop, 4)
            if SUBBASS_HYBRID and codingParams.blockType == LONG:
                pb.WriteBits(1 if is_adjacent_long else 0, 1)
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

            if AC2A_BLOCK_SWITCHING and codingParams.blockType == SHORT:
                # AC-2A single SHORT: overallScaleFactor + per-band ba/sf + entropy mantissa
                pb.WriteBits(overallScaleFactor[iCh], codingParams.nScaleBits)
                for iBand in range(sfBands_short.nBands):
                    ba = bitAlloc[iCh][iBand]
                    if ba: ba -= 1
                    pb.WriteBits(ba, codingParams.nMantSizeBits)
                    pb.WriteBits(scaleFactor[iCh][iBand], codingParams.nScaleBits)
                pb.WriteBits(entropy_pb_short.buffer, entropy_pb_short.nBits)
            elif AC2A_BLOCK_SWITCHING and codingParams.blockType == MEDIUM:
                pb.WriteBits(overallScaleFactor[iCh], codingParams.nScaleBits)
                for iBand in range(_sfb_med.nBands):
                    ba = bitAlloc[iCh][iBand]
                    if ba: ba -= 1
                    pb.WriteBits(ba, codingParams.nMantSizeBits)
                    pb.WriteBits(scaleFactor[iCh][iBand], codingParams.nScaleBits)
                pb.WriteBits(entropy_pb.buffer, entropy_pb.nBits)
            elif codingParams.blockType == SHORT:
                # Edler SHORT: grouping mask + groups
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
                # LONG and START/STOP (AC-2A or Edler) — _sfb_enc already computed above
                pb.WriteBits(overallScaleFactor[iCh], codingParams.nScaleBits)
                for iBand in range(_sfb_enc.nBands):
                    ba = bitAlloc[iCh][iBand]
                    if ba: ba -= 1
                    pb.WriteBits(ba, codingParams.nMantSizeBits)
                    pb.WriteBits(scaleFactor[iCh][iBand], codingParams.nScaleBits)
                pb.WriteBits(entropy_pb.buffer, entropy_pb.nBits)
            # Sub-bass hybrid: append K low-band bins after mantissa data
            if SUBBASS_HYBRID and (codingParams.blockType != LONG or is_adjacent_long) and iCh in lpf_low_k_bins:
                from quantize import ScaleFactor as _SF2, vMantissa as _vM2
                low_bins = lpf_low_k_bins[iCh]
                max_abs  = np.max(np.abs(low_bins)) if np.any(low_bins != 0) else 1e-10
                ovs = _SF2(max_abs, codingParams.nScaleBits)
                scaled = low_bins * (1 << ovs)
                max_sc = np.max(np.abs(scaled)) if np.any(scaled != 0) else 1e-10
                sf  = _SF2(max_sc, codingParams.nScaleBits, subbass.LOW_MANT_BITS)
                mant = _vM2(scaled, sf, codingParams.nScaleBits, subbass.LOW_MANT_BITS)
                pb.WriteBits(int(ovs), codingParams.nScaleBits)
                pb.WriteBits(int(sf),  codingParams.nScaleBits)
                mask = (1 << subbass.LOW_MANT_BITS) - 1
                for m in mant:
                    pb.WriteBits(int(m) & mask, subbass.LOW_MANT_BITS)
            self.fp.write(pb.GetPackedData())

        # AC-2A: advance sample counter and pre-set nSamplesPerBlock for the next PCM read
        if AC2A_BLOCK_SWITCHING:
            codingParams.currentSamplePos += len(data[0])
            _prev_now = codingParams.prevBlockType

            # Look ahead from new position for the next transient
            _pos_next = codingParams.currentSamplePos
            _positions = getattr(codingParams, 'transientPositions', [])
            _tidx_next = bisect.bisect_left(_positions, _pos_next)
            if _tidx_next < len(_positions) and _positions[_tidx_next] < _pos_next + 2 * halfN:
                _raw_next = _positions[_tidx_next] - _pos_next
                _k_next = max(0, min(K_ATTACK_MAX, (_raw_next - halfN) // halfN_short))
                _b_next = (15 + _k_next) * halfN_short
                _k_next_valid = True
            else:
                _b_next = halfN
                _k_next_valid = False

            if codingParams.block_queue:
                # Next block is queued (MEDIUM or STOP) — its nSPB = b of that block
                codingParams.nSamplesPerBlock = codingParams.block_queue[0]['b']
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
    rmc("Van_124.wav", "VAN_96_onlyBS.wav", rate_kb=96)
    print(f"\nDone in {time.time() - elapsed:.1f}s")
