"""
pacfile.py -- Defines a PACFile class to handle reading and writing audio
data to an audio file holding data compressed using an MDCT-based perceptual audio
coding algorithm.  The MDCT lines of each audio channel are grouped into bands,
each sharing a single scaleFactor and bit allocation that are used to block-
floating point quantize those lines.  This class is a subclass of AudioFile.

-----------------------------------------------------------------------
© 2009-2026 Marina Bosi & Richard E. Goldberg -- All rights reserved
-----------------------------------------------------------------------

See the documentation of the AudioFile class for general use of the AudioFile
class.

Notes on reading and decoding PAC files:

    The OpenFileForReading() function returns a CodedParams object containing:

        nChannels = the number of audio channels
        sampleRate = the sample rate of the audio samples
        numSamples = the total number of samples in the file for each channel
        nMDCTLines = half the MDCT block size (block switching not supported)
        nSamplesPerBlock = MDCTLines (but a name that PCM files look for)
        nScaleBits = the number of bits storing scale factors
        nMantSizeBits = the number of bits storing mantissa bit allocations
        sfBands = a ScaleFactorBands object
        overlapAndAdd = decoded data from the prior block (initially all zeros)

    The returned ScaleFactorBands object, sfBands, contains an allocation of
    the MDCT lines into groups that share a single scale factor and mantissa bit
    allocation.  sfBands has the following attributes available:

        nBands = the total number of scale factor bands
        nLines[iBand] = the number of MDCT lines in scale factor band iBand
        lowerLine[iBand] = the first MDCT line in scale factor band iBand
        upperLine[iBand] = the last MDCT line in scale factor band iBand


Notes on encoding and writing PAC files:

    When writing to a PACFile the CodingParams object passed to OpenForWriting()
    should have the following attributes set:

        nChannels = the number of audio channels
        sampleRate = the sample rate of the audio samples
        numSamples = the total number of samples in the file for each channel
        nMDCTLines = half the MDCT block size (format does not support block switching)
        nSamplesPerBlock = MDCTLines (but a name that PCM files look for)
        nScaleBits = the number of bits storing scale factors
        nMantSizeBits = the number of bits storing mantissa bit allocations
        targetBitsPerSample = the target encoding bit rate in units of bits per sample

    The first three attributes (nChannels, sampleRate, and numSamples) are
    typically added by the original data source (e.g. a PCMFile object) but
    numSamples may need to be extended to account for the MDCT coding delay of
    nMDCTLines and any zero-padding done in the final data block

    OpenForWriting() will add the following attributes to be used during the encoding
    process carried out in WriteDataBlock():

        sfBands = a ScaleFactorBands object
        priorBlock = the prior block of audio data (initially all zeros)

    The passed ScaleFactorBands object, sfBands, contains an allocation of
    the MDCT lines into groups that share a single scale factor and mantissa bit
    allocation.  sfBands has the following attributes available:

        nBands = the total number of scale factor bands
        nLines[iBand] = the number of MDCT lines in scale factor band iBand
        lowerLine[iBand] = the first MDCT line in scale factor band iBand
        upperLine[iBand] = the last MDCT line in scale factor band iBand

Description of the PAC File Format:

    Header:

        tag                 4 byte file tag equal to "PAC "
        sampleRate          little-endian unsigned long ("<L" format in struct)
        nChannels           little-endian unsigned short("<H" format in struct)
        numSamples          little-endian unsigned long ("<L" format in struct)
        nMDCTLines          little-endian unsigned long ("<L" format in struct)
        nScaleBits          little-endian unsigned short("<H" format in struct)
        nMantSizeBits       little-endian unsigned short("<H" format in struct)
        nSFBands            little-endian unsigned long ("<L" format in struct)
        for iBand in range(nSFBands):
            nLines[iBand]   little-endian unsigned short("<H" format in struct)

    Each Data Block:  (reads data blocks until end of file hit)

        for iCh in range(nChannels):
            nBytes          little-endian unsigned long ("<L" format in struct)
            as bits packed into an array of nBytes bytes:
                overallScale[iCh]                       nScaleBits bits
                for iBand in range(nSFBands):
                    scaleFactor[iCh][iBand]             nScaleBits bits
                    bitAlloc[iCh][iBand]                nMantSizeBits bits
                    if bitAlloc[iCh][iBand]:
                        for m in nLines[iBand]:
                            mantissa[iCh][iBand][m]     bitAlloc[iCh][iBand]+1 bits
                <extra custom data bits as long as space is included in nBytes>

"""

from audiofile import * # base class
from bitpack import *  # class for packing data into an array of bytes where each item's number of bits is specified
import codec    # module where the actual PAC coding functions reside(this module only specifies the PAC file format)
from psychoac import ScaleFactorBands, AssignMDCTLinesFromFreqLimits  # defines the grouping of MDCT lines into scale factor bands
from entropy import BlockEntropyCoder
from blockswitching import LONG, SHORT, N_SHORT_BLOCKS, ShortBlockSFBands, WindowForBlockType
from mdct import MDCT, IMDCT
from search import get_best_region, PRED_MAP, GAIN_TABLE

import numpy as np  # to allow conversion of data blocks to numpy's array object
MAX16BITS = 32767


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
        sfBands_short = ShortBlockSFBands(codingParams.nMDCTLines_short, codingParams.sampleRate)

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
                    prev_use_ms = getattr(codingParams, 'prev_use_ms', False)
                    if prev_use_ms and codingParams.nChannels == 2:
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
                if codingParams.blockType == LONG:  # enable flags only for LONG blocks
                    pred_enable_flags = [bool(pb.ReadBits(1))
                                         for _ in range(codingParams.sfBands.nBands)]

            if codingParams.blockType == SHORT:
                overallScaleFactor = []
                scaleFactor = []
                bitAlloc = []
                mantissa = []
                for i in range(N_SHORT_BLOCKS):
                    overallScaleFactor.append(pb.ReadBits(codingParams.nScaleBits))
                    sf_i = []
                    ba_i = []
                    for iBand in range(sfBands_short.nBands):
                        ba = pb.ReadBits(codingParams.nMantSizeBits)
                        if ba: ba += 1
                        ba_i.append(ba)
                        sf_i.append(pb.ReadBits(codingParams.nScaleBits))
                    mant_i = codingParams.entropyCoder_short.decode_block(
                        pb, ba_i, sfBands_short, codingParams.nMDCTLines_short
                    )
                    scaleFactor.append(sf_i)
                    bitAlloc.append(ba_i)
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
            mdct_P = None  # for LONG blocks: single array; for SHORT: list of 8 arrays
            if pred_type != PRED_MAP[None]:
                if pred_type == PRED_MAP['quarter']:
                    start_offset = codingParams.numSamplesQuarterNote
                elif pred_type == PRED_MAP['half']:
                    start_offset = codingParams.numSamplesHalfBar
                else:
                    start_offset = codingParams.numSamplesBar
                if use_ms and codingParams.nChannels == 2:
                    buf_L = codingParams.search_buffer[0]
                    buf_R = codingParams.search_buffer[1]
                    buf = (buf_L + buf_R) * 0.5 if iCh == 0 else (buf_L - buf_R) * 0.5
                else:
                    buf = codingParams.search_buffer[iCh]
                seg_start = len(buf) - start_offset + pred_offset
                if 0 <= seg_start and seg_start + N <= len(buf):
                    candidate = buf[seg_start : seg_start + N]
                    if codingParams.blockType != SHORT:
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
                    else:
                        # SHORT block: per-sub-block prediction using same candidate
                        halfN_short = codingParams.nMDCTLines_short
                        pad = N // 4 - N_short // 4
                        sub_preds = []
                        for i in range(N_SHORT_BLOCKS):
                            pcm_sub = candidate[pad + i*halfN_short : pad + i*halfN_short + N_short]
                            w_s = WindowForBlockType(SHORT, N, N_short)
                            mdct_P_i = MDCT(w_s * pcm_sub, halfN_short, halfN_short)[:halfN_short]
                            sub_preds.append(pred_alpha_q * mdct_P_i)
                        mdct_P = sub_preds  # list of 8 arrays for SHORT

            decodedData = self.Decode(scaleFactor, bitAlloc, mantissa, overallScaleFactor,
                                      codingParams, mdct_pred=mdct_P)
            raw_decoded.append(decodedData)

        # Handle M/S↔L/R mode transition: convert overlapAndAdd to match current block's domain
        prev_use_ms = getattr(codingParams, 'prev_use_ms', False)
        if codingParams.nChannels == 2 and use_ms != prev_use_ms:
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
            buf = codingParams.search_buffer[iCh]
            buf[0:-halfN] = buf[halfN:]
            buf[-halfN:] = 0
            buf[-N:] += lr_for_buffer[iCh]
            buf[-N:] = np.clip(buf[-N:], -1, 1)

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
        return


    def WriteDataBlock(self, data, codingParams):
        """
        Writes a block of signed-fraction data to a PACFile object that has
        already executed OpenForWriting()"""

        # Combine this block with the prior block to form full MDCT input (L/R domain)
        fullBlockData_ = []
        for iCh in range(codingParams.nChannels):
            fullBlockData_.append(np.concatenate((codingParams.priorBlock[iCh], data[iCh])))
        codingParams.priorBlock = data

        sfBands_short = ShortBlockSFBands(codingParams.nMDCTLines_short, codingParams.sampleRate)
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

        # First encode pass: detect block type only (discard results)
        saved_block_index = getattr(codingParams, 'blockIndex', 0)
        saved_prev_block_type = codingParams.prevBlockType
        self.Encode(fullBlockData_, codingParams)
        current_block_type = codingParams.blockType

        # Compute M/S search buffers on-the-fly from L/R buffers (search buffers always hold L/R)
        if use_ms and codingParams.nChannels == 2:
            buf_L = codingParams.search_buffer[0]
            buf_R = codingParams.search_buffer[1]
            search_bufs = [(buf_L + buf_R) * 0.5, (buf_L - buf_R) * 0.5]
        else:
            search_bufs = [codingParams.search_buffer[iCh] for iCh in range(codingParams.nChannels)]

        # Per-channel: search buffer for best prediction, compute residual
        fullBlockData = []
        ranges = []
        offsets = []
        pred_mdcts = []    # mdct_P per channel (or None)
        alpha_idxs = []    # 3-bit gain index
        alpha_qs = []      # quantized gain scalar
        enable_flags_list = []  # bool array shape (nBands,) per channel — for bitstream
        enable_masks = []       # float array shape (halfN,) per channel — for buffer/encoder
        corrections = []        # per-channel encoder correction (or None)
        pred_bits_freed = []    # bits to subtract from budget due to prediction
        for iCh in range(codingParams.nChannels):
            if current_block_type != SHORT:
                window = WindowForBlockType(current_block_type, N, N_short)
                mdct_X = MDCT(window * fullBlockData_[iCh], halfN, halfN)[:halfN]
                range_type, pcm_residual, rel_offset, mdct_P, alpha_idx, alpha_q = get_best_region(
                    mdct_X, fullBlockData_[iCh], codingParams,
                    search_bufs[iCh]
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
                        correction = alpha_q * mdct_P * (1.0 - enable_m)
                else:
                    enable_f = np.zeros(codingParams.sfBands.nBands, dtype=bool)
                    enable_m = np.zeros(halfN)
                    correction = None
            else:  # SHORT block only (START/STOP handled in the != SHORT branch above)
                # SHORT block: same lag search on full N_long window, one lag for all 8 sub-blocks
                window = WindowForBlockType(LONG, N, N_short)
                mdct_X = MDCT(window * fullBlockData_[iCh], halfN, halfN)[:halfN]
                range_type, pcm_residual, rel_offset, mdct_P, alpha_idx, alpha_q = get_best_region(
                    mdct_X, fullBlockData_[iCh], codingParams, search_bufs[iCh]
                )
                if range_type is not None:
                    # Accept only if overall time-domain energy is reduced (no per-band enables for SHORT)
                    if np.dot(pcm_residual, pcm_residual) >= np.dot(fullBlockData_[iCh], fullBlockData_[iCh]):
                        range_type, pcm_residual, rel_offset, mdct_P, alpha_idx, alpha_q = \
                            None, fullBlockData_[iCh], 0, None, 0, 1.0
                enable_f = np.array([], dtype=bool)  # SHORT: no per-band enables
                enable_m = np.zeros(halfN)
                correction = None
            # 6 dB/bit: bits freed = 20·log10(peak_orig/peak_res)/6 · nLines per enabled band
            # Only computed for LONG/START/STOP blocks (SHORT has no per-band enables)
            bits_freed = 0.0
            if range_type is not None and current_block_type != SHORT:
                for iBand in range(codingParams.sfBands.nBands):
                    if enable_f[iBand]:
                        lo = codingParams.sfBands.lowerLine[iBand]
                        hi = codingParams.sfBands.upperLine[iBand] + 1
                        peak_orig = np.max(np.abs(mdct_X[lo:hi]))
                        peak_res  = np.max(np.abs(residual_full[lo:hi]))
                        if peak_orig > 1e-12 and peak_res > 1e-12:
                            db_saved = max(0.0, 20.0 * np.log10(peak_orig / peak_res))
                            bits_freed += (db_saved / 6.0) * codingParams.sfBands.nLines[iBand]
            pred_bits_freed.append(bits_freed)
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

        # Restore block state; second encode on residuals with original signal for masking
        codingParams.blockIndex = saved_block_index
        codingParams.prevBlockType = saved_prev_block_type
        codingParams.masking_signals = fullBlockData_
        codingParams.mdct_pred_corrections = corrections
        codingParams.pred_bits_freed = pred_bits_freed
        (scaleFactor, bitAlloc, mantissa, overallScaleFactor) = self.Encode(fullBlockData, codingParams)
        codingParams.masking_signals = None
        codingParams.mdct_pred_corrections = None
        codingParams.pred_bits_freed = None

        # Buffer update: decode all channels (in M/S or L/R domain), convert to L/R, update buffers
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
            else:  # SHORT block only (START/STOP use the LONG path above)
                # SHORT block: expand compact mantissas per sub-block, decode with prediction
                full_mant = [codec.ExpandMantissa(mantissa[iCh][i], bitAlloc[iCh][i], sfBands_short, halfN_short)
                             for i in range(N_SHORT_BLOCKS)]
                sub_preds = None
                if ranges[iCh] is not None:
                    start_offset_map = {'quarter': codingParams.numSamplesQuarterNote,
                                        'half':    codingParams.numSamplesHalfBar,
                                        'bar':     codingParams.numSamplesBar}
                    s_off = start_offset_map[ranges[iCh]]
                    best = len(search_bufs[iCh]) - s_off + offsets[iCh]
                    candidate = search_bufs[iCh][best : best + N]
                    pad = N // 4 - N_short // 4
                    sub_preds = []
                    for i in range(N_SHORT_BLOCKS):
                        pcm_sub = candidate[pad + i*halfN_short : pad + i*halfN_short + N_short]
                        w_s = WindowForBlockType(SHORT, N, N_short)
                        mdct_P_i = MDCT(w_s * pcm_sub, halfN_short, halfN_short)[:halfN_short]
                        sub_preds.append(alpha_qs[iCh] * mdct_P_i)
                decodedData = codec.Decode(
                    scaleFactor[iCh], bitAlloc[iCh], full_mant,
                    overallScaleFactor[iCh], codingParams, mdct_pred=sub_preds
                )
            decoded_channels.append(decodedData)

        # Convert M/S decoded data back to L/R before storing in search buffer
        if use_ms and codingParams.nChannels == 2:
            M_hat, S_hat = decoded_channels[0], decoded_channels[1]
            lr_for_buffer = [M_hat + S_hat, M_hat - S_hat]
        else:
            lr_for_buffer = decoded_channels
        for iCh in range(codingParams.nChannels):
            buf = codingParams.search_buffer[iCh]
            buf[0:-halfN] = buf[halfN:]
            buf[-halfN:] = 0
            buf[-N:] += lr_for_buffer[iCh]
            buf[-N:] = np.clip(buf[-N:], -1, 1)

        # Write to file (channel 0 carries the 1-bit M/S flag for stereo)
        for iCh in range(codingParams.nChannels):
            entropy_pbs = []
            nBits = 4  # 2 bits block type + 2 bits prediction type
            if iCh == 0 and codingParams.nChannels == 2:
                nBits += 1  # M/S flag
            if ranges[iCh] is not None:
                nBits += 9 + 3  # sign+offset + gain
                if codingParams.blockType == LONG:
                    nBits += codingParams.sfBands.nBands  # per-band enables only for LONG
            if codingParams.blockType == SHORT:
                for i in range(N_SHORT_BLOCKS):
                    entropy_pb = codingParams.entropyCoder_short.encode_block(
                        mantissa[iCh][i], bitAlloc[iCh][i], sfBands_short
                    )
                    entropy_pbs.append(entropy_pb)
                    nBits += codingParams.nScaleBits
                    for iBand in range(sfBands_short.nBands):
                        nBits += codingParams.nMantSizeBits + codingParams.nScaleBits
                    nBits += entropy_pb.nBits
            else:
                entropy_pb = codingParams.entropyCoder_long.encode_block(
                    mantissa[iCh], bitAlloc[iCh], codingParams.sfBands
                )
                nBits += codingParams.nScaleBits
                for iBand in range(codingParams.sfBands.nBands):
                    nBits += codingParams.nMantSizeBits + codingParams.nScaleBits
                nBits += entropy_pb.nBits

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
                if codingParams.blockType == LONG:
                    for iBand in range(codingParams.sfBands.nBands):
                        pb.WriteBits(1 if enable_flags_list[iCh][iBand] else 0, 1)

            if codingParams.blockType == SHORT:
                for i in range(N_SHORT_BLOCKS):
                    pb.WriteBits(overallScaleFactor[iCh][i], codingParams.nScaleBits)
                    for iBand in range(sfBands_short.nBands):
                        ba = bitAlloc[iCh][i][iBand]
                        if ba: ba -= 1
                        pb.WriteBits(ba, codingParams.nMantSizeBits)
                        pb.WriteBits(scaleFactor[iCh][i][iBand], codingParams.nScaleBits)
                    pb.WriteBits(entropy_pbs[i].buffer, entropy_pbs[i].nBits)
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
    rmc("inputs/castanets.wav", "outputs/castanets_rmc.wav", rate_kb=192)
    print(f"\nDone in {time.time() - elapsed:.1f}s")
