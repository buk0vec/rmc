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
from blockswitching import LONG, START, SHORT, STOP, N_SHORT_BLOCKS, ShortBlockSFBands
from search import get_best_region, PRED_MAP

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
        myParams.search_range = 255 #byte per block + 1 for sign bit
        myParams.search_buffer = [np.zeros(myParams.numSamplesBar + myParams.search_range) for _ in range(myParams.nChannels)]

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
        # loop over channels (whose coded data are stored separately) and read in each data block
        data=[]
        for iCh in range(codingParams.nChannels):
            data.append(np.array([],dtype=np.float64))  # add location for this channel's data
            # read in string containing the number of bytes of data for this channel (but check if at end of file!)
            s=self.fp.read(calcsize("<L"))  # will be empty if at end of file
            if not s:
                # hit last block, see if final overlap and add needs returning, else return nothing
                if codingParams.overlapAndAdd:
                    overlapAndAdd=codingParams.overlapAndAdd
                    codingParams.overlapAndAdd=0  # setting it to zero so next pass will just return
                    return overlapAndAdd
                else:
                    return
            # not at end of file, get nBytes from the string we just read
            nBytes = unpack("<L",s)[0] # read it as a little-endian unsigned long
            # print(f"DEBUG ReadDataBlock: channel={iCh} nBytes={nBytes}")
            # read the nBytes of data into a PackedBits object to unpack
            pb = PackedBits()
            pb.SetPackedData( self.fp.read(nBytes) ) # PackedBits function SetPackedData() converts strings to internally-held array of bytes
            if pb.nBytes < nBytes:  raise "Only read a partial block of coded PACFile data"
            # extract block type
            # TODO: can't this just be one bit
            codingParams.blockType = pb.ReadBits(2)
            pred_type = pb.ReadBits(2)
            if pred_type is not PRED_MAP[None]:
                pred_sign = pb.ReadBits(1)
                pred_offset = pb.ReadBits(8)
                if pred_sign == 1:
                    pred_offset *= -1

            sfBands_short = ShortBlockSFBands(codingParams.nMDCTLines_short, codingParams.sampleRate)
            if codingParams.blockType == SHORT:
                overallScaleFactor = []
                scaleFactor = []
                bitAlloc = []
                mantissa = []   
                for i in range(N_SHORT_BLOCKS):
                    overallScaleFactor.append(pb.ReadBits(codingParams.nScaleBits)) 
                    sf_i=[]
                    ba_i=[]
                    for iBand in range(sfBands_short.nBands): # loop over each scale factor band to pack its data
                        ba = pb.ReadBits(codingParams.nMantSizeBits)
                        if ba: ba+=1  # no bit allocation of 1 so ba of 2 and up stored as one less
                        ba_i.append(ba)
                        sf_i.append(pb.ReadBits(codingParams.nScaleBits))
                                    # pb read pointer is now at the entropy block; decode_block reads the rest
                    mant_i = codingParams.entropyCoder_short.decode_block(
                        pb, ba_i, sfBands_short, codingParams.nMDCTLines_short
                    )
                    scaleFactor.append(sf_i)
                    bitAlloc.append(ba_i)
                    mantissa.append(mant_i)
                            
            else:
                # extract the data from the PackedBits object
                overallScaleFactor = pb.ReadBits(codingParams.nScaleBits)  # overall scale factor
                scaleFactor=[]
                bitAlloc=[]
                for _ in range(codingParams.sfBands.nBands):
                    ba = pb.ReadBits(codingParams.nMantSizeBits)
                    if ba: ba += 1
                    bitAlloc.append(ba)
                    scaleFactor.append(pb.ReadBits(codingParams.nScaleBits))
                
                # pb read pointer is now at the entropy block; decode_block reads the rest
                mantissa = codingParams.entropyCoder_long.decode_block(
                    pb, bitAlloc, codingParams.sfBands, codingParams.nMDCTLines
                )

            # (DECODE HERE) decode the unpacked data for this channel, overlap-and-add first half, and append it to the data array (saving other half for next overlap-and-add)
            decodedData = self.Decode(scaleFactor,bitAlloc,mantissa, overallScaleFactor,codingParams)

            if PRED_MAP[pred_type] is not None:
                if pred_type == 'quarter':
                    start_offset = codingParams.numSamplesQuarterNote
                elif pred_type == 'half':
                    start_offset =  codingParams.numSamplesHalfNote
                else:
                    start_offset = codingParams.numSamplesBar

                decodedData += codingParams.search_buffer[iCh][-start_offset + pred_offset: -start_offset + pred_offset + 2 * codingParams.nMDCTLines]


            data[iCh] = np.concatenate(
                ( data[iCh], 
                 np.maximum( -1., np.minimum( 1.,  # make sure result remains in [-1,1]
                        np.add( codingParams.overlapAndAdd[iCh],       
                                decodedData[:codingParams.nMDCTLines]) 
                            ) ) 
                 )
            )# data[iCh] is overlap-and-added data
            codingParams.overlapAndAdd[iCh] = decodedData[codingParams.nMDCTLines:]  # save other half for next pass
            codingParams.search_buffer[iCh][0:-codingParams.nMDCTLines] = codingParams.search_buffer[iCh][codingParams.nMDCTLines:] #shift over 1/2 N
            codingParams.search_buffer[iCh][-codingParams.nMDCTLines:] = 0
            codingParams.search_buffer[iCh][-2*codingParams.nMDCTLines:] += decodedData
            codingParams.search_buffer[iCh][-2*codingParams.nMDCTLines] = np.clip(codingParams.search_buffer[iCh][-2*codingParams.nMDCTLines], -1, 1)

        # end loop over channels, return signed-fraction samples for this block
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
        return


    def WriteDataBlock(self,data, codingParams):
        """
        Writes a block of signed-fraction data to a PACFile object that has
        already executed OpenForWriting()"""

        # combine this block of multi-channel data w/ the prior block's to prepare for MDCTs twice as long
        fullBlockData_=[]
        for iCh in range(codingParams.nChannels):
            fullBlockData_.append( np.concatenate( ( codingParams.priorBlock[iCh], data[iCh]) ) )
        codingParams.priorBlock = data  # current pass's data is next pass's prior block data

        # (ENCODE HERE) Encode the full block of multi=channel data
        (scaleFactor,bitAlloc,mantissa, overallScaleFactor) = self.Encode(fullBlockData_,codingParams)  # returns a tuple with all the block-specific info not in the file header
        sfBands_short = ShortBlockSFBands(codingParams.nMDCTLines_short, codingParams.sampleRate)
    
        fullBlockData=[]
        ranges = []
        offsets = []
        # Decode early for RMC buffer
        for iCh in range(codingParams.nChannels):
            decodedFullBlockData = self.Decode(scaleFactor[iCh],bitAlloc[iCh],mantissa[iCh], overallScaleFactor[iCh], codingParams)
            codingParams.search_buffer[iCh][0:-codingParams.nMDCTLines] = codingParams.search_buffer[iCh][codingParams.nMDCTLines:] #shift over 1/2 N
            codingParams.search_buffer[iCh][-codingParams.nMDCTLines:] = 0  
            codingParams.search_buffer[iCh][-2*codingParams.nMDCTLines:] += decodedFullBlockData
            codingParams.search_buffer[iCh][-2*codingParams.nMDCTLines] = np.clip(codingParams.search_buffer[iCh][-2*codingParams.nMDCTLines], -1, 1)
            best_range, best_residual, best_relative_offset = get_best_region(fullBlockData_[iCh], codingParams, codingParams.search_buffer[iCh], 0.8)
            fullBlockData.append(best_residual)
            offsets.append(best_relative_offset)
            ranges.append(best_range)

        (scaleFactor,bitAlloc,mantissa, overallScaleFactor) = self.Encode(fullBlockData,codingParams)  # returns a tuple with all the block-specific info not in the file header

        # for each channel, write the data to the output file
        for iCh in range(codingParams.nChannels):
            entropy_pbs = []
            nBits = 4 # the 2 addition for window type block switching + 2 bits for prediction
            if ranges[iCh] is not None:
                nBits += 9
            if codingParams.blockType == SHORT:
                for i in range(N_SHORT_BLOCKS):
                    # entropy-encode mantissas first so we know their size
                    entropy_pb = codingParams.entropyCoder_short.encode_block(
                        mantissa[iCh][i], bitAlloc[iCh][i], sfBands_short
                    )
                    entropy_pbs.append(entropy_pb)
                    # count header bits (overallScaleFactor + per-band metadata)
                    nBits += codingParams.nScaleBits
                    for iBand in range(sfBands_short.nBands):
                        nBits += codingParams.nMantSizeBits + codingParams.nScaleBits
                    nBits += entropy_pb.nBits
            else:
                entropy_pb = codingParams.entropyCoder_long.encode_block(
                    mantissa[iCh], bitAlloc[iCh], codingParams.sfBands
                )

                # count header bits (overallScaleFactor + per-band metadata)
                nBits += codingParams.nScaleBits
                for iBand in range(codingParams.sfBands.nBands):
                    nBits += codingParams.nMantSizeBits + codingParams.nScaleBits
                nBits += entropy_pb.nBits
            # print(f"DEBUG WriteDataBlock: channel={iCh} nBytes={nBytes}")
            nBytes = (nBits + BYTESIZE - 1) // BYTESIZE

            self.fp.write(pack("<L", int(nBytes)))

            pb = PackedBits()
            pb.Size(nBytes)
            pb.WriteBits(codingParams.blockType,2)
            pb.WriteBits(PRED_MAP[ranges[iCh]], 2)
            if ranges[iCh] is not None: 
                sign = 1 if offsets[iCh] < 0 else 0
                pb.WriteBits(sign, 1)
                pb.WriteBits(offsets[iCh], 8)

            if codingParams.blockType == SHORT:
                for i in range(N_SHORT_BLOCKS):
                    # write header metadata
                    pb.WriteBits(overallScaleFactor[iCh][i], codingParams.nScaleBits)
                    for iBand in range(sfBands_short.nBands):
                        ba = bitAlloc[iCh][i][iBand]
                        if ba: ba -= 1
                        pb.WriteBits(ba, codingParams.nMantSizeBits)
                        pb.WriteBits(scaleFactor[iCh][i][iBand], codingParams.nScaleBits)
                    pb.WriteBits(entropy_pbs[i].buffer, entropy_pbs[i].nBits)

            else:
                # write header metadata
                pb.WriteBits(overallScaleFactor[iCh], codingParams.nScaleBits)
                for iBand in range(codingParams.sfBands.nBands):
                    ba = bitAlloc[iCh][iBand]
                    if ba: ba -= 1
                    pb.WriteBits(ba, codingParams.nMantSizeBits)
                    pb.WriteBits(scaleFactor[iCh][iBand], codingParams.nScaleBits)

                # append entropy-coded mantissa block
                pb.WriteBits(entropy_pb.buffer, entropy_pb.nBits)
            self.fp.write(pb.GetPackedData())
        # end loop over channels, done writing coded data for all channels
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
        self.fp.close()


    def Encode(self,data,codingParams):
        """
        Encodes multichannel audio data and returns a tuple containing
        the scale factors, mantissa bit allocations, quantized mantissas,
        and the overall scale factor for each channel.
        """
        #Passes encoding logic to the Encode function defined in the codec module
        return codec.Encode(data,codingParams)

    def Decode(self,scaleFactor,bitAlloc,mantissa, overallScaleFactor,codingParams):
        """
        Decodes a single audio channel of data based on the values of its scale factors,
        bit allocations, quantized mantissas, and overall scale factor.
        """
        #Passes decoding logic to the Decode function defined in the codec module
        return codec.Decode(scaleFactor,bitAlloc,mantissa, overallScaleFactor,codingParams)








#-----------------------------------------------------------------------------

if __name__ == "__main__":
    from prepare_materials import rmc
    import time
    elapsed = time.time()
    rmc("inputs/castanets.wav", "outputs/castanets_rmc.wav", rate_kb=192)
    print(f"\nDone in {time.time() - elapsed:.1f}s")
