"""
2026 copyrights Marina Bosi & Rich Goldberg
"""
import time  # just for printing elapsed time
from pacfile import * # to get access to PAC file handling
from pcmfile import * # to get access to WAV file handling



def EncodeDecode(inFilename="input.wav",
                 outFilename="output.wav",
                 codedFilename="coded.pac",
                 nMDCTLines=1024,
                 nScaleBits=3,
                 nMantSizeBits=4,
                  targetBitsPerSample = 2.9,
                  progressCallback = None):
    """Encodes input WAV file inFilename into perceptually coded file 
codedFilename and then decodes that file into output WAV file outFilename.
Allowed parameters are the number of indep MDCT lines per block (half the block
length), the number of block floating point scale factor bits, the number of
bits to encode the mantissa bit allocations, and the target bit rate in units
of bits per sample. 
"""                     
    print ("\nRunning the PAC coder (" + inFilename +
            " -> " + codedFilename + " -> " +
            outFilename+"):")
    elapsed = time.time()
    
    for iDir, Direction in enumerate(("Encode", "Decode")):
    
        # create the audio file objects
        if Direction == "Encode":
            print( "\n\tEncoding input PCM file...", end="")
            inFile= PCMFile(inFilename)
            outFile = PACFile(codedFilename)
        else: # "Decode"
            print( "\n\tDecoding coded PAC file...",end="")
            inFile = PACFile(codedFilename)
            outFile= PCMFile(outFilename)
        # only difference is file names and type of AudioFile object
    
        # open input file
        codingParams=inFile.OpenForReading()  # (includes reading header)
    
        # pass parameters to the output file
        if Direction == "Encode":
            # set additional parameters that are needed for PAC file
            # (beyond those set by the PCM file on open)
            codingParams.nMDCTLines = nMDCTLines
            codingParams.nScaleBits = nScaleBits
            codingParams.nMantSizeBits = nMantSizeBits
            codingParams.targetBitsPerSample = targetBitsPerSample
            # tell the PCM file how large the block size is
            codingParams.nSamplesPerBlock = codingParams.nMDCTLines
        else: # "Decode"
            # set PCM parameters (the rest is same as set by PAC file on open)
            codingParams.bitsPerSample = 16
        # only difference is in setting up the output file parameters
    
    
        # open the output file
        outFile.OpenForWriting(codingParams) # (includes writing header)
    
        # Read the input file and pass its data to the output file to be written
        total_samples = codingParams.numSamples
        processed_samples = 0
        
        while True:
            data=inFile.ReadDataBlock(codingParams)
            if not data: break  # we hit the end of the input file
            outFile.WriteDataBlock(data,codingParams)
            # Progress update
            processed_samples += len(data[0])
            if progressCallback and total_samples > 0:
                # Calculate percentage for this pass (0-50 or 50-100)
                pass_progress = min(1.0, processed_samples / total_samples) * 50
                overall_progress = (iDir * 50) + pass_progress
                progressCallback(overall_progress)
                
            print( ".",end="")  # just to signal how far we've gotten to user
        # end loop over reading/writing the blocks
    
        # close the files
        inFile.Close(codingParams)
        outFile.Close(codingParams)
    # end of loop over Encode/Decode
    
    elapsed = time.time()-elapsed
    print( "\nDone with Encode/Decode test\n")
    print( elapsed ," seconds elapsed")


if __name__ == "__main__":
	EncodeDecode(inFilename='harpsichord.wav', targetBitsPerSample=1.5)  # use default parameters