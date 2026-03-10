"""
2026 copyrights Marina Bosi & Rich Goldberg
"""
import time  # just for printing elapsed time
import numpy as np
from scipy.io import wavfile
from rmcfile import *  # to get access to RMC file handling
from pcmfile import *  # to get access to WAV file handling
from simple_run import detectTransients

def EncodeDecode(inFilename="input.wav",
                 outFilename="output.wav",
                 codedFilename="coded.pac",
                 nMDCTLines=1024,
                 nScaleBits=3,
                 nMantSizeBits=5,
                 targetBitsPerSample=2.4,
                 progressCallback=None, tempo: int = 120):
    """Encodes input WAV file inFilename into perceptually coded file
codedFilename and then decodes that file into output WAV file outFilename.
Allowed parameters are the number of indep MDCT lines per block (half the block
length), the number of block floating point scale factor bits, the number of
bits to encode the mantissa bit allocations, and the target bit rate in units
of bits per sample.
"""
    print("\nRunning the RMC coder (" + inFilename +
          " -> " + codedFilename + " -> " +
          outFilename + "):")
    elapsed = time.time()

    # ------------------------------------------------------------------
    # Pre-analysis: detect transients across the whole song before encoding
    # ------------------------------------------------------------------
    

    transient_map = detectTransients(inFilename)
    # Shift transient map back by 1 block for lookahead: the START block must
    # precede the transient so that SHORT blocks capture the actual attack.
    # Without this, the kick itself gets a LONG MDCT (START) and the post-kick
    # bass content lands in SHORT blocks with terrible low-frequency resolution.
    transient_map = {max(x - 1, 0): 0 for x in transient_map}
    
    print(f" found transients in {len(transient_map)} blocks")

    # ------------------------------------------------------------------
    # Encode then Decode
    # ------------------------------------------------------------------
    for iDir, Direction in enumerate(("Encode", "Decode")):

        # create the audio file objects
        if Direction == "Encode":
            print("\n\tEncoding input PCM file...", end="")
            inFile = PCMFile(inFilename)
            outFile = RMCFile(codedFilename)
        else:  # "Decode"
            print("\n\tDecoding coded RMC file...", end="")
            inFile = RMCFile(codedFilename)
            outFile = PCMFile(outFilename)

        # open input file
        codingParams = inFile.OpenForReading()  # (includes reading header)

        # pass parameters to the output file
        if Direction == "Encode":
            codingParams.nMDCTLines = nMDCTLines
            codingParams.nScaleBits = nScaleBits
            codingParams.nMantSizeBits = nMantSizeBits
            codingParams.nSamplesPerBlock = codingParams.nMDCTLines
            codingParams.tempo = tempo
            # transient map from pre-analysis
            codingParams.transientBlocks = transient_map
            codingParams.blockIndex = 0

            # Adjust targetBitsPerSample so SHORT blocks' 2x budget
            # doesn't push the file over the target bitrate.
            # We know how many SHORT blocks there will be from pre-analysis.
            total_blocks = int(np.ceil(codingParams.numSamples / nMDCTLines)) + 1  # +1 for flush block
            n_short = len(transient_map)  # each transient → 1 SHORT block
            n_long = total_blocks - n_short
            # Budget equation: n_long * B + n_short * 2*B = total_blocks * B_target
            # → B = B_target * total_blocks / (n_long + 2*n_short)
            short_budget_factor = 2.0  # must match the multiplier in codec.py
            adjusted_bps = targetBitsPerSample * total_blocks / (n_long + short_budget_factor * n_short)
            print(f"  Adjusted bps: {targetBitsPerSample:.3f} -> {adjusted_bps:.3f} "
                  f"({n_short} SHORT / {total_blocks} total blocks)")
            codingParams.targetBitsPerSample = adjusted_bps
        else:  # "Decode"
            codingParams.bitsPerSample = 16

        # open the output file
        outFile.OpenForWriting(codingParams)  # (includes writing header)

        # Read the input file and pass its data to the output file to be written
        total_samples = codingParams.numSamples
        processed_samples = 0

        while True:
            data = inFile.ReadDataBlock(codingParams)
            if not data: break  # we hit the end of the input file
            outFile.WriteDataBlock(data, codingParams)
            processed_samples += len(data[0])
            if progressCallback and total_samples > 0:
                pass_progress = min(1.0, processed_samples / total_samples) * 50
                overall_progress = (iDir * 50) + pass_progress
                progressCallback(overall_progress)

            print(".", end="")  # just to signal how far we've gotten to user
        # end loop over reading/writing the blocks

        # close the files
        inFile.Close(codingParams)
        outFile.Close(codingParams)
    # end of loop over Encode/Decode

    elapsed = time.time() - elapsed
    print("\nDone with Encode/Decode test\n")
    print(elapsed, " seconds elapsed")


if __name__ == "__main__":
    EncodeDecode(inFilename='./brooklyn_full.wav', codedFilename='brooklyn_full_72.pac',
                 outFilename='Brooklyn_72.wav', targetBitsPerSample=72000/44100, tempo = 97)
    # EncodeDecode(inFilename='Van_124.wav', codedFilename='coded_192k_ms.pac',
    #              outFilename='VANoutput_192k_ms.wav', targetBitsPerSample=192000/44100, tempo = 124)
