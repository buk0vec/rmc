"""
2026 copyrights Marina Bosi & Rich Goldberg
"""
import time  # just for printing elapsed time
import numpy as np
from scipy.io import wavfile
from rmcfile import *  # to get access to RMC file handling
from pcmfile import *  # to get access to WAV file handling
from TransientDetction import detect_transient_blocks


def EncodeDecode(inFilename="input.wav",
                 outFilename="output.wav",
                 codedFilename="coded.pac",
                 nMDCTLines=1024,
                 nScaleBits=3,
                 nMantSizeBits=4,
                 targetBitsPerSample=2.9,
                 progressCallback=None, tempo = 120.0):
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
    print("\n\tPre-analyzing audio for transient detection...", end="", flush=True)
    sr_pre, audio_pre = wavfile.read(inFilename)
    # wavfile returns shape (nSamples,) for mono or (nSamples, nChannels) for stereo
    # normalize to [-1, 1] if integer PCM before converting to float
    if audio_pre.dtype not in (np.float32, np.float64):
        audio_pre = audio_pre / (2 ** 15)
    if audio_pre.ndim == 1:
        audio_pre = np.vstack([audio_pre.astype(np.float64), audio_pre.astype(np.float64)])
    else:
        audio_pre = audio_pre.T.astype(np.float64)  # -> (nChannels, nSamples)

    transient_map = detect_transient_blocks(audio_pre, sr_pre, nMDCTLines)
    del audio_pre  # free memory before encoding
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
            codingParams.targetBitsPerSample = targetBitsPerSample
            codingParams.nSamplesPerBlock = codingParams.nMDCTLines
            codingParams.tempo = tempo
            # transient map from pre-analysis
            codingParams.transientBlocks = transient_map
            codingParams.blockIndex = 0
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
    EncodeDecode(inFilename='Van_124BPM.wav', codedFilename='coded_128k.pac',
                 outFilename='VANoutput_128k.wav', targetBitsPerSample=128000/44100, tempo =124)
    EncodeDecode(inFilename='Van_124BPM.wav', codedFilename='coded_192k.pac',
                 outFilename='VANoutput_192k.wav', targetBitsPerSample=192000/44100, tempo = 124)
