"""
2026 copyrights Marina Bosi & Rich Goldberg
"""
import time
from xrmc import Encode, Decode

def EncodeDecode(inFilename="input.wav",
                 outFilename="output.wav",
                 codedFilename="coded.pac",
                 nMDCTLines=1024,
                 nScaleBits=3,
                 nMantSizeBits=5,
                 targetBitsPerSample=2.4,
                 tempo: int = 120,
                 verbose: bool = True):
    """Encodes input WAV file inFilename into perceptually coded file
codedFilename and then decodes that file into output WAV file outFilename.
Allowed parameters are the number of indep MDCT lines per block (half the block
length), the number of block floating point scale factor bits, the number of
bits to encode the mantissa bit allocations, and the target bit rate in units
of bits per sample.
"""
    if verbose:
        print("\nRunning the RMC coder (" + inFilename +
              " -> " + codedFilename + " -> " +
              outFilename + "):")
    elapsed = time.time()

    Encode(
        inFilename,
        codedFilename=codedFilename,
        nMDCTLines=nMDCTLines,
        nScaleBits=nScaleBits,
        nMantSizeBits=nMantSizeBits,
        targetBitsPerSample=targetBitsPerSample,
        tempo=tempo,
        verbose=verbose,
    )

    Decode(
        codedFilename,
        outFilename=outFilename,
        verbose=verbose,
    )

    elapsed = time.time() - elapsed
    if verbose:
        print("\nDone with Encode/Decode test\n")
        print(elapsed, " seconds elapsed")


if __name__ == "__main__":
    EncodeDecode(inFilename='inputs/Van_124.wav', codedFilename='van_base.pac',
                 outFilename='van_124_base.wav', targetBitsPerSample=128000/44100, tempo=124)
    # EncodeDecode(inFilename='Van_124.wav', codedFilename='coded_192k_ms.pac',
    #              outFilename='VANoutput_192k_ms.wav', targetBitsPerSample=192000/44100, tempo=124)
