"""
2026 copyrights Marina Bosi & Rich Goldberg
"""

import time

from features import RMCFeatures
from xrmc import Decode, Encode


def EncodeDecode(
    inFilename="input.wav",
    outFilename="output.wav",
    codedFilename="coded.pac",
    nMDCTLines=1024,
    nScaleBits=5,
    nMantSizeBits=5,
    targetBitsPerSample=2.4,
    tempo: int = 120,
    verbose: bool = True,
    features: RMCFeatures = RMCFeatures(),
):
    """Encodes input WAV file inFilename into perceptually coded file
    codedFilename and then decodes that file into output WAV file outFilename.
    Allowed parameters are the number of indep MDCT lines per block (half the block
    length), the number of block floating point scale factor bits, the number of
    bits to encode the mantissa bit allocations, and the target bit rate in units
    of bits per sample.
    """
    if verbose:
        print(
            "\nRunning the RMC coder ("
            + inFilename
            + " -> "
            + codedFilename
            + " -> "
            + outFilename
            + "):"
        )
    elapsed = time.time()

    Encode(
        inFilename,
        features,
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
        features,
        outFilename=outFilename,
        verbose=verbose,
    )

    elapsed = time.time() - elapsed
    if verbose:
        print("\nDone with Encode/Decode test\n")
        print(elapsed, " seconds elapsed")


if __name__ == "__main__":
    # Base feature set
    features = RMCFeatures()
    # Feature set w/ prediction enabled
    features_pred = RMCFeatures(PREDICTION=True)
    # Feature set w/ block switching enabled
    features_bs = RMCFeatures(BLOCK_SWITCHING=True)
    # Block switching and prediction
    features_bspred = RMCFeatures(PREDICTION=True, BLOCK_SWITCHING=True)
    EncodeDecode(
        inFilename="ringnoord.wav",
        codedFilename="rn.pac",
        outFilename="rn_base.wav",
        targetBitsPerSample=80000 / 44100,
        tempo=164,
        features=features,
    )
    EncodeDecode(
        inFilename="ringnoord.wav",
        codedFilename="rn_pred.pac",
        outFilename="rn_pred.wav",
        targetBitsPerSample=80000 / 44100,
        tempo=164,
        features=features_pred,
    )
    EncodeDecode(
        inFilename="ringnoord.wav",
        codedFilename="rn_bs.pac",
        outFilename="rn_bs.wav",
        targetBitsPerSample=80000 / 44100,
        tempo=164,
        features=features_bs,
    )
    EncodeDecode(
        inFilename="ringnoord.wav",
        codedFilename="rn_bspred.pac",
        outFilename="rn_bspred.wav",
        targetBitsPerSample=80000 / 44100,
        tempo=164,
        features=features_bspred,
    )
