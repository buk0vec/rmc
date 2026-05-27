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
    nMantSizeBits=3,
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
    features_entropy = RMCFeatures(ENTROPY_CODING=True)
    features_ecpred = RMCFeatures(ENTROPY_CODING=True, PREDICTION=True)
    features_ecbs = RMCFeatures(ENTROPY_CODING=True, BLOCK_SWITCHING=True)
    features_all = RMCFeatures(
        PREDICTION=True, BLOCK_SWITCHING=True, ENTROPY_CODING=True
    )
    TARGET_KBPS = 80000
    SAMPLE_RATE = 44100
    target_bps = TARGET_KBPS / SAMPLE_RATE
    EncodeDecode(
        inFilename="ringnoord.wav",
        codedFilename="rn.rmc",
        outFilename="rn_base.wav",
        targetBitsPerSample=target_bps,
        tempo=164,
        features=features,
    )
    EncodeDecode(
        inFilename="ringnoord.wav",
        codedFilename="rn_pred.rmc",
        outFilename="rn_pred.wav",
        targetBitsPerSample=target_bps,
        tempo=164,
        features=features_pred,
    )
    EncodeDecode(
        inFilename="ringnoord.wav",
        codedFilename="rn_bs.rmc",
        outFilename="rn_bs.wav",
        targetBitsPerSample=target_bps,
        tempo=164,
        features=features_bs,
    )
    EncodeDecode(
        inFilename="ringnoord.wav",
        codedFilename="rn_bspred.rmc",
        outFilename="rn_bspred.wav",
        targetBitsPerSample=target_bps,
        tempo=164,
        features=features_bspred,
    )
    EncodeDecode(
        inFilename="ringnoord.wav",
        codedFilename="rn_entropy.rmc",
        outFilename="rn_entropy.wav",
        targetBitsPerSample=target_bps,
        tempo=164,
        features=features_entropy,
    )
    EncodeDecode(
        inFilename="ringnoord.wav",
        codedFilename="rn_ecpred.rmc",
        outFilename="rn_ecpred.wav",
        targetBitsPerSample=target_bps,
        tempo=164,
        features=features_ecpred,
    )
    EncodeDecode(
        inFilename="ringnoord.wav",
        codedFilename="rn_ecbs.rmc",
        outFilename="rn_ecbs.wav",
        targetBitsPerSample=target_bps,
        tempo=164,
        features=features_ecbs,
    )
    EncodeDecode(
        inFilename="ringnoord.wav",
        codedFilename="rn_all.rmc",
        outFilename="rn_all.wav",
        targetBitsPerSample=target_bps,
        tempo=164,
        features=features_all,
    )
