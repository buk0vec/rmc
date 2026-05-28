# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
# "llvmlite==0.45.0",
# "madmom @ git+https://github.com/CPJKU/madmom.git@27f032e8947204902c675e5e341a3faf5dc86dae",
# "matplotlib>=3.10.6",
# "numba>=0.61.2",
# "numpy>=2",
# "scipy>=1.16.3",
# "tqdm>=4.67.3",
# ]
# ///

"""
xrmc.py

Awesome CLI for encoding/decoding RMC files.

"""

from pathlib import Path

import numpy as np

from blockswitching import K_ATTACK_MAX, LONG, STOP
from features import RMCFeatures
from pcmfile import *  # to get access to WAV file handling
from rmcfile import *  # to get access to RMC file handling
from spe import RingBuffer

TEMPO_EXCERPT_SECONDS = 30


def detect_tempo(inFilename: str) -> int:
    """Estimate the dominant tempo of an audio file using madmom's RNN beat tracker."""
    from madmom.audio.signal import Signal
    from madmom.features.beats import RNNBeatProcessor
    from madmom.features.tempo import (
        CombFilterTempoHistogramProcessor,
        TempoEstimationProcessor,
    )

    act_proc = RNNBeatProcessor()
    hist_proc = CombFilterTempoHistogramProcessor(fps=100)
    # method=None suppresses the deprecation warning for the default 'comb' method
    tempo_proc = TempoEstimationProcessor(
        fps=100, method=None, histogram_processor=hist_proc
    )

    sig = Signal(
        inFilename,
        sample_rate=44100,
        num_channels=1,
        stop=TEMPO_EXCERPT_SECONDS * 44100,
    )
    activations = act_proc(sig)
    tempi = tempo_proc(activations)
    if len(tempi) == 0:
        print("Auto-tempo detection returned no result — defaulting to 120 BPM")
        return 120
    raw_tempo = float(tempi[0][0])
    # Octave-fold into [50, 200] range
    while raw_tempo < 50.0:
        raw_tempo *= 2.0
    while raw_tempo > 200.0:
        raw_tempo /= 2.0
    return int(round(raw_tempo))


def print_flavor():
    flavor = """
⠀⠀⠀⠀⠀⢀⣼⣿⣆⠀⠀⠀
⠀⠀⠀⠀⠀⣾⡿⠛⢻⡆⠀⠀
⠀⠀⠀⠀⢰⣿⠀⠀⢸⡇⠀⠀
⠀⠀⠀⠀⠸⡇⠀⢀⣾⠇⠀⠀          ___           ___           ___
⠀⠀⠀⠀⠀⣿⣤⣿⡟⠀⠀⠀         /\\  \\         /\\__\\         /\\  \\
⠀⠀⠀⠀⣠⣾⣿⣿⠀⠀⠀⠀        /::\\  \\       /::|  |       /::\\  \\
⠀⣴⣿⡿⠋⠀⢻⡉⠀⠀⠀⠀       /:/\\:\\  \\     /:|:|  |      /:/\\:\\  \\
⢰⣿⡟⠀⢀⣴⣿⣿⣿⣿⣦⠀      /::\\~\\:\\  \\   /:/|:|__|__   /:/  \\:\\  \\
⢸⡿⠀⠀⣿⠟⠛⣿⠟⠛⣿⣧     /:/\\:\\ \\:\\__\\ /:/ |::::\\__\\ /:/__/ \\:\\__\\
⠘⣿⡀⠀⢿⡀⠀⢻⣤⠖⢻⡿     \\/_|::\\/:/  / \\/__/~~/:/  / \\:\\  \\  \\/__/
⠀⠘⢷⣄⠈⠙⠦⠸⡇⢀⡾⠃        |:|::/  /        /:/  /   \\:\\  \\
⠀⠀⠀⠙⠛⠶⠤⠶⣿⠉⠀⠀        |:|\\/__/        /:/  /     \\:\\  \\
⠀⠀⠀⠀⠀⠀⠀⠀⢹⡇⠀⠀        |:|  |         /:/  /       \\:\\__\\
⠀⠀⢀⣴⣾⣿⣆⠀⠈⣧⠀⠀         \\|__|         \\/__/         \\/__/
⠀⠀⠈⣿⣿⡿⠃⠀⣰⡏⠀⠀
⠀⠀⠀⠈⣙⠓⠒⠚⠉⠀⠀
"""
    print(flavor)


def Encode(
    inFilename,
    features: RMCFeatures,
    codedFilename=None,
    nMDCTLines=1024,
    nScaleBits=5,
    nMantSizeBits=3,
    kbps=128,
    targetBitsPerSample=None,
    tempo: int | None = None,
    verbose: bool = False,
):
    if codedFilename is None:
        codedFilename = f"{Path(inFilename).stem}.rmc"

    if tempo is None:
        tempo = detect_tempo(inFilename)
        print(f"Auto-detected tempo: {tempo} BPM")

    inFile = PCMFile(inFilename)
    outFile = RMCFile(codedFilename, features=features)

    codingParams = inFile.OpenForReading()

    codingParams.nMDCTLines = nMDCTLines
    codingParams.nScaleBits = nScaleBits
    codingParams.nMantSizeBits = nMantSizeBits
    codingParams.nSamplesPerBlock = codingParams.nMDCTLines
    codingParams.tempo = tempo

    if targetBitsPerSample is None:
        targetBitsPerSample = kbps * 1000 / codingParams.sampleRate
    else:
        kbps = int(targetBitsPerSample * codingParams.sampleRate / 1000)

    if verbose:
        print(f"\nEncoding {inFilename} -> {codedFilename} at {kbps} kb/s")

    if verbose:
        if features.BLOCK_SWITCHING:
            print("Block switching enabled (streaming transient detection)")
        else:
            print("Block switching disabled, using all LONG blocks")

    codingParams.targetBitsPerSample = targetBitsPerSample

    # open the output file
    outFile.OpenForWriting(codingParams)  # (includes writing header)

    halfN = codingParams.nMDCTLines
    halfN_short = codingParams.nMDCTLines_short

    total_samples = codingParams.numSamples
    processed_samples = 0

    from tqdm import tqdm

    rb = RingBuffer(inFile, codingParams, halfN)

    pbar = tqdm(total=total_samples, unit="samp", desc="Encoding", disable=not verbose)

    pos = 0
    prev = LONG

    while True:
        if prev in (LONG, STOP):
            hit = rb.spe_peek() if features.BLOCK_SWITCHING else None
            if hit is not None:
                k = max(0, min(K_ATTACK_MAX, (hit - pos - halfN) // halfN_short))
                codingParams.nSamplesPerBlock = (15 + k) * halfN_short
                codingParams.next_k_attack = k
            else:
                codingParams.nSamplesPerBlock = halfN
                codingParams.next_k_attack = -1
        elif codingParams.block_queue:
            codingParams.nSamplesPerBlock = codingParams.block_queue[0]["b"]
        else:
            codingParams.nSamplesPerBlock = halfN_short

        n = codingParams.nSamplesPerBlock
        data = rb.step(n)
        if data is None:
            break
        outFile.WriteDataBlock(data, codingParams)
        pos = codingParams.currentSamplePos
        prev = codingParams.blockType
        processed_samples += n
        pbar.update(n)

    pbar.close()

    inFile.Close(codingParams)
    outFile.Close(codingParams)
    return codingParams


def Decode(
    inFilename,
    features: RMCFeatures | None = None,
    outFilename=None,
    verbose: bool = False,
):
    if outFilename is None:
        outFilename = f"{Path(inFilename).stem}.wav"

    if verbose:
        print(f"\nDecoding {inFilename} -> {outFilename}")

    inFile = RMCFile(inFilename, features=features or RMCFeatures())
    outFile = PCMFile(outFilename)

    codingParams = inFile.OpenForReading()
    codingParams.bitsPerSample = 16

    outFile.OpenForWriting(codingParams)

    total_samples = codingParams.numSamples  # original source length
    # Skip the first nMDCTLines decoded samples (MDCT startup latency) so that
    # the output is aligned with the input file in time.
    skip_remaining = codingParams.nMDCTLines
    write_remaining = total_samples

    from tqdm import tqdm

    pbar = tqdm(total=total_samples, unit="samp", desc="Decoding", disable=not verbose)

    while True:
        data = inFile.ReadDataBlock(codingParams)
        if not data:
            break

        if skip_remaining > 0:
            skip = min(skip_remaining, len(data[0]))
            data = [ch[skip:] for ch in data]
            skip_remaining -= skip

        if len(data[0]) == 0:
            continue

        if write_remaining <= 0:
            break

        n = len(data[0])
        if n > write_remaining:
            data = [ch[:write_remaining] for ch in data]
            n = write_remaining

        outFile.WriteDataBlock(data, codingParams)
        write_remaining -= n
        pbar.update(n)

        if write_remaining <= 0:
            break
    pbar.close()

    inFile.Close(codingParams)
    outFile.Close(codingParams)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(prog="xrmc", description="RMC audio codec CLI")
    parser.add_argument("-v", "--verbose", action="store_true")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "-c",
        "--compress",
        nargs="+",
        metavar="FILE",
        help="Encode: input.wav [output.rmc]",
    )
    mode.add_argument(
        "-d",
        "--decompress",
        nargs="+",
        metavar="FILE",
        help="Decode: input.rmc [output.wav]",
    )

    parser.add_argument(
        "-b",
        "--bitrate",
        type=int,
        default=128,
        help="Bitrate in kbps (encode only, default: 128)",
    )
    parser.add_argument(
        "-t",
        "--tempo",
        type=int,
        default=None,
        help="Tempo in BPM (encode only, default: auto-detect via madmom)",
    )

    parser.add_argument(
        "--tdbs",
        action="store_true",
        help="Enable transient detection + block switching",
    )
    parser.add_argument(
        "--pred", action="store_true", help="Enable rhythmic prediction"
    )
    parser.add_argument("--entropy", action="store_true", help="Enable entropy coding")

    args = parser.parse_args()

    features = RMCFeatures(
        BLOCK_SWITCHING=args.tdbs,
        PREDICTION=args.pred,
        ENTROPY_CODING=args.entropy,
    )

    print_flavor()

    if args.compress:
        inFile = args.compress[0]
        outFile = args.compress[1] if len(args.compress) > 1 else None
        Encode(
            inFile,
            features,
            codedFilename=outFile,
            kbps=args.bitrate,
            tempo=args.tempo,
            verbose=args.verbose,
        )
    else:
        inFile = args.decompress[0]
        outFile = args.decompress[1] if len(args.decompress) > 1 else None
        Decode(inFile, features=features, outFilename=outFile, verbose=args.verbose)
