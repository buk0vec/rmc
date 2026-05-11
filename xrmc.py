# /// script
# requires-python = ">=3.12"
# dependencies = [
# "llvmlite==0.45.0",
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

from features import RMCFeatures
from pcmfile import *  # to get access to WAV file handling
from rmcfile import *  # to get access to RMC file handling
from simple_run import detectTransients
from spe import detectTransientsSPESamples


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
    nMantSizeBits=5,
    kbps=128,
    targetBitsPerSample=None,
    tempo: int = 120,
    verbose: bool = False,
):
    if codedFilename is None:
        codedFilename = f"{Path(inFilename).stem}.rmc"

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

    if features.BLOCK_SWITCHING:
        if features.AC2A_BLOCK_SWITCHING:
            events = detectTransientsSPESamples(inFilename, nMDCTLines=nMDCTLines, verbose=verbose)
        else:
            events = detectTransients(
                inFilename,
                verbose=False,
                return_events=True,
                forceBlockSize=nMDCTLines,
            )
        if features.AC2A_BLOCK_SWITCHING:
            # Exact sample positions — no grid alignment or shift heuristics needed.
            # Enforce minimum 2048-sample spacing so cascades can never collide.
            # min cascade = (15+0)*halfN_short + halfN_short + halfN_short + halfN
            #             = 17*halfN_short + halfN  (halfN_short = nMDCTLines//16)
            min_spacing = 17 * (nMDCTLines // 16) + nMDCTLines  # 2112 samples
            transient_positions = []
            last_pos = -min_spacing
            for e in events:
                si = int(e["sample_index"])
                if si - last_pos >= min_spacing and si >= nMDCTLines:
                    transient_positions.append(si)
                    last_pos = si
            transient_map = {}  # unused in AC2A path
            if verbose:
                print(f"TransientDetection: {len(transient_positions)} events (exact positions)")
        else:
            # Edler fallback: grid-aligned block-based map with shift heuristic
            transient_map = {}
            n_shift_m1 = 0
            n_shift_0 = 0
            for event in events:
                block = int(event["block"])
                p = int(event["sample_offset"])
                if p < 448:
                    shift = -1
                elif p > 576:
                    shift = 0
                else:
                    shift = -1 if p < 512 else 0
                shifted_block = max(block + shift, 0)
                k_attack = min(p // (nMDCTLines // 8), 7)
                transient_map[shifted_block] = k_attack
                if shift == -1:
                    n_shift_m1 += 1
                else:
                    n_shift_0 += 1
            transient_positions = []
            if verbose:
                print(
                    f"TransientDetection: {len(transient_map)} blocks "
                    f"(shift: -1={n_shift_m1}, 0={n_shift_0})"
                )
    else:
        transient_map = {}
        transient_positions = []
        if verbose:
            print("Block switching disabled, using all LONG blocks")

    codingParams.transientBlocks = transient_map
    codingParams.transientPositions = transient_positions
    codingParams.blockIndex = 0

    # Adjust targetBitsPerSample so SHORT blocks' boosted budget
    # doesn't push the file over the target bitrate.
    total_blocks = (
        int(np.ceil(codingParams.numSamples / nMDCTLines)) + 1
    )  # +1 for flush block
    n_short = len(transient_map)
    n_long = total_blocks - n_short
    short_budget_factor = (
        2.0 if features.SHORT_BLOCK_BITBOOST else 1.0
    )  # must match codec.py
    if n_short > 0 and short_budget_factor != 1.0:
        adjusted_bps = (
            targetBitsPerSample
            * total_blocks
            / (n_long + short_budget_factor * n_short)
        )
        if verbose:
            print(
                f"Adjusted bps: {targetBitsPerSample:.3f} -> {adjusted_bps:.3f} "
                f"({n_short} short / {total_blocks} total blocks)"
            )
        codingParams.targetBitsPerSample = adjusted_bps
    else:
        codingParams.targetBitsPerSample = targetBitsPerSample

    # open the output file
    outFile.OpenForWriting(codingParams)  # (includes writing header)
    # Read the input file and pass its data to the output file to be written
    total_samples = codingParams.numSamples
    processed_samples = 0

    from tqdm import tqdm

    pbar = tqdm(total=total_samples, unit="samp", desc="Encoding", disable=not verbose)

    while True:
        data = inFile.ReadDataBlock(codingParams)
        if not data:
            break
        outFile.WriteDataBlock(data, codingParams)
        n = len(data[0])
        processed_samples += n
        pbar.update(n)
    pbar.close()

    inFile.Close(codingParams)
    outFile.Close(codingParams)


def Decode(
    inFilename,
    features: RMCFeatures,
    outFilename=None,
    verbose: bool = False,
):
    if outFilename is None:
        outFilename = f"{Path(inFilename).stem}.wav"

    if verbose:
        print(f"\nDecoding {inFilename} -> {outFilename}")

    inFile = RMCFile(inFilename, features=features)
    outFile = PCMFile(outFilename)

    codingParams = inFile.OpenForReading()
    codingParams.bitsPerSample = 16

    outFile.OpenForWriting(codingParams)

    total_samples = codingParams.numSamples
    processed_samples = 0

    from tqdm import tqdm

    pbar = tqdm(total=total_samples, unit="samp", desc="Decoding", disable=not verbose)

    while True:
        data = inFile.ReadDataBlock(codingParams)
        if not data:
            break
        outFile.WriteDataBlock(data, codingParams)
        n = len(data[0])
        processed_samples += n
        pbar.update(n)
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
        default=120,
        help="Tempo in BPM (encode only, default: 120)",
    )

    args = parser.parse_args()

    print_flavor()

    if args.compress:
        inFile = args.compress[0]
        outFile = args.compress[1] if len(args.compress) > 1 else None
        Encode(
            inFile,
            RMCFeatures(),
            codedFilename=outFile,
            kbps=args.bitrate,
            tempo=args.tempo,
            verbose=args.verbose,
        )
    else:
        inFile = args.decompress[0]
        outFile = args.decompress[1] if len(args.decompress) > 1 else None
        Decode(inFile, RMCFeatures(), outFilename=outFile, verbose=args.verbose)
