# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib>=3.10.8",
#     "numpy>=2.4.3",
#     "scipy>=1.17.1",
#     "tqdm>=4.67.3",
# ]
# ///

"""
xrmc.py

Awesome CLI for encoding/decoding RMC files.

"""

import time  # just for printing elapsed time
import numpy as np
from rmcfile import *  # to get access to RMC file handling
from pcmfile import *  # to get access to WAV file handling
from simple_run import detectTransients
from pathlib import Path

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

def Encode(inFilename,
            codedFilename=None,
            nMDCTLines=1024,
            nScaleBits=3,
            nMantSizeBits=5,
            kbps=128,
            tempo: int = 120,
            verbose: bool = False
    ):
    # ------------------------------------------------------------------
    # Pre-analysis: detect transients across the whole song before encoding
    # ------------------------------------------------------------------
        
    if codedFilename is None:
        codedFilename = f"{Path(inFilename).stem}.rmc"
        
    inFile = PCMFile(inFilename)
    outFile = RMCFile(codedFilename)

    # open input file
    codingParams = inFile.OpenForReading()  # (includes reading header)

    # pass parameters to the output file
    codingParams.nMDCTLines = nMDCTLines
    codingParams.nScaleBits = nScaleBits
    codingParams.nMantSizeBits = nMantSizeBits
    codingParams.nSamplesPerBlock = codingParams.nMDCTLines
    codingParams.tempo = tempo
    targetBitsPerSample = kbps * 1000 / codingParams.sampleRate

    if verbose:
        print(f"\nEncoding {inFilename} -> {codedFilename} at {kbps} kb/s")

    # This is too verbose for the CLI lol
    transient_map = detectTransients(inFilename, verbose=False)
    # Shift transient map back by 1 block for lookahead: the START block must
    # precede the transient so that SHORT blocks capture the actual attack.
    # Without this, the kick itself gets a LONG MDCT (START) and the post-kick
    # bass content lands in SHORT blocks with terrible low-frequency resolution.
    transient_map = {max(x - 1, 0): 0 for x in transient_map}

    codingParams.transientBlocks = transient_map
    codingParams.blockIndex = 0

    if verbose:
        print(f"TransientDetection complete: found transients in {len(transient_map)} blocks")

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
    if verbose:
        print(f"Adjusted bps: {targetBitsPerSample:.3f} -> {adjusted_bps:.3f} "
f"({n_short} Short / {total_blocks} total blocks)")
    
    codingParams.targetBitsPerSample = adjusted_bps
    
    # open the output file
    outFile.OpenForWriting(codingParams)  # (includes writing header)

    # Read the input file and pass its data to the output file to be written
    total_samples = codingParams.numSamples
    processed_samples = 0

    from tqdm import tqdm
    pbar = tqdm(total=total_samples, unit="samp", desc="Encoding", disable=not verbose)

    while True:
        data = inFile.ReadDataBlock(codingParams)
        if not data: break
        outFile.WriteDataBlock(data, codingParams)
        n = len(data[0])
        processed_samples += n
        pbar.update(n)

    pbar.close()
    # end loop over reading/writing the blocks

    # close the files
    inFile.Close(codingParams)
    outFile.Close(codingParams)

def Decode(inFilename,
           outFilename=None,
           verbose: bool = False
    ):
    if outFilename is None:
        outFilename = f"{Path(inFilename).stem}.wav"

    if verbose:
        print(f"\nDecoding {inFilename} -> {outFilename}")

    inFile = RMCFile(inFilename)
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
        if not data: break
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
    mode.add_argument("-c", "--compress", nargs="+", metavar="FILE",
                      help="Encode: input.wav [output.rmc]")
    mode.add_argument("-d", "--decompress", nargs="+", metavar="FILE",
                      help="Decode: input.rmc [output.wav]")

    parser.add_argument("-b", "--bitrate", type=int, default=128,
                        help="Bitrate in kbps (encode only, default: 128)")
    parser.add_argument("-t", "--tempo", type=int, default=120,
                        help="Tempo in BPM (encode only, default: 120)")

    args = parser.parse_args()

    print_flavor()

    if args.compress:
        inFile = args.compress[0]
        outFile = args.compress[1] if len(args.compress) > 1 else None
        Encode(inFile, codedFilename=outFile, kbps=args.bitrate,
               tempo=args.tempo, verbose=args.verbose)
    else:
        inFile = args.decompress[0]
        outFile = args.decompress[1] if len(args.decompress) > 1 else None
        Decode(inFile, outFilename=outFile, verbose=args.verbose)