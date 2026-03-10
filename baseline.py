from prepare_materials import pac, rmc, pacb
import argparse
import glob
from pathlib import Path
from DoEncodeDecode import EncodeDecode

# Default to 120 for untimed pieces
TEMPOS = {
    'inputs/harpsichord.wav': 120,
    'inputs/Brooklyn.wav': 97,
    'inputs/castanets.wav': 120,
    'inputs/spgm.wav': 120,
    'inputs/Van_124.wav': 124,
    'inputs/glockenspiel.wav': 120
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--codec", choices=["pac", "rmc", "pacb"], default="pac",
                        help="Codec to use: 'pac' (default), 'pacb' (pac w/ block switching), or 'rmc' (entropy-coded)")
    args = parser.parse_args()
    codecs = {
        'pac': pac,
        'pacb': pacb,
        'rmc': lambda inFilename, outFilename, rate_kb:  EncodeDecode(inFilename, outFilename, codedFilename=f"coded/{Path(outFilename).stem}.rmc", targetBitsPerSample=rate_kb * 1000 / 44100, tempo=TEMPOS[inFilename])
    }
    encode = codecs[args.codec]

    reference_files = glob.glob("inputs/*.wav")
    for i, file in enumerate(reference_files):
        reference_name = Path(file).stem
        print(f"[{i + 1}/{len(reference_files)}] Running for {file} [{args.codec.upper()}]")
        suffix = f"_{args.codec}" if args.codec != "pac" else ""
        print(f"{args.codec.upper()} 128 kb/s/ch")
        encode(file, f"outputs/{reference_name}{suffix}_128kbps.wav", rate_kb=128)
        print(f"{args.codec.upper()} 96 kb/s/ch")
        encode(file, f"outputs/{reference_name}{suffix}_96kbps.wav", rate_kb=96)
        print("Done\n")