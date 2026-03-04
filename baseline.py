from prepare_materials import pac, rmc, pacb
import argparse
import glob
from pathlib import Path



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--codec", choices=["pac", "rmc", "pacb"], default="pac",
                        help="Codec to use: 'pac' (default), 'pacb' (pac w/ block switching), or 'rmc' (entropy-coded)")
    args = parser.parse_args()
    codecs = {
        'pac': pac,
        'pacb': pacb,
        'rmc': rmc
    }
    encode = codecs[args.codec]

    reference_files = glob.glob("inputs/*.wav")
    for i, file in enumerate(reference_files):
        reference_name = Path(file).stem
        print(f"[{i + 1}/{len(reference_files)}] Running for {file} [{args.codec.upper()}]")
        suffix = f"_{args.codec}" if args.codec != "pac" else ""
        print(f"{args.codec.upper()} 128 kb/s/ch")
        encode(file, f"outputs/{reference_name}{suffix}_128kbps.wav", rate_kb=128)
        print(f"{args.codec.upper()} 192 kb/s/ch")
        encode(file, f"outputs/{reference_name}{suffix}_192kbps.wav", rate_kb=192)
        print("Done\n")