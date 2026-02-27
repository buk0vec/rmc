from prepare_materials import pac
import glob
from pathlib import Path

if __name__ == "__main__":
    reference_files = glob.glob("inputs/*.wav")
    for i, file in enumerate(reference_files):
        reference_name = Path(file).stem
        print(f"[{i + 1}/{len(reference_files)}] Running for {file}")
        print("PAC 128 kb/s/ch")
        pac(file, f"outputs/{reference_name}_128kbps.wav", rate_kb=128)
        print("PAC 192 kb/s/ch")
        pac(file, f"outputs/{reference_name}_192kpbs.wav", rate_kb=192)
        print("Done\n")