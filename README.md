# RMC (Rhythmic Music Coder)

```
⠀⠀⠀⠀⠀⢀⣼⣿⣆⠀⠀⠀
⠀⠀⠀⠀⠀⣾⡿⠛⢻⡆⠀⠀MUSIC 422 project by Nick, Summer, and Simon
⠀⠀⠀⠀⢰⣿⠀⠀⢸⡇⠀⠀
⠀⠀⠀⠀⠸⡇⠀⢀⣾⠇⠀⠀
⠀⠀⠀⠀⠀⣿⣤⣿⡟⠀⠀⠀a "state-of-the-art" audio coder for 
⠀⠀⠀⠀⣠⣾⣿⣿⠀⠀⠀⠀songs with known tempos.
⠀⠀⣠⣾⣿⡿⣏⠀⠀⠀⠀⠀
⠀⣴⣿⡿⠋⠀⢻⡉⠀⠀⠀⠀
⢰⣿⡟⠀⢀⣴⣿⣿⣿⣿⣦⠀
⢸⡿⠀⠀⣿⠟⠛⣿⠟⠛⣿⣧      features:
⠘⣿⡀⠀⢿⡀⠀⢻⣤⠖⢻⡿      - tempo-correlated rhythmic prediction.
⠀⠘⢷⣄⠈⠙⠦⠸⡇⢀⡾⠃      - block switching and transient detection.
⠀⠀⠀⠙⠛⠶⠤⠶⣿⠉⠀⠀      - entropy coding.
⠀⠀⠀⠀⠀⠀⠀⠀⢹⡇⠀⠀ 
⠀⠀⢀⣴⣾⣿⣆⠀⠈⣧⠀⠀  
⠀⠀⠈⣿⣿⡿⠃⠀⣰⡏⠀⠀man i hope this works
⠀⠀⠀⠈⣙⠓⠒⠚⠉⠀⠀
```

> [!NOTE]
> Does not support numpy < 2, please make sure you have numpy 2.0 or higher installed!!!

## xrmc 

`xrmc` is a CLI for encoding and decoding RMC files. It also provides a Python API for encoding and decoding. If you have `uv` installed, you can run the CLI commands with `uv run`. Otherwise, you must install the dev dependencies as detailed below.

### Encode

```bash
uv run python xrmc.py -c song.wav -b 128 -t 120 -v
```

- `-c` compress (input.wav [output.rmc])
- `-b` bitrate in kbps (default 128)
- `-t` tempo in BPM for rhythmic prediction (default 120)
- `-v` verbose

### Decode

```bash
uv run python xrmc.py -d song.rmc -v
```

### Python API

```python
from xrmc import Encode, Decode

Encode("song.wav", "song.rmc", kbps=128, tempo=120, verbose=True)
Decode("song.rmc", "song_decoded.wav", verbose=True)
```

## Development with uv

Just run `uv sync` to install packages and then `uv run __.py` to run files

## Development with conda

```
conda create -n rmc python=3.13
conda activate rmc
pip install -r requirements.txt
```

Then, run as normal with `python3 __.py`

