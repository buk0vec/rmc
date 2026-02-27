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
⠘⣿⡀⠀⢿⡀⠀⢻⣤⠖⢻⡿      - multiple predictive models.
⠀⠘⢷⣄⠈⠙⠦⠸⡇⢀⡾⠃      - block switching and transient detection.
⠀⠀⠀⠙⠛⠶⠤⠶⣿⠉⠀⠀      - entropy coding.
⠀⠀⠀⠀⠀⠀⠀⠀⢹⡇⠀⠀ 
⠀⠀⢀⣴⣾⣿⣆⠀⠈⣧⠀⠀  
⠀⠀⠈⣿⣿⡿⠃⠀⣰⡏⠀⠀man i hope this works
⠀⠀⠀⠈⣙⠓⠒⠚⠉⠀⠀
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