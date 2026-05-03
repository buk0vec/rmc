"""
test_spe.py

Visualise SPE transient detection on the WAV files in ./inputs/.
For each file, produces a 4-panel figure:
  - Panel 1: waveform with detected transient positions marked
  - Panels 2-4: per-layer peak ratio over time with the detection threshold

Run:
    python test_spe.py
"""

import os
import numpy as np
import scipy.io.wavfile
import scipy.signal
import matplotlib
import matplotlib.pyplot as plt

from spe import (
    detectTransientsSPE,
    spe_block_details,
    _design_highpass,
    _run_spe_pass,
    THRESHOLDS,
    ZERO_THRESHOLD,
    HPF_CUTOFF,
    HPF_CUTOFF2,
    HPF_CUTOFF3,
)

INPUT_DIR = os.path.join(os.path.dirname(__file__), 'inputs')
WAV_FILES = ['castanets.wav', 'glockenspiel.wav', 'harpsichord.wav', 'Van_124.wav',
             'oboe.wav', 'violin2.wav',]


# ---------------------------------------------------------------------------
# Internal helper: re-run SPE while collecting per-block diagnostic data
# ---------------------------------------------------------------------------

def _run_with_details(audioPath: str, nMDCTLines: int = 1024, cutoff: float = HPF_CUTOFF):
    sr, raw = scipy.io.wavfile.read(audioPath)
    if raw.dtype == np.int16:
        audio = raw.astype(np.float64) / 32768.0
    elif raw.dtype == np.int32:
        audio = raw.astype(np.float64) / 2147483648.0
    else:
        audio = raw.astype(np.float64)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # Primary band: collect per-block ratios for plotting
    sos = _design_highpass(sr, cutoff)
    zi = scipy.signal.sosfilt_zi(sos) * audio[0]

    N = nMDCTLines * 2
    half = nMDCTLines
    n_samples = len(audio)
    n_blocks = int(np.ceil(n_samples / half))

    prev_hp = np.zeros(half, dtype=np.float64)
    primary_flagged = []
    all_ratios = []       # shape (n_blocks, 4) — from primary band only
    all_layer_flags = []  # shape (n_blocks, 4)

    for block_idx in range(n_blocks):
        start = block_idx * half
        end = min(start + half, n_samples)
        chunk = audio[start:end]
        if len(chunk) < half:
            chunk = np.pad(chunk, (0, half - len(chunk)))

        curr_hp, zi = scipy.signal.sosfilt(sos, chunk, zi=zi)
        buf = np.concatenate([prev_hp, curr_hp])

        flagged, ratios, layer_flags = spe_block_details(buf, N)
        all_ratios.append(ratios)
        all_layer_flags.append(layer_flags)
        primary_flagged.append(flagged)

        prev_hp = curr_hp

    # Tri-band detection: union of primary + secondary + tertiary (matches detectTransientsSPE)
    primary_detected = {i for i, f in enumerate(primary_flagged) if f}
    secondary_detected = _run_spe_pass(audio, sr, HPF_CUTOFF2, nMDCTLines,
                                       THRESHOLDS, ZERO_THRESHOLD)
    tertiary_detected = _run_spe_pass(audio, sr, HPF_CUTOFF3, nMDCTLines,
                                      THRESHOLDS, ZERO_THRESHOLD)
    combined = sorted(primary_detected | secondary_detected | tertiary_detected)

    # Aggressive boundary dedup (matches detectTransientsSPE): in any pair
    # (N, N+1) where the gap is exactly 1, drop N+1.
    filtered = []
    skip_next = False
    for i, b in enumerate(combined):
        if skip_next:
            skip_next = False
            continue
        filtered.append(b)
        if i + 1 < len(combined) and combined[i + 1] == b + 1:
            skip_next = True
    transient_blocks = filtered

    return (sr, audio, n_blocks, transient_blocks,
            np.array(all_ratios), np.array(all_layer_flags))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_spe(audioPath: str, nMDCTLines: int = 1024, save: bool = True):
    name = os.path.splitext(os.path.basename(audioPath))[0]
    print(f"\nProcessing: {name}")

    (sr, audio, n_blocks, transient_blocks,
     ratios, layer_flags) = _run_with_details(audioPath, nMDCTLines)

    half = nMDCTLines
    t_audio = np.arange(len(audio)) / sr
    t_blocks = np.arange(n_blocks) * half / sr   # block centre times

    # Detection threshold in ratio space: curr*T > prev  ↔  ratio > 1/T
    thresh_ratios = [1.0 / T for T in THRESHOLDS]   # [2.5, 2.5, 14.3]

    sub_sizes = [nMDCTLines, nMDCTLines // 2, nMDCTLines // 4, nMDCTLines // 8]
    layer_labels = [
        f'Layer 1  sub-block {sub_sizes[0]} samples  (threshold ×{thresh_ratios[0]:.1f})',
        f'Layer 2  sub-block {sub_sizes[1]} samples  (threshold ×{thresh_ratios[1]:.1f})',
        f'Layer 3  sub-block {sub_sizes[2]} samples  (threshold ×{thresh_ratios[2]:.1f})',
        f'Layer 4  sub-block {sub_sizes[3]} samples  (threshold ×{thresh_ratios[3]:.1f})',
    ]
    layer_colors = ['#d95f02', '#1b9e77', '#7570b3', '#e7298a']

    fig, axes = plt.subplots(5, 1, figsize=(14, 14))
    fig.suptitle(
        f'SPE Transient Detection — {name}\n'
        f'({len(transient_blocks)} transient block(s) detected, '
        f'block size = {nMDCTLines} samples)',
        fontsize=12, fontweight='bold'
    )

    # ------------------------------------------------------------------
    # Panel 0: waveform
    # ------------------------------------------------------------------
    ax = axes[0]
    ax.plot(t_audio, audio, color='steelblue', linewidth=0.35, alpha=0.85, rasterized=True)
    for b in transient_blocks:
        ax.axvline(b * half / sr, color='red', linewidth=1.0, alpha=0.75, zorder=3)
    ax.set_ylabel('Amplitude', fontsize=9)
    ax.set_title('Waveform  (red = detected transient blocks)', fontsize=9)
    ax.set_xlim(0, len(audio) / sr)
    ax.tick_params(labelbottom=False)

    # ------------------------------------------------------------------
    # Panels 1-4: per-layer peak ratio
    # ------------------------------------------------------------------
    for j in range(4):
        ax = axes[j + 1]
        ax.plot(t_blocks, ratios[:, j],
                color=layer_colors[j], linewidth=0.9, alpha=0.9, label='curr/prev peak ratio')
        ax.axhline(thresh_ratios[j], color='red', linewidth=1.2, linestyle='--',
                   label=f'Threshold (1/T = {thresh_ratios[j]:.1f}×)')
        for b in transient_blocks:
            ax.axvline(b * half / sr, color='red', linewidth=0.7, alpha=0.35, zorder=2)
        ax.set_ylabel('Peak ratio', fontsize=9)
        ax.set_title(layer_labels[j], fontsize=9)
        ax.legend(loc='upper right', fontsize=7.5)
        ax.set_xlim(0, len(audio) / sr)
        if j < 3:
            ax.tick_params(labelbottom=False)

    axes[-1].set_xlabel('Time (s)', fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save:
        out_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'output')
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, f'test_spe_{name}.png')
        fig.savefig(out, dpi=150)
        print(f"  Saved: {out}")

    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print('SPE Transient Detection Test')
    print('=' * 60)

    found_any = False
    for fname in WAV_FILES:
        fpath = os.path.join(INPUT_DIR, fname)
        if not os.path.exists(fpath):
            print(f'  [skip] {fpath} not found')
            continue
        found_any = True

        # Quick detection summary
        blocks = detectTransientsSPE(fpath, verbose=True)
        print(f'  → {len(blocks)} transient block(s): {blocks.tolist()}')

        # Full plot
        plot_spe(fpath)

    if not found_any:
        print(f'No WAV files found in {INPUT_DIR}')
        print('Place castanets.wav / glockenspiel.wav / harpsichord.wav there and re-run.')
