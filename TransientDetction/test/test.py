import numpy as np
import librosa
import matplotlib.pyplot as plt

# Fix pycwt compatibility
import pycwt.helpers
pycwt.helpers.fft_kwargs = lambda signal: {'n': int(2 ** np.ceil(np.log2(len(signal))))}
if not hasattr(np, 'int'):
    np.int = int
    np.float = float
    np.bool = bool
    np.complex = complex
import pycwt
from TD import *

def detectTransients_CWT_auto(audioPath, sr=22050, duration=None, 
                              override_params=None, verbose=True):
    """
    Automatic transient detection with adaptive parameters.
    
    Parameters:
    -----------
    audioPath : str
        Path to audio file
    sr : int
        Sample rate
    duration : float or None
        Duration to load (None = full file)
    override_params : dict or None
        Override specific auto-detected parameters
    verbose : bool
        Print analysis information
    
    Returns:
    --------
    dict : Detection results
    """
    # Load audio for analysis
    audioData, sr = librosa.load(audioPath, mono=False, sr=sr, duration=duration)
    if audioData.ndim == 1:
        audioData = np.vstack([audioData, audioData])
    
    # Analyze and get recommended parameters
    analysis = analyzeAudioCharacteristics(audioData, sr)
    params = analysis["params"]
    
    # Override with user-specified parameters if provided
    if override_params:
        params.update(override_params)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"AUTO-ANALYSIS: {audioPath}")
        print(f"{'='*70}")
        print(f"Detected type: {analysis['audio_type']}")
        print(f"\nAudio characteristics:")
        for key, val in analysis['characteristics'].items():
            if isinstance(val, float):
                print(f"  {key}: {val:.2f}")
            else:
                print(f"  {key}: {val}")
        print(f"\nRecommended parameters:")
        for key, val in params.items():
            if isinstance(val, float):
                print(f"  {key}: {val:.2f}")
            else:
                print(f"  {key}: {val}")
        print(f"{'='*70}\n")
    
    # Run detection with auto parameters
    results = detectTransients_CWT(
        audioPath=audioPath,
        sr=sr,
        duration=duration,
        blockSize=params["blockSize"],
        cwt_onset_threshold=params["cwt_onset_threshold"],
        cwt_min_distance_ms=params["cwt_min_distance_ms"],
        cwt_freq_weight=params["cwt_freq_weight"],
        time_threshold_factor=params["time_threshold_factor"]
    )
    
    # Add analysis info to results
    results["auto_analysis"] = analysis
    results["used_params"] = params
    
    return results


def batchProcessAudio(audio_paths, sr=22050, duration=None, output_dir='batch_results'):
    """
    Process multiple audio files with automatic parameter detection.
    
    Parameters:
    -----------
    audio_paths : list of str
        Paths to audio files
    sr : int
        Sample rate
    duration : float or None
        Duration to load per file
    output_dir : str
        Directory to save results
    
    Returns:
    --------
    dict : Results for each file
    """
    import os
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = {}
    
    print(f"\n{'='*70}")
    print(f"BATCH PROCESSING: {len(audio_paths)} files")
    print(f"{'='*70}\n")
    
    for i, audio_path in enumerate(audio_paths):
        print(f"\n[{i+1}/{len(audio_paths)}] Processing: {audio_path}")
        
        try:
            # Detect with auto parameters
            results = detectTransients_CWT_auto(
                audio_path, 
                sr=sr, 
                duration=duration,
                verbose=True
            )
            
            # Save plot
            filename = os.path.basename(audio_path).replace('.wav', '')
            plot_path = os.path.join(output_dir, f'{filename}_detection.png')
            plotTransientDetection(results, plot_path)
            
            all_results[audio_path] = {
                "num_cwt_onsets": len(results['blocks_cwt_single']),
                "num_time_onsets": len(results['blocks_time_single']),
                "audio_type": results['auto_analysis']['audio_type'],
                "parameters": results['used_params'],
                "onset_times": results['cwt_peak_times']
            }
            
        except Exception as e:
            print(f"ERROR processing {audio_path}: {e}")
            all_results[audio_path] = {"error": str(e)}
    
    # Save summary
    summary_path = os.path.join(output_dir, 'batch_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("BATCH PROCESSING SUMMARY\n")
        f.write("="*70 + "\n\n")
        for path, info in all_results.items():
            f.write(f"{path}\n")
            if "error" in info:
                f.write(f"  ERROR: {info['error']}\n")
            else:
                f.write(f"  Type: {info['audio_type']}\n")
                f.write(f"  CWT onsets: {info['num_cwt_onsets']}\n")
                f.write(f"  Time onsets: {info['num_time_onsets']}\n")
                f.write(f"  Parameters: {info['parameters']}\n")
            f.write("\n")
    
    print(f"\n{'='*70}")
    print(f"BATCH COMPLETE - Results saved to: {output_dir}")
    print(f"{'='*70}\n")
    
    return all_results

def groupConsecutiveBlocks(blockList):
    """Returns only the first block of each consecutive group"""
    if len(blockList) == 0:
        return []
    
    groups = [[blockList[0]]]
    for block in blockList[1:]:
        if block - groups[-1][-1] == 1:
            groups[-1].append(block)
        else:
            groups.append([block])
    
    return [group[0] for group in groups]

def combineTimeAndFreqOnsets(blocks_time, blocks_cwt, tolerance_blocks=1):
    """
    Combine time-domain and frequency-domain onset detections.
    
    Strategy: Time-domain is conservative (few false positives).
              Add any time-domain onsets that CWT missed.
    
    Parameters:
    -----------
    blocks_time : list or array
        Block indices from time-domain detection
    blocks_cwt : list or array
        Block indices from CWT detection
    tolerance_blocks : int
        How many blocks apart to consider "same" onset (default=1)
    
    Returns:
    --------
    array : Combined unique onset blocks, sorted
    """
    if len(blocks_time) == 0:
        return np.array(sorted(blocks_cwt))
    if len(blocks_cwt) == 0:
        return np.array(sorted(blocks_time))
    
    blocks_time = np.array(blocks_time)
    blocks_cwt = np.array(blocks_cwt)
    
    # Start with CWT detections
    combined = set(blocks_cwt)
    
    # Check each time-domain onset
    for time_block in blocks_time:
        # Check if this onset is already covered by CWT
        # (within tolerance range)
        is_covered = False
        for cwt_block in blocks_cwt:
            if abs(time_block - cwt_block) <= tolerance_blocks:
                is_covered = True
                break
        
        # If not covered, add it
        if not is_covered:
            combined.add(time_block)
            print(f"  Added missing onset: block {time_block} (from time-domain)")
    
    return np.array(sorted(combined))

def detectTransients_CWT(audioPath, sr=22050, duration=10.0, blockSize=1024,
                        cwt_onset_threshold=0.5,
                        cwt_min_distance_ms=50,
                        cwt_freq_weight=1.5,
                        time_threshold_factor=0.3,
                        combine_detections=True,
                        onset_tolerance_blocks=1):
    """
    Complete transient detection using ONSET-BASED CWT method.
    
    Parameters:
    -----------
    audioPath : str
        Path to audio file
    sr : int
        Sample rate
    duration : float or None
        Duration to load (None = full file)
    blockSize : int
        Block size in samples
    cwt_onset_threshold : float
        Onset detection threshold (0.3-0.7 recommended)
        Lower = more sensitive, Higher = only strong attacks
    cwt_min_distance_ms : float
        Minimum time between detected transients (milliseconds)
    cwt_freq_weight : float
        High-frequency emphasis (1.0-2.5)
        Higher = focus more on high-frequency transients
    time_threshold_factor : float
        Time-domain detection threshold factor
    combine_detections : bool
        If True, combine CWT and time-domain detections (recommended)
    onset_tolerance_blocks : int
        Blocks apart to consider same onset when combining
    """
    
    # Load audio
    audioData, sr = librosa.load(audioPath, mono=False, sr=sr, duration=duration)
    if audioData.ndim == 1:
        audioData = np.vstack([audioData, audioData])
    
    print(f"Audio: {audioPath}")
    print(f"Duration: {audioData.shape[1]/sr:.2f}s, SR: {sr}Hz")
    print(f"Block size: {blockSize} samples ({blockSize/sr*1000:.1f}ms)")
    
    print("\n--- Time-Domain Detection ---")
    
    fast_attack_ms = 0.5
    fast_release_ms = 5.0
    slow_attack_ms = 10.0
    slow_release_ms = 50.0
    
    fast_envelope = envelopeFollower(audioData, sr, fast_attack_ms, fast_release_ms, mode='peak')
    slow_envelope = envelopeFollower(audioData, sr, slow_attack_ms, slow_release_ms, mode='peak')
    envDiff_time = fast_envelope - slow_envelope
    
    envDiff_time_mono = np.maximum(envDiff_time[0], envDiff_time[1])
    threshold_time = np.max(envDiff_time_mono) * time_threshold_factor
    
    print(f"Envelope parameters: attack={fast_attack_ms}/{slow_attack_ms}ms, release={fast_release_ms}/{slow_release_ms}ms")
    print(f"Threshold: {threshold_time:.4f} ({time_threshold_factor*100:.0f}% of max)")
    
    transient_time = extractTransient(audioData, envDiff_time, threshold=threshold_time)
    transient_time_mono = np.maximum(np.abs(transient_time[0]), np.abs(transient_time[1]))
    
    # Time-domain block detection
    numBlocks_time = int(np.ceil(len(transient_time_mono) / blockSize))
    transientBlocks_time_multi = [i for i in range(numBlocks_time) 
                                  if np.any(transient_time_mono[i*blockSize:min((i+1)*blockSize, len(transient_time_mono))] > 0.001)]
    transientBlocks_time_single = groupConsecutiveBlocks(transientBlocks_time_multi)
    
    print(f"Detected: {len(transientBlocks_time_single)} onsets (time-domain)")
    
    print("\n--- CWT Detection (Onset-Based) ---")
    
    # CWT parameters
    wavelet = pycwt.Morlet(6)
    dt = 1 / sr
    dj = 1/8  # 8 scales per octave
    f_min = 50
    f_max = sr/2
    s0 = 2 * dt
    J = int(np.ceil(np.log2(f_max / f_min) / dj))
    
    print(f"Frequency range: {f_min:.0f}-{f_max:.0f}Hz")
    print(f"Scales: {int(1/dj)} per octave, {J} total")
    print(f"Frequency weight power: {cwt_freq_weight}")
    
    print("Computing CWT...")
    coeff_l, scales_l, freq_l, _, _, _ = pycwt.cwt(audioData[0], dt, dj, s0, J, wavelet)
    coeff_r, scales_r, freq_r, _, _, _ = pycwt.cwt(audioData[1], dt, dj, s0, J, wavelet)
    print("✓ CWT complete")

    print("Detecting onset peaks...")
    cwt_results = CWT_detect_transients_onset(
        coeff_l, coeff_r, sr,
        onset_threshold_factor=cwt_onset_threshold,
        min_distance_ms=cwt_min_distance_ms,
        freq_weight_power=cwt_freq_weight
    )
    
    peaks = cwt_results['peaks']
    peak_times = cwt_results['peak_times']
    onset_strength = cwt_results['onset_strength']
    cwt_amplitude = cwt_results['amplitude']
    threshold_cwt = cwt_results['threshold']
    
    print(f"Onset threshold: {threshold_cwt:.2f} ({cwt_onset_threshold*100:.0f}% factor)")
    print(f"Min distance: {cwt_min_distance_ms}ms")
    print(f"Detected: {len(peaks)} transient onsets (CWT)")
    
    # Convert peaks to blocks
    transientBlocks_cwt_single = sorted(list(set([p // blockSize for p in peaks])))
    transientBlocks_cwt_original = transientBlocks_cwt_single.copy()  # Store original before combining
    
    if combine_detections:
        print("\n--- Combining Time + Freq Detections ---")
        print(f"CWT detected: {len(transientBlocks_cwt_single)} onsets")
        print(f"Time detected: {len(transientBlocks_time_single)} onsets")
        
        combined_blocks = combineTimeAndFreqOnsets(
            transientBlocks_time_single,
            transientBlocks_cwt_single,
            tolerance_blocks=onset_tolerance_blocks
        )
        
        print(f"Combined total: {len(combined_blocks)} onsets")
        print(f"Added {len(combined_blocks) - len(transientBlocks_cwt_single)} onsets from time-domain")
        
        # Update the CWT blocks with combined result
        transientBlocks_cwt_single = combined_blocks.tolist()
    
    print("\n" + "="*70)
    print("DETECTION SUMMARY")
    print("="*70)
    
    if combine_detections:
        print(f"\nCombined Detection: {len(transientBlocks_cwt_single)} onsets")
    else:
        print(f"\nCWT Onset Detection: {len(transientBlocks_cwt_single)} onsets")
    
    # Convert blocks to times for display
    combined_times = np.array(transientBlocks_cwt_single) * blockSize / sr
    for i, (block, time) in enumerate(zip(transientBlocks_cwt_single[:15], combined_times[:15])):
        sample = block * blockSize
        print(f"  {i+1}. Block {block}, Time {time:6.3f}s, Sample ~{sample}")
    if len(transientBlocks_cwt_single) > 15:
        print(f"  ... and {len(transientBlocks_cwt_single)-15} more")
    
    print(f"\nTime-Domain Detection: {len(transientBlocks_time_single)} onsets")
    for i, block in enumerate(transientBlocks_time_single[:15]):
        time = block * blockSize / sr
        print(f"  {i+1}. Block {block}, Time {time:6.3f}s")
    if len(transientBlocks_time_single) > 15:
        print(f"  ... and {len(transientBlocks_time_single)-15} more")
    
    print("="*70)
    
    return {
        # Audio data
        'audioData': audioData,
        'sr': sr,
        'blockSize': blockSize,
        
        # CWT results (potentially combined with time-domain)
        'coeff_l': coeff_l,
        'freq_l': freq_l,
        'cwt_amplitude': cwt_amplitude,
        'onset_strength': onset_strength,
        'cwt_peaks': peaks,
        'cwt_peak_times': peak_times,
        'cwt_peak_heights': cwt_results['peak_heights'],
        'threshold_cwt': threshold_cwt,
        'blocks_cwt_single': transientBlocks_cwt_single,  # May include time-domain additions
        'blocks_cwt_original': transientBlocks_cwt_original,  # Store original CWT-only detections
        
        # Time-domain results
        'envDiff_time': envDiff_time,
        'transient_time': transient_time,
        'transient_time_mono': transient_time_mono,
        'threshold_time': threshold_time,
        'blocks_time_multi': transientBlocks_time_multi,
        'blocks_time_single': transientBlocks_time_single,
        
        # Combination info
        'combined_detections': combine_detections,
    }

def plotTransientDetection(results, savePath='transient_detection.png'):
    """Plot all transient detection results"""
    
    audioData = results['audioData']
    sr = results['sr']
    blockSize = results['blockSize']
    
    time_axis = np.arange(audioData.shape[1]) / sr
    time_cwt = np.arange(results['cwt_amplitude'].shape[0]) / sr
    
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Original Audio
    ax1 = plt.subplot(5, 1, 1)
    ax1.plot(time_axis, audioData[0], linewidth=0.5, alpha=0.7)
    ax1.set_title('Original Audio (Time Domain)', fontweight='bold')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, audioData.shape[1]/sr])
    
    # 2. CWT Summed Amplitude with Detected Peaks
    ax2 = plt.subplot(5, 1, 2)
    ax2.plot(time_cwt, results['cwt_amplitude'], linewidth=1, label='CWT Amplitude')
    ax2.axhline(y=results['threshold_cwt'], color='r', linestyle='--', 
                linewidth=1, label=f"Threshold = {results['threshold_cwt']:.2f}")
    
    peak_times = results['cwt_peak_times']
    peak_heights = results['cwt_peak_heights']
    ax2.scatter(peak_times, peak_heights, color='red', s=50, zorder=5, 
                marker='v', label=f'{len(peak_times)} Peaks')
    
    ax2.set_title('CWT Amplitude + Peak Detection', fontweight='bold')
    ax2.set_ylabel('Amplitude')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, audioData.shape[1]/sr])
    
    # 3. Time-Domain Envelope Difference
    ax3 = plt.subplot(5, 1, 3)
    envDiff_mono = np.maximum(results['envDiff_time'][0], results['envDiff_time'][1])
    time_env = np.arange(len(envDiff_mono)) / sr
    ax3.plot(time_env, envDiff_mono, linewidth=1, label='Envelope Difference')
    ax3.axhline(y=results['threshold_time'], color='b', linestyle='--', 
                linewidth=1, label=f"Threshold = {results['threshold_time']:.4f}")
    ax3.set_title('Time-Domain Envelope Difference', fontweight='bold')
    ax3.set_ylabel('Value')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, audioData.shape[1]/sr])
    
    # 4. Extracted Transient
    ax4 = plt.subplot(5, 1, 4)
    ax4.plot(time_axis[:len(results['transient_time'][0])], 
             results['transient_time'][0], 
             linewidth=0.5, alpha=0.7, color='blue')
    ax4.set_title('Extracted Transient (Time-domain)', fontweight='bold')
    ax4.set_ylabel('Amplitude')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([0, audioData.shape[1]/sr])
    
    # 5. Block Detection Comparison
    ax5 = plt.subplot(5, 1, 5)
    numBlocks = max(max(results['blocks_cwt_single']) if results['blocks_cwt_single'] else 0,
                    max(results['blocks_time_single']) if results['blocks_time_single'] else 0) + 1
    
    block_times = np.arange(numBlocks) * blockSize / sr
    
    # Different visualization if combined
    if results.get('combined_detections', False):
        # Show which onsets came from where
        cwt_original_set = set(results.get('blocks_cwt_original', []))
        time_set = set(results['blocks_time_single'])
        combined_set = set(results['blocks_cwt_single'])
        
        # Both methods detected (intersection of original CWT and time)
        both = cwt_original_set & time_set
        # CWT only (original CWT but not time)
        cwt_only = cwt_original_set - time_set
        # Time only (added from time-domain that CWT missed)
        time_only = combined_set - cwt_original_set
        
        block_mask_both = np.zeros(numBlocks)
        block_mask_both[list(both)] = 1
        block_mask_cwt_only = np.zeros(numBlocks)
        block_mask_cwt_only[list(cwt_only)] = 1
        block_mask_time_only = np.zeros(numBlocks)
        block_mask_time_only[list(time_only)] = 1
        block_mask_time_all = np.zeros(numBlocks)
        block_mask_time_all[results['blocks_time_single']] = 1
        
        # Upper bars: Combined CWT result (original + added from time)
        ax5.bar(block_times, block_mask_both, width=blockSize/sr, 
                align='edge', alpha=0.7, color='green', edgecolor='darkgreen',
                label=f'Both methods ({len(both)})')
        ax5.bar(block_times, block_mask_cwt_only, width=blockSize/sr,
                align='edge', alpha=0.7, color='orange', edgecolor='darkorange',
                label=f'CWT only ({len(cwt_only)})')
        ax5.bar(block_times, block_mask_time_only, width=blockSize/sr,
                align='edge', alpha=0.7, color='purple', edgecolor='darkviolet',
                label=f'Added from time ({len(time_only)})')
        
        # Lower bar: All time-domain detections
        ax5.bar(block_times, -block_mask_time_all, width=blockSize/sr,
                align='edge', alpha=0.6, color='blue', edgecolor='darkblue',
                label=f'Time-domain ({len(results["blocks_time_single"])})')
        
        ax5.set_title(f'Combined Detection (Total: {len(results["blocks_cwt_single"])} onsets, '
                     f'Original CWT: {len(cwt_original_set)}, Added: {len(time_only)})', 
                     fontweight='bold')
    else:
        # Original visualization (no combination)
        block_mask_cwt = np.zeros(numBlocks)
        block_mask_cwt[results['blocks_cwt_single']] = 1
        block_mask_time = np.zeros(numBlocks)
        block_mask_time[results['blocks_time_single']] = 1
        
        ax5.bar(block_times, block_mask_cwt, width=blockSize/sr, 
                align='edge', alpha=0.6, color='red', edgecolor='darkred',
                label=f'CWT ({len(results["blocks_cwt_single"])} onsets)')
        ax5.bar(block_times, -block_mask_time, width=blockSize/sr,
                align='edge', alpha=0.6, color='blue', edgecolor='darkblue',
                label=f'Time ({len(results["blocks_time_single"])} onsets)')
        
        ax5.set_title(f'Onset Detection (Block size = {blockSize})', 
                     fontweight='bold')
    
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Combined ↑ | Time ↓')
    ax5.set_ylim([-1.2, 1.2])
    ax5.axhline(y=0, color='k', linewidth=0.5)
    ax5.legend(loc='upper right')
    ax5.grid(True, alpha=0.3, axis='x')
    ax5.set_xlim([0, audioData.shape[1]/sr])
    
    plt.tight_layout()
    plt.savefig(savePath, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved: {savePath}")
    plt.show()

if __name__ == "__main__":
    
    audio_files = [
        'Samples/ringnoord.wav'
    ]
    
    batch_results = batchProcessAudio(audio_files, output_dir='auto_detection_results')
