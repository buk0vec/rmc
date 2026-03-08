import numpy as np
import scipy

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
    """Combine time-domain and frequency-domain onset detections"""
    if len(blocks_time) == 0:
        return sorted([int(b) for b in blocks_cwt])
    if len(blocks_cwt) == 0:
        return sorted([int(b) for b in blocks_time])
    
    blocks_time = np.array(blocks_time)
    blocks_cwt = np.array(blocks_cwt)
    combined = set(blocks_cwt)
    
    for time_block in blocks_time:
        is_covered = False
        for cwt_block in blocks_cwt:
            if abs(time_block - cwt_block) <= tolerance_blocks:
                is_covered = True
                break
        if not is_covered:
            combined.add(time_block)
    
    return sorted([int(b) for b in combined])

def detectTransients(audioPath, sr=22050, duration=None, 
                    use_auto_params=True,
                    blockSize=1024,
                    cwt_onset_threshold=0.3, 
                    time_threshold_factor=0.25):
    """
    Detect transients and return combined block array.
    
    Parameters:
    -----------
    use_auto_params : bool
        If True, automatically analyze audio and use adaptive parameters
    
    Returns:
    --------
    numpy array : Combined transient block indices
    """
    # Load audio
    audioData, sr = scipy.io.wavfile.read(audioPath)
    if audioData.dtype == 'int16':
        # For 16-bit audio, the max value is 2**15 - 1 (or 2**15 for scaling range)
        max_val = float(2**15)
    elif audioData.dtype == 'int32':
        # For 32-bit audio, the max value is 2**31 - 1
        max_val = float(2**31)
    elif audioData.dtype == 'float32' or audioData.dtype == 'float64':
        # If the file is already float PCM, no scaling is typically needed
        # (values are usually within -1.0 to 1.0, though can exceed in file)
        max_val = 1.0
    else:
        # Handle other cases like 8-bit, 24-bit, etc.
        print(f"Unsupported data type for scaling: {audioData.dtype}")
        max_val = 1.0 # Default to 1.0 if unsure or not scaling

    # 3. Convert and scale the data to a float array in the range [-1.0, 1.0]
    # Use float64 for better precision during calculations
    audioData = audioData.astype(np.float64) / max_val

    if audioData.ndim == 1:
        audioData = np.vstack([audioData, audioData])
    
    # AUTO-ANALYSIS (if enabled)
    if use_auto_params:
        analysis = analyzeAudioCharacteristics(audioData, sr)
        params = analysis["params"]
        blockSize = params["blockSize"]
        cwt_onset_threshold = params["cwt_onset_threshold"]
        time_threshold_factor = params["time_threshold_factor"]
        cwt_min_distance_ms = params["cwt_min_distance_ms"]
        cwt_freq_weight = params["cwt_freq_weight"]
    else:
        cwt_min_distance_ms = 50
        cwt_freq_weight = 1.5
    
    # TIME-DOMAIN DETECTION
    fast_envelope = envelopeFollower(audioData, sr, 0.5, 5.0, mode='peak')
    slow_envelope = envelopeFollower(audioData, sr, 10.0, 50.0, mode='peak')
    envDiff_time = fast_envelope - slow_envelope
    
    envDiff_time_mono = np.maximum(envDiff_time[0], envDiff_time[1])
    threshold_time = np.max(envDiff_time_mono) * time_threshold_factor
    
    transient_time = extractTransient(audioData, envDiff_time, threshold=threshold_time)
    transient_time_mono = np.maximum(np.abs(transient_time[0]), np.abs(transient_time[1]))
    
    numBlocks_time = int(np.ceil(len(transient_time_mono) / blockSize))
    transientBlocks_time_multi = [i for i in range(numBlocks_time) 
                                  if np.any(transient_time_mono[i*blockSize:min((i+1)*blockSize, len(transient_time_mono))] > 0.001)]
    transientBlocks_time = groupConsecutiveBlocks(transientBlocks_time_multi)
    
    # CWT DETECTION
    wavelet = pycwt.Morlet(6)
    dt = 1 / sr
    dj = 1/8
    f_min = 50
    f_max = sr/2
    s0 = 2 * dt
    J = int(np.ceil(np.log2(f_max / f_min) / dj))
    
    coeff_l, _, _, _, _, _ = pycwt.cwt(audioData[0], dt, dj, s0, J, wavelet)
    coeff_r, _, _, _, _, _ = pycwt.cwt(audioData[1], dt, dj, s0, J, wavelet)
    
    cwt_results = CWT_detect_transients_onset(
        coeff_l, coeff_r, sr,
        onset_threshold_factor=cwt_onset_threshold,
        min_distance_ms=cwt_min_distance_ms,
        freq_weight_power=cwt_freq_weight
    )
    
    peaks = cwt_results['peaks']
    transientBlocks_cwt = sorted(list(set([p // blockSize for p in peaks])))
    
    # COMBINE
    combined_blocks = combineTimeAndFreqOnsets(transientBlocks_time, transientBlocks_cwt)
    
    return np.array(combined_blocks)


if __name__ == "__main__":
    
    # Multiple files with auto params
    print("\n" + "="*70)
    audio_files = [
        'Van_124.wav',
    ]
    
    for file in audio_files:
        blocks = detectTransients(file, use_auto_params=True)
        print(f"\n{file}")
        print(f"  Blocks: {blocks}")
        print(f"  Count: {len(blocks)}")
