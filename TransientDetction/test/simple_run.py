import numpy as np
import librosa
import pycwt.helpers
pycwt.helpers.fft_kwargs = lambda signal: {'n': int(2 ** np.ceil(np.log2(len(signal))))}
import pycwt
from MTE_functions import *

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
        return sorted([int(b) for b in blocks_cwt])  # Convert to int
    if len(blocks_cwt) == 0:
        return sorted([int(b) for b in blocks_time])  # Convert to int
    
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
    
    return sorted([int(b) for b in combined])  # Convert to int

def detectTransients(audioPath, sr=22050, duration=None, blockSize=1024,
                    cwt_onset_threshold=0.3,
                    time_threshold_factor=0.25):
    """
    Simple transient detection - returns combined and separate results.
    
    Returns:
    --------
    dict with:
        'combined': {
            'num_transients': int,
            'blocks': list,
            'times': list
        },
        'cwt': {
            'num_transients': int,
            'blocks': list,
            'times': list
        },
        'time': {
            'num_transients': int,
            'blocks': list,
            'times': list
        }
    """
    # Load audio
    audioData, sr = librosa.load(audioPath, mono=False, sr=sr, duration=duration)
    if audioData.ndim == 1:
        audioData = np.vstack([audioData, audioData])
    
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
        min_distance_ms=50,
        freq_weight_power=1.5
    )
    
    peaks = cwt_results['peaks']
    transientBlocks_cwt = sorted(list(set([p // blockSize for p in peaks])))
    
    # COMBINE
    combined_blocks = combineTimeAndFreqOnsets(
        transientBlocks_time,
        transientBlocks_cwt,
        tolerance_blocks=1
    )
    
    # Format results
    return {
        'combined': {
            'num_transients': len(combined_blocks),
            'blocks': [int(b) for b in combined_blocks],  # Convert to int
            'times': [float(block * blockSize / sr) for block in combined_blocks]  # Convert to float
        },
        'cwt': {
            'num_transients': len(transientBlocks_cwt),
            'blocks': [int(b) for b in transientBlocks_cwt],  # Convert to int
            'times': [float(block * blockSize / sr) for block in transientBlocks_cwt]
        },
        'time': {
            'num_transients': len(transientBlocks_time),
            'blocks': [int(b) for b in transientBlocks_time],  # Convert to int
            'times': [float(block * blockSize / sr) for block in transientBlocks_time]
        },
        'sr': sr,
        'blockSize': blockSize
    }

def batchDetect(audio_files, sr=22050, duration=None):
    """
    Process multiple files and return simple results.
    
    Returns:
    --------
    dict : {filename: detection_results}
    """
    results = {}
    
    for i, audio_path in enumerate(audio_files):
        print(f"[{i+1}/{len(audio_files)}] Processing: {audio_path}")
        
        try:
            result = detectTransients(audio_path, sr=sr, duration=duration)
            results[audio_path] = result
            
            # Print summary
            print(f"  Combined: {result['combined']['num_transients']} transients")
            print(f"  CWT: {result['cwt']['num_transients']}, Time: {result['time']['num_transients']}")
            print(f"  Blocks: {result['combined']['blocks']}\n")
            
        except Exception as e:
            print(f"  ERROR: {e}\n")
            results[audio_path] = {'error': str(e)}
    
    return results

if __name__ == "__main__":
    
    
    audio_files = [
        'Samples/CHIME Kick 2017.wav',
        'Samples/Piano_Hard_C4.wav',
        'Samples/castanets_192kbps.wav',
    ]
    
    batch_results = batchDetect(audio_files, sr=22050)
    
