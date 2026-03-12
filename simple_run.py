import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import scipy.io.wavfile
import scipy.signal
# import pywt
from TD import envelopeFollower, extractTransient, CWT_detect_transients_onset


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


def _spectral_centroid(audio_mono, sr):
    f, _, Sxx = scipy.signal.spectrogram(audio_mono, fs=sr, nperseg=2048, noverlap=1024)
    power = Sxx.mean(axis=1)
    return float(np.sum(f * power) / (np.sum(power) + 1e-10))


def _zero_crossing_rate(audio_mono):
    signs = np.sign(audio_mono)
    signs[signs == 0] = 1
    return len(np.where(np.diff(signs))[0]) / len(audio_mono)


def _onset_density(audio_mono, sr, hop_len=512, n_fft=2048):
    """Spectral flux onset detection."""
    n_frames = (len(audio_mono) - n_fft) // hop_len
    win = np.hanning(n_fft)
    prev_spec = np.zeros(n_fft // 2 + 1)
    strengths = np.empty(n_frames)
    for i in range(n_frames):
        spec = np.abs(np.fft.rfft(audio_mono[i*hop_len : i*hop_len+n_fft] * win))
        strengths[i] = np.sum(np.maximum(spec - prev_spec, 0))
        prev_spec = spec
    min_dist = max(1, int(sr / hop_len * 0.05))
    peaks, _ = scipy.signal.find_peaks(
        strengths,
        height=np.max(strengths) * 0.15,
        distance=min_dist
    )
    return len(peaks) / (len(audio_mono) / sr)


def _rms_variability(audio_mono, hop=512, frame=2048):
    rms = np.array([
        np.sqrt(np.mean(audio_mono[i:i+frame] ** 2))
        for i in range(0, len(audio_mono) - frame, hop)
    ])
    return float(np.std(rms) / (np.mean(rms) + 1e-6))


def analyzeAudioCharacteristics(audioData, sr):
    if audioData.ndim == 2:
        audio_mono = audioData.mean(axis=0)
    else:
        audio_mono = audioData

    avg_centroid = _spectral_centroid(audio_mono, sr)
    avg_zcr = _zero_crossing_rate(audio_mono)
    onset_density = _onset_density(audio_mono, sr)
    rms_var = _rms_variability(audio_mono)

    abs_audio = np.abs(audio_mono)
    peak_val = np.max(abs_audio)
    peak_idx = int(np.argmax(abs_audio))
    decay_threshold = peak_val * 0.1
    decay_samples = len(abs_audio) - peak_idx
    for i in range(peak_idx, len(abs_audio)):
        if abs_audio[i] < decay_threshold:
            decay_samples = i - peak_idx
            break
    decay_time_ms = decay_samples / sr * 1000

    dynamic_range_db = 20 * np.log10((peak_val + 1e-6) / (np.mean(abs_audio) + 1e-6))

    if rms_var < 0.5 and onset_density > 3 and dynamic_range_db < 20:
        audio_type = "electronic_compressed"
    elif onset_density > 5:
        audio_type = "percussive_rapid"
    elif decay_time_ms < 100:
        audio_type = "percussive_short"
    elif avg_centroid > 2000 and decay_time_ms > 200:
        audio_type = "plucked_sustained"
    elif decay_time_ms > 500:
        audio_type = "sustained"
    else:
        audio_type = "general"

    param_presets = {
        "electronic_compressed": {
            "cwt_onset_threshold": 0.65,
            "cwt_min_distance_ms": 100,
            "cwt_freq_weight": 1.3,
            "time_threshold_factor": 0.45,
            "blockSize": 1024,
        },
        "percussive_rapid": {
            "cwt_onset_threshold": 0.35,
            "cwt_min_distance_ms": 15,
            "cwt_freq_weight": 1.5,
            "time_threshold_factor": 0.28,
            "blockSize": 512,
        },
        "percussive_short": {
            "cwt_onset_threshold": 0.3,
            "cwt_min_distance_ms": 40,
            "cwt_freq_weight": 1.5,
            "time_threshold_factor": 0.33,
            "blockSize": 1024,
        },
        "plucked_sustained": {
            "cwt_onset_threshold": 0.45,
            "cwt_min_distance_ms": 150,
            "cwt_freq_weight": 2.5,
            "time_threshold_factor": 0.38,
            "blockSize": 2048,
        },
        "sustained": {
            "cwt_onset_threshold": 0.5,
            "cwt_min_distance_ms": 100,
            "cwt_freq_weight": 2.0,
            "time_threshold_factor": 0.33,
            "blockSize": 2048,
        },
        "general": {
            "cwt_onset_threshold": 0.45,
            "cwt_min_distance_ms": 30,
            "cwt_freq_weight": 1.8,
            "time_threshold_factor": 0.33,
            "blockSize": 1024,
        },
    }

    params = param_presets[audio_type].copy()

    if onset_density > 10:
        params["cwt_min_distance_ms"] = max(10, params["cwt_min_distance_ms"] * 0.5)
    elif onset_density < 1:
        params["cwt_min_distance_ms"] = min(200, params["cwt_min_distance_ms"] * 1.5)

    if avg_centroid > 3000:
        params["cwt_freq_weight"] = min(3.0, params["cwt_freq_weight"] * 1.2)
    elif avg_centroid < 1000:
        params["cwt_freq_weight"] = max(1.0, params["cwt_freq_weight"] * 0.8)

    return {
        "params": params,
        "audio_type": audio_type,
        "characteristics": {
            "spectral_centroid_hz": avg_centroid,
            "zero_crossing_rate": avg_zcr,
            "onset_density_per_sec": onset_density,
            "decay_time_ms": decay_time_ms,
            "duration_sec": len(audio_mono) / sr,
            "rms_variability": rms_var,
            "dynamic_range_db": dynamic_range_db,
        },
    }


def detectTransients(audioPath, sr=22050, duration=None,
                     use_auto_params=True,
                     blockSize=1024,
                     cwt_onset_threshold=0.3,
                     time_threshold_factor=0.25,
                     verbose=True):
    """
    Detect transients and return combined block array.
    """
    sr, audioData = scipy.io.wavfile.read(audioPath)

    if audioData.dtype == np.int16:
        audioData = audioData.astype(np.float64) / float(2**15)
    elif audioData.dtype == np.int32:
        audioData = audioData.astype(np.float64) / float(2**31)
    elif audioData.dtype in (np.float32, np.float64):
        audioData = audioData.astype(np.float64)
    else:
        audioData = audioData.astype(np.float64)

    if audioData.ndim == 1:
        audioData = np.vstack([audioData, audioData])
    elif audioData.ndim == 2 and audioData.shape[1] == 2:
        # (nSamples, 2) -> (2, nSamples)
        audioData = audioData.T

    if use_auto_params:
        analysis = analyzeAudioCharacteristics(audioData, sr)
        params = analysis["params"]
        blockSize = params["blockSize"]
        cwt_onset_threshold = params["cwt_onset_threshold"]
        time_threshold_factor = params["time_threshold_factor"]
        cwt_min_distance_ms = params["cwt_min_distance_ms"]
        cwt_freq_weight = params["cwt_freq_weight"]
        # print(f"  audio_type: {analysis['audio_type']}")
        # print(f"  characteristics: {analysis['characteristics']}")
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
    transientBlocks_time_multi = [
        i for i in range(numBlocks_time)
        if np.any(transient_time_mono[i*blockSize : min((i+1)*blockSize, len(transient_time_mono))] > 0.001)
    ]
    transientBlocks_time = groupConsecutiveBlocks(transientBlocks_time_multi)

    # CWT DETECTION via pywt
    # dt = 1.0 / sr
    # wavelet = 'cmor1.5-1.0'
    # fc = pywt.central_frequency(wavelet)
    # f_min, f_max = 50.0, sr / 2.0
    # n_scales = 64
    # freqs = np.geomspace(f_min, f_max, n_scales)
    # scales = fc / (freqs * dt)

    # coeff_l, _ = pywt.cwt(audioData[0], scales, wavelet, sampling_period=dt)
    # coeff_r, _ = pywt.cwt(audioData[1], scales, wavelet, sampling_period=dt)

    # cwt_results = CWT_detect_transients_onset(
    #     coeff_l, coeff_r, sr,
    #     onset_threshold_factor=cwt_onset_threshold,
    #     min_distance_ms=cwt_min_distance_ms,
    #     freq_weight_power=cwt_freq_weight,
    # )

    # peaks = cwt_results['peaks']
    # transientBlocks_cwt = sorted(list(set([p // blockSize for p in peaks])))

    # # COMBINE
    # combined_blocks = combineTimeAndFreqOnsets(transientBlocks_time, transientBlocks_cwt)
    if verbose:
        print(f"  Time-domain blocks ({len(transientBlocks_time)}): {transientBlocks_time}")
    # print(f"  CWT blocks         ({len(transientBlocks_cwt)}): {transientBlocks_cwt}")

    return np.array(transientBlocks_time)


if __name__ == "__main__":
    print("\n" + "="*70)
    audio_files = [
        '/Users/summerkrinsky/Documents/GitHub/rmc/Van_124.wav',
    ]

    for file in audio_files:
        print(f"\n{file}")
        blocks = detectTransients(file, use_auto_params=True)
        print(f"  Blocks: {blocks}")
        print(f"  Count: {len(blocks)}")
