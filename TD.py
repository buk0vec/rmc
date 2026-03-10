import numpy as np

def getEnvelopFollowerCoefficient(time_ms, sampleRate):
    if time_ms <= 0:
        return 1
    Ts = 1/sampleRate
    timeInSeconds = time_ms / 1000
    return 1 - np.exp(-Ts/timeInSeconds)

def envelopFollowerSingleChannel(audioData, sampleRate, attack_ms, release_ms, mode = "peak"):
    #make all data be positive, define as x[n]
    if mode == 'peak':
        x = np.abs(audioData)
    elif mode == 'rms':
        x = audioData ** 2
    else:
        raise ValueError("mode must be peak or rms")
    #calculate attack and release coefficient with time input
    attackCoefficient = getEnvelopFollowerCoefficient(attack_ms, sampleRate)
    releaseCoefficient = getEnvelopFollowerCoefficient(release_ms, sampleRate)
    #initialize evelop follower as y[n]
    y = np.zeros(len(x))
    y[0] = x[0]
    #apply the evelop follower algorithm
    for n in range (1, len(x)):
        if x[n] > y[n-1]: #when amp is rising
            y[n] = attackCoefficient * x[n] + (1-attackCoefficient) * y[n-1]
        elif x[n] < y[n-1]: #when amp is falling
            y[n] = releaseCoefficient *  x[n] + (1-releaseCoefficient) * y[n-1]
        else:
            y[n] = y[n-1]
    if mode == 'rms':
        y = np.sqrt(y)
    return y

#general function that handels both single channel and multi-channel
def envelopeFollower(audioData, sampleRate, attack_ms, release_ms, mode='peak'):
    if audioData.ndim > 1:
        y = np.zeros_like(audioData)
        for i in range(audioData.shape[0]):
            y_channel = envelopFollowerSingleChannel(audioData[i], sampleRate, attack_ms, release_ms, mode)
            y[i] = y_channel
        return y
    elif audioData.ndim == 1:
        return envelopFollowerSingleChannel(audioData, sampleRate, attack_ms, release_ms, mode)
    else:
        raise ValueError("invalid number of audio channels")

def extractMonoTransient(audioData, envelopeDifference, threshold = 0.05):
    transient = np.zeros_like(audioData)
    for n in range(len(audioData)):
        if envelopeDifference[n] >= threshold:
            transient[n] = audioData[n]
        else:
            transient[n] = 0
    return transient

def adjustLength(input, target):
    if input.ndim == 1:
        #when input < target, zero padding at the end
        if len(input) < target:
            return np.pad(input, (0, target - len(input)), 'constant')
        elif len(input) > target:
            #when input > target, discard the exceeded part
            return input[:target]
        return input

    elif input.ndim == 2:
        #len(input) = input.shape[1]
        if input.shape[1] < target:
            pad_width = ((0, 0), (0, target - input.shape[1]))
            return np.pad(input, pad_width, 'constant')
        elif input.shape[1] > target:
            return input[:, :target]
        return input

def extractTransient(audioData, envelopeDifference, threshold = 0.05):

    #ad and ed shouldn't exceed 2 channels
    if audioData.shape[0] > 2 and audioData.ndim > 2:
        print(audioData.shape)
        raise ValueError("audioData must be mono or stereo")
    if audioData.shape[0] > 2 and audioData.ndim > 2:
        print(audioData.shape)
        raise ValueError("envelopeDifference must be mono or stereo")

    #case 1: mono ad
    if audioData.ndim == 1:
        #adjust ed's length if needed
        ed_processed = adjustLength(envelopeDifference, len(audioData))

        #case 1a: mono ad + mono ed
        if ed_processed.ndim == 1:
            transient = extractMonoTransient(audioData, ed_processed, threshold)

        #case 1b: mono ad + stereo ed
        elif ed_processed.ndim == 2 and ed_processed.shape[0] == 2:
            transient = extractMonoTransient(audioData, np.maximum(ed_processed[0], ed_processed[1]), threshold)
        else:
            raise ValueError("Invalid envelopeDifference shape for mono audioData")
        return transient

    #case 2: stereo ad
    elif audioData.ndim == 2 and audioData.shape[0] == 2:
        transient = np.zeros_like(audioData)

        #case 2a: stereo ad + mono ed
        if envelopeDifference.ndim == 1:
            ed_processed = adjustLength(envelopeDifference, audioData.shape[1])
            for channel in range(2):
                transient[channel] = extractMonoTransient(audioData[channel], ed_processed, threshold)

        #case 2b: stereo ad + stereo ed
        elif envelopeDifference.ndim == 2 and envelopeDifference.shape[0] == 2:
            ed_processed = adjustLength(envelopeDifference, audioData.shape[1])
            for channel in range(2):
                transient[channel] = extractMonoTransient(audioData[channel], ed_processed[channel], threshold)
        else:
            raise ValueError("Invalid envelopeDifference shape for stereo audioData")
        return transient
    else:
        raise ValueError("Invalid audioData format")

def CWT_detect_transients_onset(CWT_coeff_l, CWT_coeff_r, sr,
                                     onset_threshold_factor=0.3,
                                     min_distance_ms=50,
                                     freq_weight_power=1.5,
                                     steepness_samples=5):
    """
    Detect transients by finding STEEP onsets only.
    
    Key idea: Real transients have rapid onset over just a few samples.
    Decay oscillations have gradual changes.
    
    Parameters:
    -----------
    steepness_samples : int
        Number of samples to check for steep rise (3-10 recommended)
        Lower = more strict (only very sharp attacks)
    """
    from scipy.signal import find_peaks
    from scipy.ndimage import gaussian_filter1d
    
    min_distance_samples = int(sr * min_distance_ms / 1000)
    
    # Calculate weighted amplitude
    num_scales = CWT_coeff_l.shape[0]
    freq_weights = np.linspace(1.0, 0.3, num_scales) ** freq_weight_power
    freq_weights = freq_weights[:, np.newaxis]
    
    amp_l = np.sum(np.abs(CWT_coeff_l) * freq_weights, axis=0)
    amp_r = np.sum(np.abs(CWT_coeff_r) * freq_weights, axis=0)
    amp_combined = np.maximum(amp_l, amp_r)
    
    # Light smoothing only
    amp_smooth = gaussian_filter1d(amp_combined, sigma=2)
    
    # Calculate onset strength
    onset_strength = np.diff(amp_smooth, prepend=amp_smooth[0])
    onset_strength = np.maximum(onset_strength, 0)
    
    # KEY DIFFERENCE: Calculate "steepness" = sum of onset over short window
    # Real transients accumulate large onset strength quickly
    steepness = np.convolve(onset_strength, np.ones(steepness_samples), mode='same')
    
    # Also require the CURRENT sample to have strong onset
    # (prevents detecting middle of a ramp-up)
    onset_strength_smooth = gaussian_filter1d(onset_strength, sigma=1)
    
    # Combined criterion: both steepness AND instant onset must be strong
    combined_strength = steepness * onset_strength_smooth
    
    # Adaptive threshold
    nonzero_vals = combined_strength[combined_strength > 0]
    if len(nonzero_vals) > 0:
        base_threshold = np.percentile(nonzero_vals, 85)
        threshold = base_threshold * onset_threshold_factor
    else:
        threshold = np.max(combined_strength) * onset_threshold_factor
    
    # Ensure minimum threshold
    threshold = max(threshold, np.max(combined_strength) * 0.1)
    
    # Find peaks in combined strength
    peaks, properties = find_peaks(combined_strength,
                                   height=threshold,
                                   distance=min_distance_samples,
                                   prominence=threshold * 0.2)
    
    # Additional filtering: check if amplitude actually increased significantly
    # in a small window before the peak
    filtered_peaks = []
    for peak in peaks:
        if peak < steepness_samples:
            filtered_peaks.append(peak)
            continue
            
        # Check amplitude change in window before peak
        window_start = max(0, peak - steepness_samples)
        amp_before = amp_smooth[window_start]
        amp_at_peak = amp_smooth[peak]
        
        # Require significant amplitude increase (not just noise fluctuation)
        min_increase = np.max(amp_smooth) * 0.04  # 5% of max amplitude
        if amp_at_peak - amp_before > min_increase:
            filtered_peaks.append(peak)
    
    peaks = np.array(filtered_peaks)
    peak_heights = combined_strength[peaks] if len(peaks) > 0 else np.array([])
    
    return {
        'peaks': peaks,
        'peak_times': peaks / sr,
        'onset_strength': onset_strength,
        'steepness': steepness,
        'combined_strength': combined_strength,
        'amplitude': amp_smooth,
        'threshold': threshold,
        'peak_heights': peak_heights
    }
def analyzeAudioCharacteristics(audioData, sr):
    """
    Analyze audio to determine optimal transient detection parameters.
    
    Returns:
    --------
    dict : Recommended parameters based on audio characteristics
    """
    import librosa
    
    # Convert to mono for analysis
    if audioData.ndim == 2:
        audio_mono = np.mean(audioData, axis=0)
    else:
        audio_mono = audioData
    
    # 1. Calculate spectral centroid (frequency content)
    spectral_centroid = librosa.feature.spectral_centroid(y=audio_mono, sr=sr)[0]
    avg_centroid = np.mean(spectral_centroid)
    
    # 2. Calculate zero crossing rate (transient density indicator)
    zcr = librosa.feature.zero_crossing_rate(audio_mono)[0]
    avg_zcr = np.mean(zcr)
    
    # 3. Calculate RMS energy
    rms = librosa.feature.rms(y=audio_mono)[0]
    
    # 4. Estimate onset density using librosa 
    onset_env = librosa.onset.onset_strength(y=audio_mono, sr=sr)
    onset_frames = librosa.onset.onset_detect(y=audio_mono, sr=sr)
    onset_density = len(onset_frames) / (len(audio_mono) / sr)  # onsets per second
    
    # 5. Calculate temporal characteristics
    abs_audio = np.abs(audio_mono)
    peak_val = np.max(abs_audio)
    peak_idx = np.argmax(abs_audio)
    
    decay_threshold = peak_val * 0.1
    decay_samples = 0
    for i in range(peak_idx, len(abs_audio)):
        if abs_audio[i] < decay_threshold:
            decay_samples = i - peak_idx
            break
    if decay_samples == 0:
        decay_samples = len(abs_audio) - peak_idx
    
    decay_time_ms = (decay_samples / sr) * 1000
    
    # NEW: Calculate compression/dynamic range
    rms_variability = np.std(rms) / (np.mean(rms) + 1e-6)
    dynamic_range_db = 20 * np.log10((np.max(abs_audio) + 1e-6) / (np.mean(abs_audio) + 1e-6))
    
    # 6. Determine audio type based on characteristics
    audio_type = "unknown"
    
    # NEW: Detect compressed electronic music
    if rms_variability < 0.5 and onset_density > 3 and dynamic_range_db < 20:
        audio_type = "electronic_compressed"  # Dense techno/EDM
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
    
    # 7. Set parameters based on audio type
    param_presets = {
        "electronic_compressed": {
            "cwt_onset_threshold": 0.65,      # Much higher - only strong kicks/snares
            "cwt_min_distance_ms": 100,        # Longer distance - avoid hi-hat spam
            "cwt_freq_weight": 1.3,            # Less freq weighting (bass-heavy)
            "time_threshold_factor": 0.45,     # Higher - avoid noise triggering
            "blockSize": 1024,                 # Larger blocks for stability
            "steepness_samples": 3             # Keep short for compressed onsets
        },
        "percussive_rapid": {
            "cwt_onset_threshold": 0.35,
            "cwt_min_distance_ms": 15,
            "cwt_freq_weight": 1.5,
            "time_threshold_factor": 0.28,
            "blockSize": 512
        },
        "percussive_short": {
            "cwt_onset_threshold": 0.3,
            "cwt_min_distance_ms": 40,
            "cwt_freq_weight": 1.5,
            "time_threshold_factor": 0.33,
            "blockSize": 1024
        },
        "plucked_sustained": {
            "cwt_onset_threshold": 0.45,
            "cwt_min_distance_ms": 150,
            "cwt_freq_weight": 2.5,
            "time_threshold_factor": 0.38,
            "blockSize": 2048
        },
        "sustained": {
            "cwt_onset_threshold": 0.5,
            "cwt_min_distance_ms": 100,
            "cwt_freq_weight": 2.0,
            "time_threshold_factor": 0.33,
            "blockSize": 2048
        },
        "general": {
            "cwt_onset_threshold": 0.45,
            "cwt_min_distance_ms": 30,
            "cwt_freq_weight": 1.8,
            "time_threshold_factor": 0.33,
            "blockSize": 1024
        }
    }
    
    params = param_presets[audio_type].copy()
    
    # Fine-tune based on specific measurements
    if onset_density > 10:
        params["cwt_min_distance_ms"] = max(10, params["cwt_min_distance_ms"] * 0.5)
    elif onset_density < 1:
        params["cwt_min_distance_ms"] = min(200, params["cwt_min_distance_ms"] * 1.5)
    
    if avg_centroid > 3000:
        params["cwt_freq_weight"] = min(3.0, params["cwt_freq_weight"] * 1.2)
    elif avg_centroid < 1000:
        params["cwt_freq_weight"] = max(1.0, params["cwt_freq_weight"] * 0.8)
    
    # Return analysis info along with parameters
    return {
        "params": params,
        "audio_type": audio_type,
        "characteristics": {
            "spectral_centroid_hz": avg_centroid,
            "zero_crossing_rate": avg_zcr,
            "onset_density_per_sec": onset_density,
            "decay_time_ms": decay_time_ms,
            "duration_sec": len(audio_mono) / sr,
            "rms_variability": rms_variability,
            "dynamic_range_db": dynamic_range_db
        }
    }
    
    params = param_presets[audio_type].copy()
    
    # Fine-tune based on specific measurements
    # Adjust min_distance based on onset density
    if onset_density > 10:
        params["cwt_min_distance_ms"] = max(10, params["cwt_min_distance_ms"] * 0.5)
    elif onset_density < 1:
        params["cwt_min_distance_ms"] = min(200, params["cwt_min_distance_ms"] * 1.5)
    
    # Adjust frequency weight based on spectral centroid
    if avg_centroid > 3000:
        params["cwt_freq_weight"] = min(3.0, params["cwt_freq_weight"] * 1.2)
    elif avg_centroid < 1000:
        params["cwt_freq_weight"] = max(1.0, params["cwt_freq_weight"] * 0.8)
    
    # Return analysis info along with parameters
    return {
        "params": params,
        "audio_type": audio_type,
        "characteristics": {
            "spectral_centroid_hz": avg_centroid,
            "zero_crossing_rate": avg_zcr,
            "onset_density_per_sec": onset_density,
            "decay_time_ms": decay_time_ms,
            "duration_sec": len(audio_mono) / sr
        }
    }
