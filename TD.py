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
