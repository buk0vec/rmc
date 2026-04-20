"""
psychoac.py -- masking models implementation

-----------------------------------------------------------------------
(c) 2011-2026 Marina Bosi & Richard E. Goldberg -- All rights reserved
-----------------------------------------------------------------------
"""

import numpy as np
from window import *
import scipy.fft
import scipy.signal
import scipy.stats
from numba import njit

@njit(cache=True)
def SPL(intensity):
    """
    Returns the SPL corresponding to intensity
    """
    return np.where(intensity == 0, -30, np.maximum(96 + 10 * np.log10(intensity), -30))

_LOG10_OVER_10 = np.log(10) / 10

@njit(cache=True)
def Intensity(spl):
    """
    Returns the intensity  for SPL spl
    """
    return np.exp((spl - 96) * _LOG10_OVER_10)

@njit(cache=True)
def Thresh(f):
    """Returns the threshold in quiet measured in SPL at frequency f (in Hz)"""
    f = np.maximum(f, 20)
    t1 = 3.64 * ((f / 1000) ** (-0.8))
    t2 = -6.5 * np.exp(-0.6 * ((f / 1000 - 3.3) ** 2))
    t3 = (10 ** (-3)) * ((f / 1000) ** 4)
    return t1 + t2 + t3

@njit(cache=True)
def Bark(f):
    """Returns the bark-scale frequency for input frequency f (in Hz) """
    t1 = 13 * np.arctan(0.76 * f / 1000)
    t2 = 3.5 * np.arctan((f / 7500) ** 2)

    return t1 + t2

class Masker:
    """
    a Masker whose masking curve drops linearly in Bark beyond 0.5 Bark from the
    masker frequency
    """

    def __init__(self,f,SPL,isTonal=True):
        """
        initialized with the frequency and SPL of a masker and whether or not
        it is Tonal
        """
        self.f = f
        self.SPL = SPL
        self.isTonal = isTonal

    def IntensityAtFreq(self,freq):
        """The intensity at frequency freq"""
        return self.IntensityAtBark(Bark(freq))
    def IntensityAtBark(self,z):
        """The intensity at Bark location z"""
        dz = z - Bark(self.f)
        if dz < -0.5:
            spl = -27 * (abs(dz) - 0.5)
        elif dz > 0.5:
            spl = (-27 + 0.367 * max(self.SPL - 40, 0)) * (abs(dz) - 0.5)
        else:
            spl = 0
        spl = self.SPL + spl
        if self.isTonal:
            spl -= 16
        else:
            spl -= 6
        return Intensity(spl)

    def vIntensityAtBark(self,zVec):
        """The intensity at vector of Bark locations zVec"""
        spl = np.zeros_like(zVec)
        dz = zVec - Bark(self.f)
        spl = np.where(dz < -0.5, -27 * (np.abs(dz) - 0.5), spl)
        spl = np.where(dz > 0.5,(-27 + 0.367 * max(self.SPL - 40, 0)) * (np.abs(dz) - 0.5), spl)
        spl = self.SPL + spl
        if self.isTonal:
            spl -= 16
        else:
            spl -= 6
        return Intensity(spl)

# Default data for 25 scale factor bands based on the traditional 25 critical bands
cbFreqLimits = [100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720, 2000, 2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500] 

def AssignMDCTLinesFromFreqLimits(nMDCTLines, sampleRate, flimit = cbFreqLimits):
    """
    Assigns MDCT lines to scale factor bands for given sample rate and number
    of MDCT lines using predefined frequency band cutoffs passed as an array
    in flimit (in units of Hz). If flimit isn't passed it uses the traditional
    25 Zwicker & Fastl critical bands as scale factor bands.
    """
    nyq = sampleRate / 2
    freqs = np.arange(nMDCTLines)/ nMDCTLines * nyq
    flimit = [0] + flimit
    if flimit[-1] < nyq:
        flimit.append(nyq)
    num_lines, _  = np.histogram(freqs, bins=flimit)
    
    return num_lines 

class ScaleFactorBands:
    """
    A set of scale factor bands (each of which will share a scale factor and a
    mantissa bit allocation) and associated MDCT line mappings.

    Instances know the number of bands nBands; the upper and lower limits for
    each band lowerLimit[i in range(nBands)], upperLimit[i in range(nBands)];
    and the number of lines in each band nLines[i in range(nBands)]
    """

    def __init__(self,nLines):
        """
        Assigns MDCT lines to scale factor bands based on a vector of the number
        of lines in each band
        """
        self.nLines = nLines
        self.nBands = len(nLines)
        self.lowerLine = np.cumsum(np.concatenate([[0], nLines[:-1]]))
        self.upperLine = np.cumsum(nLines) - 1

_psychoac_cache = {}

@njit(cache=True)
def _compute_masking_curve(masker_bark_f, masker_spl_v, masker_tonal, bark_MDCT):
    """
    JIT-compiled masking curve calculation.

    Computes the masked threshold contribution from all maskers at each MDCT line.
    This is the hot loop extracted from getMaskedThreshold.

    Args:
        masker_bark_f: (n_maskers,) Bark frequencies of maskers
        masker_spl_v: (n_maskers,) SPL values of maskers
        masker_tonal: (n_maskers,) boolean array, True if tonal masker
        bark_MDCT: (n_mdct,) Bark frequencies of MDCT lines

    Returns:
        total_spl: (n_maskers, n_mdct) masking SPL contribution from each masker at each MDCT line
    """
    # Broadcast to create (n_maskers, n_mdct) matrix
    # dz[i,j] = bark_MDCT[j] - masker_bark_f[i]
    dz = bark_MDCT.reshape(1, -1) - masker_bark_f.reshape(-1, 1)
    abs_dz = np.abs(dz)

    # Initialize slope matrix
    slope = np.zeros_like(dz)

    # Lower slope: dz < -0.5
    slope = np.where(dz < -0.5, -27.0 * (abs_dz - 0.5), slope)

    # Upper slope: dz > 0.5
    # Note: (-27 + 0.367 * max(SPL - 40, 0)) depends on each masker
    upper = (-27.0 + 0.367 * np.maximum(masker_spl_v - 40.0, 0.0)).reshape(-1, 1)
    slope = np.where(dz > 0.5, upper * (abs_dz - 0.5), slope)

    # Add base SPL for each masker (broadcast across MDCT lines)
    total_spl = masker_spl_v.reshape(-1, 1) + slope

    # Subtract tonal/noise offset: -16 for tonal, -6 for noise
    tonal_offset = np.where(masker_tonal, -16.0, -6.0).reshape(-1, 1)
    total_spl = total_spl + tonal_offset

    return total_spl

def _build_psychoac_cache(N, N_mdct, sampleRate):
    """Precompute all values in getMaskedThreshold that depend only on N and sampleRate."""
    w_n = KBDWindow(np.ones(N), alpha=4)
    P_w = np.mean(w_n ** 2)
    freqs = scipy.fft.rfftfreq(N, 1/sampleRate)
    bands_fft = ScaleFactorBands(AssignMDCTLinesFromFreqLimits(N//2 + 1, sampleRate))
    # Per-band noise masker frequencies: gmean of FFT bin indices, only depends on band boundaries
    band_noise_freqs = np.zeros(bands_fft.nBands)
    band_valid = np.zeros(bands_fft.nBands, dtype=bool)
    for b in range(bands_fft.nBands):
        if bands_fft.nLines[b] > 0:
            idx = int(np.round(scipy.stats.gmean(
                np.arange(bands_fft.lowerLine[b], bands_fft.upperLine[b] + 1)
            )))
            band_noise_freqs[b] = freqs[idx]
            band_valid[b] = True
    MDCT_freqs = (np.arange(N_mdct) / N_mdct) * sampleRate / 2
    quiet_thresh = Thresh(MDCT_freqs)
    intensity_quiet = Intensity(quiet_thresh)
    bark_MDCT = Bark(MDCT_freqs)
    return (w_n, P_w, freqs, bands_fft, band_noise_freqs, band_valid, MDCT_freqs, quiet_thresh, intensity_quiet, bark_MDCT)


def getMaskedThreshold(data, MDCTdata, MDCTscale, sampleRate, sfBands):
    """
    Return Masked Threshold evaluated at MDCT lines.

    Used by CalcSMR, but can also be called from outside this module, which may
    be helpful when debugging the bit allocation code.
    """
    N = len(data)
    N_mdct = len(MDCTdata)
    key = (N, N_mdct, sampleRate)
    if key not in _psychoac_cache:
        _psychoac_cache[key] = _build_psychoac_cache(N, N_mdct, sampleRate)
    w_n, P_w, freqs, bands_fft, band_noise_freqs, band_valid, MDCT_freqs, quiet_thresh, intensity_quiet, bark_MDCT = _psychoac_cache[key]

    x_windowed = data * w_n
    x_fft = scipy.fft.rfft(x_windowed)
    x_mag = (np.abs(x_fft) ** 2)
    x_intensity = 4/(N ** 2 * P_w) * x_mag
    x_spl = SPL(x_intensity)

    # Following textbook pg. 281-282, except just using scipy for peak finding
    peaks, _ = scipy.signal.find_peaks(x_spl, prominence=5)
    peak_amp_squared = x_intensity[peaks - 1] + x_intensity[peaks] + x_intensity[peaks + 1]
    peak_spl = SPL(peak_amp_squared)
    peak_freqs = sampleRate/N * \
        ((peaks - 1) * x_mag[peaks - 1] + peaks * x_mag[peaks] + (peaks + 1) * x_mag[peaks+1]) \
        / (x_mag[peaks - 1] + x_mag[peaks] + x_mag[peaks + 1])

    # ===== OPTIMIZED: Build masker arrays directly (no Masker objects) =====
    # Count total maskers: peaks + valid noise bands
    n_peaks = len(peaks)
    n_noise = np.sum(band_valid)
    n_maskers = n_peaks + n_noise

    # Pre-allocate masker arrays
    masker_freqs = np.zeros(n_maskers)
    masker_spl = np.zeros(n_maskers)
    masker_tonal = np.zeros(n_maskers, dtype=np.bool_)

    # Fill tonal (peak) maskers
    masker_freqs[:n_peaks] = peak_freqs
    masker_spl[:n_peaks] = peak_spl
    masker_tonal[:n_peaks] = True

    # Fill noise maskers (from bands with no peaks)
    noise_intensity = np.copy(x_intensity)
    noise_intensity[peaks] = 0

    idx = n_peaks
    for b in range(bands_fft.nBands):
        if band_valid[b]:
            noise_int = np.sum(noise_intensity[bands_fft.lowerLine[b]:bands_fft.upperLine[b] + 1])
            masker_freqs[idx] = band_noise_freqs[b]
            masker_spl[idx] = SPL(noise_int)
            masker_tonal[idx] = False
            idx += 1

    # Convert to Bark scale
    masker_bark_f = Bark(masker_freqs)

    # ===== OPTIMIZED: Use JIT-compiled masking curve calculation =====
    total_spl = _compute_masking_curve(masker_bark_f, masker_spl, masker_tonal, bark_MDCT)

    # Sum masker intensities and add threshold in quiet
    masked_threshold = SPL(np.sum(Intensity(total_spl), axis=0) + intensity_quiet)

    return masked_threshold
    


def CalcSMRs(data, MDCTdata, MDCTscale, sampleRate, sfBands):
    """
    Set SMR for each critical band in sfBands.

    Arguments:
                data:       is an array of N time domain samples
                MDCTdata:   is an array of N/2 MDCT frequency coefficients for the time domain samples
                            in data; note that the MDCT coefficients have been scaled up by a factor
                            of 2^MDCTscale
                MDCTscale:  corresponds to an overall scale factor 2^MDCTscale for the set of MDCT
                            frequency coefficients
                sampleRate: is the sampling rate of the time domain samples
                sfBands:    points to information about which MDCT frequency lines
                            are in which scale factor band

    Returns:
                SMR[sfBands.nBands] is the maximum signal-to-mask ratio in each
                                    scale factor band

    Logic:
                Performs an FFT of data[N] and identifies tonal and noise maskers.
                Combines their relative masking curves and the hearing threshold
                to calculate the overall masked threshold at the MDCT frequency locations. 
				Then determines the maximum signal-to-mask ratio within
                each critical band and returns that result in the SMR[] array.
    """

    
    masked_threshold = getMaskedThreshold(data, MDCTdata, MDCTscale, sampleRate, sfBands)
    
    mdct = MDCTdata / (2 ** MDCTscale)
    mdct_mag = np.abs(mdct) ** 2
    # Use scaling that we used in last homework. Assuming KBD w/ average power = 1/2
    mdct_intensity= 1 / (1/2) * mdct_mag
    mdct_spl = SPL(mdct_intensity)
    
    smr = np.zeros(sfBands.nBands)
    
    for b in range(sfBands.nBands):
        low = sfBands.lowerLine[b]
        hi = sfBands.upperLine[b]
        smr[b] = np.max(mdct_spl[low:hi + 1] - masked_threshold[low:hi + 1])
        
        
    return smr

#-----------------------------------------------------------------------------

#Testing code
if __name__ == "__main__":
    pass