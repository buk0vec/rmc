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

def SPL(intensity):
    """
    Returns the SPL corresponding to intensity 
    """
    return np.where(intensity == 0, -30, np.maximum(96 + 10 * np.log10(intensity), -30))

def Intensity(spl):
    """
    Returns the intensity  for SPL spl
    """
    return 10. ** ((spl - 96) / 10)

def Thresh(f):
    """Returns the threshold in quiet measured in SPL at frequency f (in Hz)"""
    f = np.maximum(f, 20)
    t1 = 3.64 * ((f / 1000) ** (-0.8))
    t2 = -6.5 * np.exp(-0.6 * ((f / 1000 - 3.3) ** 2))
    t3 = (10 ** (-3)) * ((f / 1000) ** 4)
    return t1 + t2 + t3

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

def getMaskedThreshold(data, MDCTdata, MDCTscale, sampleRate, sfBands):
    """
    Return Masked Threshold evaluated at MDCT lines.

    Used by CalcSMR, but can also be called from outside this module, which may
    be helpful when debugging the bit allocation code.
    """
    N = len(data)
    N_mdct = len(MDCTdata)
    w_n = KBDWindow(np.ones(N), alpha=4)
    x_windowed = data * w_n
    x_fft = scipy.fft.rfft(x_windowed)
    x_mag = (np.abs(x_fft) ** 2)
    x_intensity = 4/(N ** 2) * x_mag
    x_spl = SPL(x_intensity)
    freqs = scipy.fft.rfftfreq(N, 1/sampleRate)
    
    # Following textbook pg. 281-282, except just using scipy for peak finding
    peaks, _ = scipy.signal.find_peaks(x_spl, prominence=5)
    peak_amp_squared = x_intensity[peaks - 1] + x_intensity[peaks] + x_intensity[peaks + 1]
    peak_spl = SPL(peak_amp_squared)
    peak_freqs = sampleRate/N * \
    ((peaks - 1) * x_mag[peaks - 1] + peaks * x_mag[peaks] + (peaks + 1) * x_mag[peaks+1]) \
    / (x_mag[peaks - 1] + x_mag[peaks] + x_mag[peaks + 1])
    
    maskers = [Masker(f, spl, isTonal=True) for f, spl in zip(peak_freqs, peak_spl)]
    
    noise_intensity = np.copy(x_intensity)
    noise_intensity[peaks] = 0
    
    # repurpose scalefactorbands class
    bands_fft = ScaleFactorBands(AssignMDCTLinesFromFreqLimits(len(x_intensity), sampleRate))
    # Identify noise maskers
    for band in range(bands_fft.nBands):
        if bands_fft.nLines[band] > 0:
            noise_int = np.sum(noise_intensity[bands_fft.lowerLine[band]:bands_fft.upperLine[band] + 1])
            if bands_fft.nLines[band] > 0:
                # MPEG-1 geometric mean of fft band indices
                noise_fft_idx= int(np.round(scipy.stats.gmean(np.arange(bands_fft.lowerLine[band], bands_fft.upperLine[band] + 1))))
                noise_freq = freqs[noise_fft_idx]
                maskers.append(Masker(noise_freq, SPL(noise_int), isTonal=False))

    
    MDCT_freqs =  (np.arange(N_mdct) / N_mdct) * sampleRate / 2
    mask_curve_intensities = [m.vIntensityAtBark(Bark(MDCT_freqs)) for m in maskers]
    quiet_thresh = Thresh(MDCT_freqs)
    masked_threshold = SPL(np.sum([*mask_curve_intensities, Intensity(quiet_thresh)], axis=0)) 
    
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