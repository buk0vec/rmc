"""

Music 422  Marina Bosi

window.py -- Defines functions to window an array of discrete-time data samples

-----------------------------------------------------------------------
© 2009-2026 Marina Bosi & Richard E. Goldberg -- All rights reserved
-----------------------------------------------------------------------


"""

### ADD YOUR CODE AT THE SPECIFIED LOCATIONS ###

import numpy as np
import scipy.special
import matplotlib.pyplot as plt
from mdct import MDCT

### Problem 1.d ###
def SineWindow(dataSampleArray):
    """
    Returns a copy of the dataSampleArray sine-windowed
    Sine window is defined following pp. 106-107 of
    Bosi & Goldberg, "Introduction to Digital Audio..." book
    """

    ### YOUR CODE STARTS HERE ###
    N = len(dataSampleArray)
    ns = np.arange(N)
    window = np.sin(np.pi * (ns + 0.5) / N)
    return dataSampleArray * window 
    ### YOUR CODE ENDS HERE ###


def HanningWindow(dataSampleArray):
    """
    Returns a copy of the dataSampleArray Hanning-windowed
    Hann window is defined following pp. 106-107 of
    Bosi & Goldberg, "Introduction to Digital Audio..." book
    """

    ### YOUR CODE STARTS HERE ###
    N = len(dataSampleArray)
    ns = np.arange(N)
    window = 0.5 * (1 - np.cos(2 * np.pi * (ns + 0.5) / N))
    return dataSampleArray * window 
    ### YOUR CODE ENDS HERE ###


### Problem 1.d - OPTIONAL ###
def KBDWindow(dataSampleArray,alpha=4.):
    """
    Returns a copy of the dataSampleArray KBD-windowed
    KBD window is defined following the KDB Window handout in the 
	Canvas Files/Assignments/HW3 folder
    """

    ### YOUR CODE STARTS HERE ###
    N = len(dataSampleArray)
    ns_div2p1 = np.arange(N // 2 + 1, dtype=np.int32)
    kb = scipy.special.i0(alpha * np.pi * np.sqrt(1 - ((2 * ns_div2p1 + 1) / (N / 2 + 1) - 1) ** 2 )) / scipy.special.i0(np.pi * alpha)
    window_half = np.sqrt(np.cumsum(kb[:N//2]) / np.sum(kb))
    window = np.concatenate([window_half, np.flip(window_half)])
    return window * dataSampleArray 
    ### YOUR CODE ENDS HERE ###

#-----------------------------------------------------------------------------

#Testing code
if __name__ == "__main__":

    ### YOUR TESTING CODE STARTS HERE ###
    plt.figure()
    plt.plot(SineWindow(np.ones(1024)), label="Sine")

    plt.plot(HanningWindow(np.ones(1024)), label="Hanning")

    plt.plot(KBDWindow(np.ones(1024), alpha=4), label="KBD $\\alpha=4$")
    plt.legend()
    plt.savefig("windows.png")
    plt.show()
    
    ns = np.arange(1024)
    Fs = 44100
    x = np.cos(2 * np.pi * 7000 * ns / Fs)
    
    sin_fft = scipy.fft.fft(SineWindow(x))
    hann_fft =  scipy.fft.fft(HanningWindow(x))
    sin_mdct = MDCT(SineWindow(x), 512, 512)
    kbd_mdct = MDCT(KBDWindow(x), 512, 512)
    
    sin_fft_spl = 96 + 10 * np.log10(4 / (np.mean(SineWindow(np.ones(1024)) ** 2) * 1024 ** 2) * np.sum(np.abs(sin_fft[:512]) ** 2))
    hann_fft_spl =  96 + 10 * np.log10(4 / (np.mean(HanningWindow(np.ones(1024)) ** 2) * 1024 ** 2) * np.sum(np.abs(hann_fft[:512]) ** 2))
    # We apply 2/N scaling to the MDCT, so we modify the textbook formula to use (N/2)X[k] to get the actual SPL
    # |(N/2)X[k]|^2 -> N^2/4 |X[k]|^2, N^2/4  * 8/N^2<w^2> = 2/<w^2>
    sin_mdct_spl =  96 + 10 * np.log10(2 / (np.mean(SineWindow(np.ones(1024)) ** 2) ) * np.sum(np.abs(sin_mdct) ** 2))
    kbd_mdct_spl =  96 + 10 * np.log10(2 / (np.mean(KBDWindow(np.ones(1024)) ** 2)) * np.sum(np.abs(kbd_mdct) ** 2))
    
    plt.figure()
    
    plt.bar(["Sine DFT", "Hanning DFT", "Sin MDCT", "KBD MDCT"],[sin_fft_spl, hann_fft_spl, sin_mdct_spl, kbd_mdct_spl])
    plt.ylabel("SPL (db)")
    plt.savefig("spl.png")
    plt.show()
    
    print(sin_fft_spl, hann_fft_spl, sin_mdct_spl, kbd_mdct_spl)
    ### YOUR TESTING CODE ENDS HERE ###

