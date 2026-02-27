"""
Music 422 Marina Bosi

- mdct.py -- Computes a reasonably fast MDCT/IMDCT using the FFT/IFFT

-----------------------------------------------------------------------
© 2009-2026 Marina Bosi & Richard E. Goldberg -- All rights reserved
-----------------------------------------------------------------------

"""

### ADD YOUR CODE AT THE SPECIFIED LOCATIONS ###

import numpy as np
import scipy.fft
import timeit

### Problem 1.a ###
def MDCTslow(data, a, b, isInverse=False):
    """
    Slow MDCT algorithm for window length a+b following pp. 130 of
    Bosi & Goldberg, "Introduction to Digital Audio..." book
    and where the 2/N factor is included in the forward transform instead of inverse.
    a: left half-window length
    b: right half-window length
    """

    ### YOUR CODE STARTS HERE ###
    N = a + b
    ks = np.arange(N // 2)
    ns = np.arange(N)
    n_0 = (b + 1) / 2
    if isInverse:
        basis = np.cos(np.pi * 2 / N * (ns[:, None] + n_0) * (ks + 0.5))
    else:
        basis = np.cos(np.pi * 2 / N * (ns  + n_0) * (ks[:, None]+ 0.5)) / N
    output = 2 * basis @ data
    return output 
    ### YOUR CODE ENDS HERE ###

### Problem 1.c ###
def MDCT(data, a, b, isInverse=False):
    """
    Fast MDCT algorithm for window length a+b following pp. 141-143 of
    Bosi & Goldberg, "Introduction to Digital Audio..." book
    and where the 2/N factor is included in forward transform instead of inverse.
    a: left half-window length
    b: right half-window length
    """

    ### YOUR CODE STARTS HERE ###
    N = a + b
    ks = np.arange(N // 2)
    ns = np.arange(N)
    n_0 = (b + 1) / 2
    pre_twiddles = np.exp(-1j * np.pi * ns / N)
    out = data * pre_twiddles
    out = scipy.fft.fft(out, norm="forward")
    post_twiddles = np.exp(-1j * 2 * np.pi / N * n_0 * (ks + 0.5))
    out = out[:N // 2] * post_twiddles
    return out.real * 2
    ### YOUR CODE ENDS HERE ###

def IMDCT(data,a,b):

    ### YOUR CODE STARTS HERE ###
    N = a + b
    ks = np.arange(N)
    ns = np.arange(N)
    n_0 = (b + 1) / 2
    out = np.concatenate([data, -np.flip(data)])
    pre_twiddles = np.exp(2j * np.pi / N * ks * n_0)
    out = out * pre_twiddles
    out = scipy.fft.ifft(out, norm="forward")
    post_twiddles = np.exp(1j * np.pi/N * (ns + n_0))
    out = post_twiddles * out
    return out.real
    ### YOUR CODE ENDS HERE ###

#-----------------------------------------------------------------------------

#Testing code
if __name__ == "__main__":

    ### YOUR TESTING CODE STARTS HERE ###
    def print1(*args):
        print("|-", *args)
    def print2(*args):
        print("  |-", *args)
        
    def printh(*args):
        print("\n|", *args)
    
    x = np.array([0,1,2,3,4,4,4,4,3,1,-1,-3], dtype=np.float32)
    x = np.pad(x, (4, 4), 'constant', constant_values=(0,0))
    a = b = 4
    np.set_printoptions(precision=4, suppress=True)
    
    # for i in range(4):
    #    x_mdct = MDCTslow(x[i*4:i*4 + a + b], a, b)
    #    print(x[i*4:i*4 + a + b])
    #    print(MDCTslow(x_mdct, a, b, isInverse=True) / 2)
    
    # Fast method that does not follow the steps exactly
    output = np.zeros_like(x, dtype=np.float32)
    for i in range(4):
        x_out = MDCTslow(MDCTslow(x[i*4:i*4 + 8], a, b), a, b, isInverse=True) / 2
        output[i*4:i*4 + 8] += x_out
    
    printh("Fast method")
    print1(x[4:-4])
    print1(output[4:-4])
    
    # Slower method that does follow the input steps exactly
    x = np.array([0,1,2,3,4,4,4,4,3,1,-1,-3], dtype=np.float32)
    prior_half = np.zeros(4)
    prior_out = np.zeros(4)
    output = np.empty((0))
    for i in range(4):
        if i == 3:
            block = np.concatenate([prior_half, np.zeros(4)])
        else:
            block = np.concatenate([prior_half, x[i*4: (i + 1) * 4]])
        x_out = MDCTslow(MDCTslow(block, a, b), a, b, isInverse=True) / 2
        output = np.concatenate([output, prior_out + x_out[:4]])
        prior_half = block[4:]
        prior_out = x_out[4:]
       
    printh("Correct method")
    print1(x)
    print1(output)
    
    printh("MDCT Comparison")
    print1("Sanity check")
    x_out = MDCTslow(x[:8], a, b)
    x_out_fast = MDCT(x[:8], a, b)
    print2(x_out)
    print2(x_out_fast)
    assert np.allclose(x_out, x_out_fast)
    
    print1("Speed comparison")
    test_signal = np.random.rand(2048)
    assert np.allclose(MDCTslow(test_signal, 1024, 1024), MDCT(test_signal, 1024, 1024))
    test_mdct = MDCT(test_signal, 1024, 1024)
    assert np.allclose(MDCTslow(test_mdct, 1024, 1024, isInverse=True), IMDCT(test_mdct, 1024, 1024))
    
    slow_time = timeit.timeit(lambda: MDCTslow(test_signal, 1024, 1024), number=100) / 100
    fast_time = timeit.timeit(lambda: MDCT(test_signal, 1024, 1024), number=100) / 100
    
    
    print2(f"Slow time: {slow_time}s")
    print2(f"Fast time: {fast_time}s")
    print2(f"Speedup: {slow_time/fast_time:0.3f}x")
    
    printh("IMDCT Comparison")
    print1("Sanity check")
    x_out = MDCTslow(x_out, a, b, isInverse=True)
    x_out_fast = IMDCT(x_out_fast, a, b)
    print2(x_out)
    print2(x_out_fast)
    assert np.allclose(x_out, x_out_fast)
    
    test_signal = MDCT(test_signal, 1024, 1024)
    
    slow_time = timeit.timeit(lambda: MDCTslow(test_signal, 1024, 1024, isInverse=True), number=100) / 100
    fast_time = timeit.timeit(lambda: IMDCT(test_signal, 1024, 1024), number=100) / 100
    print1("Speed comparison")
    print2(f"Slow time: {slow_time}s")
    print2(f"Fast time: {fast_time}s")
    print2(f"Speedup: {slow_time/fast_time:0.3f}x")
    
    ### YOUR TESTING CODE ENDS HERE ###

