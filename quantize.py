"""
Music 422 - Marina Bosi

quantize.py -- routines to quantize and dequantize floating point values
between -1.0 and 1.0 ("signed fractions")

-----------------------------------------------------------------------
© 2009-2026 Marina Bosi & Richard E. Goldberg -- All rights reserved
-----------------------------------------------------------------------
"""

### ADD YOUR CODE AT THE SPECIFIED LOCATIONS ###

import numpy as np

### Problem 1.a.i ###
def QuantizeUniform(aNum,nBits):
    """
    Uniformly quantize signed fraction aNum with nBits
    """
    #Notes:
    #The overload level of the quantizer should be 1.0

    ### YOUR CODE STARTS HERE ###
    s = 0 if aNum >= 0 else 1
    code = (1 << (nBits - 1)) - 1 if abs(aNum) >= 1 else int((((1 << nBits) - 1) * abs(aNum)+1)/2)
    aQuantizedNum = (s << (nBits - 1)) | code
    ### YOUR CODE ENDS HERE ###

    return aQuantizedNum

### Problem 1.a.i ###
def DequantizeUniform(aQuantizedNum,nBits):
    """
    Uniformly dequantizes nBits-long number aQuantizedNum into a signed fraction
    """

    ### YOUR CODE STARTS HERE ###
    sign = (aQuantizedNum >> (nBits - 1)) 
    aQuantizedNum = aQuantizedNum ^ (sign << (nBits - 1))
    sign = sign * -2 + 1
    number = 2 * aQuantizedNum / ((1 << nBits) - 1)
    aNum = sign * number 
    ### YOUR CODE ENDS HERE ###

    return aNum

### Problem 1.a.ii ###
def vQuantizeUniform(aNumVec, nBits):
    """
    Uniformly quantize vector aNumberVec of signed fractions with nBits
    """

    aQuantizedNumVec = np.zeros_like(aNumVec, dtype = int) # REMOVE THIS LINE WHEN YOUR FUNCTION IS DONE

    #Notes:
    #Make sure to vectorize properly your function as specified in the homework instructions

    ### YOUR CODE STARTS HERE ###
    s = np.where(aNumVec >= 0, 0, 1).astype(np.uint64)
    code = np.where(
        np.abs(aNumVec) >= 1, 
        (1 << (nBits - 1)) - 1, 
        ((((1 << nBits) - 1) * np.abs(aNumVec)+1)/2)
    ).astype(np.uint64)
    aQuantizedNumVec = (s << np.uint64(nBits - 1)) | code
    ### YOUR CODE ENDS HERE ###

    return aQuantizedNumVec

### Problem 1.a.ii ###
def vDequantizeUniform(aQuantizedNumVec, nBits):
    """
    Uniformly dequantizes vector of nBits-long numbers aQuantizedNumVec into vector of  signed fractions
    """

    aNumVec = np.zeros_like(aQuantizedNumVec, dtype = float) # REMOVE THIS LINE WHEN YOUR FUNCTION IS DONE

    ### YOUR CODE STARTS HERE ###
    sign = (aQuantizedNumVec >> (nBits - 1))
    aQuantizedNumVec = aQuantizedNumVec ^ (sign << (nBits - 1))
    sign = 1 + -2 * np.int64(sign)
    number = 2 * np.abs(aQuantizedNumVec) / ((1 << nBits) - 1)
    aNumVec = sign * number
    ### YOUR CODE ENDS HERE ###

    return aNumVec

### Problem 1.b ###
def ScaleFactor(aNum, nScaleBits=3, nMantBits=5):
    """
    Return the floating-point scale factor for a  signed fraction aNum given nScaleBits scale bits and nMantBits mantissa bits
    """
    #Notes:
    #The scale factor should be the number of leading zeros

    ### YOUR CODE STARTS HERE ###
    quant_bits = (1 << nScaleBits) - 1 + nMantBits
    num_q = QuantizeUniform(aNum, quant_bits)
    # Go from two's complement to twelve-bit folded. Don't care about sign bit for this
    
    binary = np.binary_repr(num_q, width=quant_bits)
    scale = min(binary[1:].find('1'), (1 << nScaleBits) - 1)
    if scale < 0:
        scale = (1 << nScaleBits) - 1
    ### YOUR CODE ENDS HERE ###

    return scale

### Problem 1.b ###
def MantissaFP(aNum, scale, nScaleBits=3, nMantBits=5):
    """
    Return the floating-point mantissa for a  signed fraction aNum given nScaleBits scale bits and nMantBits mantissa bits
    """


    ### YOUR CODE STARTS HERE ###
    quant_bits = (1 << nScaleBits) - 1 + nMantBits
    num_q = QuantizeUniform(aNum, quant_bits)
    s = num_q >> (quant_bits - 1)

    if nMantBits == 0:
        mantissa = 0

    elif scale == (1 << nScaleBits) - 1:
        mantissa = num_q & ((1 << (nMantBits - 1)) - 1)
    
    else:
        # Filter out bits before the scale + 2nd
        mask = (1 << (quant_bits - (scale + 2))) - 1
        mantissa = num_q & mask
        # Filter out nMantBits-1 bits after the scale + 2nd
        mask = ~((1 << (quant_bits - (scale + nMantBits + 1)))-1)
        mantissa &= mask
        # Reshift mantissa bits back
        mantissa >>= quant_bits - (scale + nMantBits + 1)
    
    mantissa |= s <<(nMantBits - 1)
    ### YOUR CODE ENDS HERE ###

    return mantissa

### Problem 1.b ###
def DequantizeFP(scale, mantissa, nScaleBits=3, nMantBits=5):
    """
    Returns a  signed fraction for floating-point scale and mantissa given specified scale and mantissa bits
    """

    ### YOUR CODE STARTS HERE ###
    quant_bits = (1 << nScaleBits) - 1 + nMantBits
    if nMantBits == 0:
        return 0
    sign = mantissa >> (nMantBits - 1)
    
    if scale == (1 << nScaleBits) - 1:
        # Use the first 4 digits of the number fo the mantissa
        num_q = mantissa & ~(1 << (nMantBits - 1))
    
    else:
        # Add a 1 onto first mantissa bit 
        num_q = mantissa | (1 << (nMantBits - 1))
        # Shift mantissa bits forward as needed
        shift_amt = quant_bits - (scale + nMantBits + 1)
        num_q = num_q << shift_amt
        
    num_q |= sign << (quant_bits - 1)
    aNum = DequantizeUniform(num_q, quant_bits)
    ### YOUR CODE ENDS HERE ###

    return aNum

### Problem 1.c.i ###
def Mantissa(aNum, scale, nScaleBits=3, nMantBits=5):
    """
    Return the block floating-point mantissa for a  signed fraction aNum given nScaleBits scale bits and nMantBits mantissa bits
    """


    ### YOUR CODE STARTS HERE ###
    quant_bits = (1 << nScaleBits) - 1 + nMantBits
    num_q = QuantizeUniform(aNum, quant_bits)
    s = num_q >> (quant_bits - 1)

    # If scale is maxed out, just encode the last nMantBits - 1 bits
    if scale == (1 << nScaleBits) - 1:
        mantissa = num_q & ((1 << (nMantBits - 1)) - 1)
    
    else:
        # Filter out bits before the scale + 1st (not assuming leading 1)
        mask = (1 << (quant_bits - (scale + 1))) - 1
        mantissa = num_q & mask
        # Filter out nMantBits-1 bits after the scale + 1st
        mask = ~((1 << (quant_bits - (scale + nMantBits)))-1)
        mantissa &= mask
        # Reshift mantissa bits back
        mantissa >>= quant_bits - (scale + nMantBits)
    
    mantissa |= s << (nMantBits - 1)

    ### YOUR CODE ENDS HERE ###

    return mantissa

### Problem 1.c.i ###
def Dequantize(scale, mantissa, nScaleBits=3, nMantBits=5):
    """
    Returns a  signed fraction for block floating-point scale and mantissa given specified scale and mantissa bits
    """

    ### YOUR CODE STARTS HERE ###
    quant_bits = (1 << nScaleBits) - 1 + nMantBits
    if nMantBits == 0:
        return 0
    s = ((mantissa >> (nMantBits - 1)) & 1) 
    
    # Use the first 4 digits of the number for the mantissa
    num_q = mantissa & ~(1 << (nMantBits - 1))
    
    if scale != (1 << nScaleBits) - 1:
        # Do NOT add 1 onto first mantissa bit, instead zero out the sign bit
        # num_q = mantissa & ~(1 << (nMantBits - 1)) (we do this above)
        # Shift mantissa bits forward as needed
        shift_amt = quant_bits - (scale + nMantBits)
        num_q = num_q << shift_amt

    num_q |= s << (quant_bits - 1)
    
    aNum = DequantizeUniform(num_q, quant_bits)

    ### YOUR CODE ENDS HERE ###

    return aNum

### Problem 1.c.ii ###
def vMantissa(aNumVec, scale, nScaleBits=3, nMantBits=5):
    """
    Return a vector of block floating-point mantissas for a vector of  signed fractions aNum given nScaleBits scale bits and nMantBits mantissa bits
    """

    mantissaVec = np.zeros_like(aNumVec, dtype = int) # REMOVE THIS LINE WHEN YOUR FUNCTION IS DONE

    ### YOUR CODE STARTS HERE ###
    if nMantBits == 0:
        return np.zeros_like(aNumVec, dtype = np.uint64)
    quant_bits = np.uint64((1 << nScaleBits) - 1 + nMantBits)
    num_q = vQuantizeUniform(aNumVec, quant_bits).astype(np.uint64)
    s = num_q >> (quant_bits - 1)
    
    # If scale is maxed out, just encode the last nMantBits - 1 bits
    if scale == (1 << nScaleBits) - 1:
        mantissa = num_q & ((1 << (nMantBits - 1)) - 1)
    
    
    else:
        # Filter out bits before the scale + 1st (not assuming leading 1)
        mask = (1 << (quant_bits - (scale + 1))) - 1
        mantissa = num_q & mask
        # Filter out nMantBits-1 bits after the scale + 1st
        mask = ~((np.uint64(1) << (quant_bits - (scale + nMantBits)))-1)
        mantissa &= mask
        # Reshift mantissa bits back
        mantissa >>= quant_bits - (scale + nMantBits)
        
    mantissaVec = mantissa | s <<(nMantBits - 1)

    ### YOUR CODE ENDS HERE ###

    return mantissaVec

### Problem 1.c.ii ###
def vDequantize(scale, mantissaVec, nScaleBits=3, nMantBits=5):
    """
    Returns a vector of  signed fractions for block floating-point scale and vector of block floating-point mantissas given specified scale and mantissa bits
    """

    aNumVec = np.zeros_like(mantissaVec, dtype = float) # REMOVE THIS LINE WHEN YOUR FUNCTION IS DONE

    ### YOUR CODE STARTS HERE ###
    if nMantBits == 0:
        return np.zeros_like(mantissaVec, dtype = float)
    quant_bits = (1 << nScaleBits) - 1 + nMantBits
    mantissaVec = mantissaVec.astype(np.uint64)
    sign = (mantissaVec >> (nMantBits - 1)) & 1
    
    # Use the first 4 digits of the number for the mantissa
    num_q = mantissaVec & ~(np.uint64(1) << np.uint64(nMantBits - 1))
    if scale != (1 << nScaleBits) - 1:
        # Do NOT add 1 onto first mantissa bit, instead zero out the sign bit
        num_q = mantissaVec & ~(np.uint64(1) << np.uint64(nMantBits - 1))
        # Shift mantissa bits forward as needed
        shift_amt = quant_bits - (scale + nMantBits)
        num_q = num_q << np.uint64(shift_amt)

    num_q |= sign << (quant_bits - 1)
    
    aNumVec = vDequantizeUniform(num_q, quant_bits)
    ### YOUR CODE ENDS HERE ###

    return aNumVec

#-----------------------------------------------------------------------------

#Testing code
if __name__ == "__main__":

    ### YOUR TESTING CODE STARTS HERE ###

    num = 0
    quantized = QuantizeUniform(num, 2)
    assert quantized == 0
    dequantized = DequantizeUniform(quantized, 2)
    assert dequantized == 0
    
    num = 1
    quantized = QuantizeUniform(num, 2)
    assert quantized == 1
    dequantized = DequantizeUniform(quantized, 2)
    assert dequantized == 2/3
    
    num = -1
    quantized = QuantizeUniform(num, 2)
    assert quantized == 0b11
    dequantized = DequantizeUniform(quantized, 2)
    assert dequantized == -2/3
    
    num = 2
    quantized = QuantizeUniform(num, 2)
    assert quantized == 1
    dequantized = DequantizeUniform(quantized, 2)
    assert dequantized == 2/3
    
    num = -2
    quantized = QuantizeUniform(num, 2)
    assert quantized == 0b11
    dequantized = DequantizeUniform(quantized, 2)
    assert dequantized == -2/3
    
    num = 0.25
    quantized = QuantizeUniform(num, 2)
    assert quantized == 0
    dequantized = DequantizeUniform(quantized, 2)
    assert dequantized == 0

    num = -0.75
    quantized = QuantizeUniform(num, 2)
    assert quantized == 0b11
    dequantized = DequantizeUniform(quantized, 2)
    assert dequantized == -2/3
    
    inputs = np.array([-0.99, -0.38, -0.10, -0.01, -0.001, 0.0, 0.05, 0.28, 0.65, 0.97, 1.0])
    
    vectorized_ins_8 = vQuantizeUniform(inputs, 8)
    vectorized_ins_12 = vQuantizeUniform(inputs, 12)
    
    vectorized_outs_8 = vDequantizeUniform(vectorized_ins_8, 8)
    vectorized_outs_12 = vDequantizeUniform(vectorized_ins_12, 12)
    
    vectorized_bfp_scale = ScaleFactor(1.0)
    vectorized_bfp_mant = vMantissa(inputs, vectorized_bfp_scale)
    vectorized_bfp_outs = vDequantize(vectorized_bfp_scale, vectorized_bfp_mant)
    
    for idx, i in enumerate(inputs):
        # Clamp so that we don't overflow at max
        binary = np.binary_repr(np.clip(round(abs(i) * (1 << 11)), -(1 << 11) + 1, (1 << 11) - 1), width=12)
        if i < 0:
            binary = "1" + binary[1:]
        print(f"-- {i} -- ")
        print(f"12 bit binary repr: {binary}")
        
        q_8 = QuantizeUniform(i, 8)
        assert q_8 == vectorized_ins_8[idx]
        dq_8 = DequantizeUniform(q_8, 8)
        assert dq_8 == vectorized_outs_8[idx]
        
        q_12 = QuantizeUniform(i, 12)
        assert q_12 == vectorized_ins_12[idx]
        dq_12 = DequantizeUniform(q_12, 12)
        assert dq_12 == vectorized_outs_12[idx]
        
        print(f"8 bit midtread: {dq_8:0.3f}")
        print(f"12 bit midtread: {dq_12:0.3f}")
        
        scale_fp = ScaleFactor(i)
        mantissa_fp = MantissaFP(i, scale_fp)
        dq_fp = DequantizeFP(scale_fp, mantissa_fp)
        # print(f"Scale: {scale_fp}, Mantissa: {mantissa_fp}")
        print(f"FP: {dq_fp:0.3f}")
        
        mantissa_bfp = Mantissa(i, scale_fp)
        dq_bfp = Dequantize(scale_fp, mantissa_bfp)
        print(f"Scale: {scale_fp}, Mantissa: {mantissa_fp}")
        print(f"BFP: {dq_bfp:0.3f}")
        
        mantissa_bfp_veccmp = Mantissa(i, vectorized_bfp_scale)
        assert mantissa_bfp_veccmp == vectorized_bfp_mant[idx]
        
        dq_bfp_veccmp = Dequantize(vectorized_bfp_scale, mantissa_bfp_veccmp)
        assert dq_bfp_veccmp == vectorized_bfp_outs[idx]
        
    ### YOUR TESTING CODE ENDS HERE ###

