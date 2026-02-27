"""
A simple range coder for performing entropy coding on blocks of quantized MDCT coefficients
Supports encoding a range of values as well as using exp-Golomb coding for values outside
of that range
All values treated as unsigned.
"""
import numpy as np
from bitpack import *

class RangeCoder:
    def __init__(self, block_order: int, codebook_order: int = 4):
        assert codebook_order > 0
        assert block_order > 0
        
        self.codebook = None
        self.block_order = block_order
        self.block_size = 2 ** block_order
        self.codebook_order = codebook_order
        self.codebook_size = 2 ** codebook_order
        self.max_value = self.codebook_size - 2
        self.escape_idx = self.codebook_size - 1
        self.total_bytes = (self.codebook_size * self.block_order + 7) // 8
    
    def codebook_from_values(self, values):
        self.codebook = np.zeros(self.codebook_size, dtype=np.uint64)
        for value in values:
            if value <= self.max_value:
                self.codebook[value] += 1
            else:
                self.codebook[self.escape_idx] += 1
    
    def codebook_from_packed(self, pb: PackedBits):
        self.codebook = pb.ReadBitsArray(self.block_order, self.codebook_size)
        
    def pack_codebook(self) -> PackedBits:
        pb = PackedBits()
        pb.Size(self.total_bytes)
        pb.WriteBitsArray(self.codebook, self.block_order)
        return pb

    def _cdf(self):
        cdf = np.zeros(self.codebook_size + 1, dtype=np.uint64)
        cdf[1:] = np.cumsum(self.codebook)
        return cdf
    
    def encode(self, values) -> PackedBits:
        pb = PackedBits()
        pb.Size(self.block_size * 2) 
        cdf = self._get_cdf()
        total_freq = int(cdf[-1])
        low = 0
        high = 0xFFFFFFFF
        escaped = []
        for value in values:
            if value <= self.max_value:
                # Encode as normal
                idx = int(value)
            else:
                # Encode as escaped
                idx = self.escape_idx
                escaped.append(values)
        
            step = (high - low + 1) // total_freq
            high = low + (int(cdf[idx + 1]) * step) - 1
            low = low + (int(cdf[idx]) * step)
            
            # Renormalize
            while (low >> 24) == (high >> 24):
                pb.WriteBits(low >> 24, 8)
                low = (low << 8) & 0xFFFFFFFF
                high = ((high << 8) | 0xFF) & 0xFFFFFFFF
                
        # Flush to pb
        for _ in range(4):
            pb.WriteBits((low >> 24) & 0xFF, 8)
            low = (low << 8) & 0xFFFFFFFF
        
        # Write out escaped values to pb
        for value in escaped:
            self._write_exp_golomb(pb, value)
                
    def _write_exp_golomb(self, pb, k):
        x = int(k) + 1
        num_bits = x.bit_length() 
        for _ in range(num_bits - 1):
            pb.WriteBits(0, 1)
        
        pb.WriteBits(x, num_bits)

    def _read_exp_golomb(self, pb):
        leading_zeros = 0
        while pb.ReadBits(1) == 0:
            leading_zeros += 1
        
        rem = pb.ReadBits(leading_zeros)
        return (1 << leading_zeros) + rem - 1
    
    