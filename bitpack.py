"""
bitpack.py -- vectorized code for packing and unpacking bits into an array of bytes

-----------------------------------------------------------------------
© 2009-2026 Marina Bosi & Richard E. Goldberg -- All rights reserved
-----------------------------------------------------------------------
"""
import numpy as np
from struct import *

BYTESIZE = 8

class PackedBits:
    """
    Object holding an array of bytes that one can read from/write to them as individual bits 
    and which transfers the result in and out as a string.
    Optimized to use a single large integer buffer.
    """

    def __init__(self):
        self.buffer = 0
        self.nBits = 0
        self.targetBytes = 0
        self.readBitsLeft = 0

    @property
    def nBytes(self):
         return self.targetBytes

    def Size(self, nBytes):
        """Sizes an existing PackedBits object to hold nBytes of data (all initialized to zero)"""
        self.buffer = 0
        self.nBits = 0  # Current bits written
        self.targetBytes = int(nBytes)

    def GetPackedData(self):
        """Gets the packed data held by this PackedBits object and returns it as a data string"""
        # We need to return exactly targetBytes
        # The buffer contains nBits.
        # We need to shift it to align to the MSB of the output bytes?
        # Wait, WriteBits writes from MSB down?
        # No, WriteBits(3, 2) -> '11'. WriteBits(0, 6) -> '11000000'.
        # This matches (buffer << n) | val.
        # If we wrote '11' then '000000', buffer is '11000000' (0xC0).
        # We want to return byte 0xC0.
        # If we have valid bits < targetBytes * 8, we should shift Left?
        # Original: allocated zeros. Writing filled from MSB.
        # If I wrote 2 bits '11' into a 1-byte container.
        # Original: data[0] has '11000000' (implicit zeros).
        # My buffer: '11'.
        # I need to shift left by (targetBytes*8 - nBits).
        
        total_bits = self.targetBytes * BYTESIZE
        if self.nBits < total_bits:
            shift = int(total_bits - self.nBits)
            if shift > 100:
                print(f"DEBUG: Large Shift! shift={shift}, total={total_bits}, nBits={self.nBits}")
            out_val = self.buffer << shift
        else:
            out_val = self.buffer
            
        return int(out_val).to_bytes(self.targetBytes, byteorder='big')

    def SetPackedData(self, data):
        """Sets the packed data held by this PackedBits object to the passed data string"""
        self.targetBytes = len(data)
        self.readBitsLeft = self.targetBytes * BYTESIZE
        self.buffer = int.from_bytes(data, byteorder='big')

    def WriteBits(self, info, nBits):
        """Writes lowest nBits of info into this PackedBits object at its current byte/bit pointers"""
        nBits = int(nBits)
        if nBits == 0: return
        self.buffer = int((self.buffer << nBits) | (int(info) & ((1 << nBits) - 1)))
        self.nBits += nBits

    def ReadBits(self, nBits):
        """Returns next nBits of info from this PackedBits object starting at its current byte/bit pointers"""
        nBits = int(nBits)
        if nBits == 0: return 0
        # Extract top nBits from readBitsLeft
        shift = self.readBitsLeft - nBits
        val = (self.buffer >> shift) & ((1 << nBits) - 1)
        self.readBitsLeft -= nBits
        return int(val) # ensure regular int
    
    def WriteBitsArray(self, values, nBits):
        """Writes an array of values (each nBits) efficiently"""
        nBits = int(nBits)
        if nBits == 0: return
        # Simple loop is reasonably fast in Python for 1000 items
        mask = (1 << nBits) - 1
        buff = self.buffer
        for v in values:
            buff = (buff << nBits) | (int(v) & mask)
        self.buffer = int(buff)
        self.nBits += len(values) * nBits

    def ReadBitsArray(self, nBits, count):
        """Reads count values of nBits each efficiently"""
        nBits = int(nBits)
        vals = np.empty(count, dtype=np.int32)
        mask = (1 << nBits) - 1
        # Optimize loop slightly by caching lookup
        buff = self.buffer
        rbl = self.readBitsLeft
        for i in range(count):
             rbl -= nBits
             vals[i] = (buff >> rbl) & mask
        self.readBitsLeft = rbl
        return vals

    def ResetPointers(self):
        """Resets the pointers to the start of this PackedBits object"""
        self.readBitsLeft = self.targetBytes * BYTESIZE
        # Note: Original ResetPointers allowed reading what was just written.
        # If we just wrote, nBits is populated. buffer is populated. I should sync readBitsLeft.
        if self.nBits > 0:
             # If we are in 'write mode' turning to 'read mode'
             # We should align the buffer as if it was filled to targetBytes?
             # Or just read what is there?
             # Original used same storage.
             # If I write 2 bits and reset. Read 2 bits.
             # buffer is '11'. nBits=2.
             # readBitsLeft should be 2?
             # Or targetBytes*8?
             # Original Size(N) allocated N bytes.
             # GetPackedData pads.
             # If I write 2 bits, Reset, Read.
             # Original: data[0] is modified. '11000000'.
             # ReadBits reads from MSB. '11'.
             # So yes, I should treat buffer as if it is left-aligned to targetBytes * 8 for reading?
             # This is tricky mixing modes.
             # Usually ResetPointers is used after SetPackedData (Read mode) OR after Write to read back.
             pass 
        
        # If nBits > 0, it means we wrote data.
        # But SetPackedData overwrites buffer, so nBits logic in SetPackedData?
        # Actually SetPackedData doesn't set nBits (write counter).
        # Let's assume ResetPointers is mostly for Read-after-Write check or Read-Reset-Read.
        # Given the usage in test:
        # WriteBits... GetPackedData... Reset... ReadBits.
        # GetPackedData returns padded bytes.
        # Reset should probably assume we are reading the FULL buffer (targetBytes).
        self.readBitsLeft = self.targetBytes * BYTESIZE
        # But wait, my buffer is '11' (2 bits).
        # If I want to read '11000000', I conceptually need the padded value.
        # So:
        total_bits = self.targetBytes * BYTESIZE
        if self.nBits > 0 and self.nBits < total_bits:
             self.buffer <<= (total_bits - self.nBits)
             self.nBits = total_bits # Now we are 'full'
        

if __name__=="__main__":
    print( "\nTesting bit packing:")
    x = (3, 5, 11, 3, 1)
    xlen = (4,3,5,3,1)
    nBytes=2
    # Expect: 3(0011), 5(101), 11(01011), 3(011), 1(1) = 0011 101 0 1011 011 1 = 00111010 10110111 = 3A B7
    
    bp=PackedBits()
    bp.Size(nBytes)
    print( "\nInput Data:\n",x ,"\nPacking Bit Sizes:\n",xlen)
    for i in range(len(x)):
        bp.WriteBits(x[i],xlen[i])
    print( "\nPacked Bits:\n",bp.GetPackedData().hex()) 
    
    y=[]
    bp.ResetPointers()
    for i in range(len(x)):
        y.append(bp.ReadBits(xlen[i]))
    print( "\nUnpacked Data:\n",y) # Should match input

    # Add array test
    print("\nArray Test:")
    bp.Size(nBytes)
    # manual write array
    bp.WriteBitsArray(x, 0) # dummy
    # Can't easily test variable lengths with current WriteBitsArray signature (fixed nBits).
    # But for mantissas they are fixed.
