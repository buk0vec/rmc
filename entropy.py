"""
A simple range coder for performing entropy coding on blocks of quantized MDCT coefficients
Supports encoding a range of values as well as using exp-Golomb coding for values outside
of that range
All values treated as unsigned.
"""
import numpy as np
from bitpack import *

LAPLACIAN_MAP = [0.125, 0.25, 0.5, 0.75, 1, 2, 4, 10]


class ExpGolombCoder:
    @staticmethod
    def write(pb: PackedBits, k: int) -> None:
        """Write exp-Golomb code for k >= 0. k=0 → '1' (1 bit)."""
        x = int(k) + 1
        n = x.bit_length()
        for _ in range(n - 1):
            pb.WriteBits(0, 1)
        pb.WriteBits(x, n)

    @staticmethod
    def read(pb: PackedBits) -> int:
        """Read exp-Golomb code, return k >= 0."""
        leading = 0
        while pb.ReadBits(1) == 0:
            leading += 1
        rem = pb.ReadBits(leading)
        return (1 << leading) + rem - 1


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

    def _cdf(self, codebook):
        cdf = np.zeros(self.codebook_size + 1, dtype=np.uint64)
        cdf[1:] = np.cumsum(codebook)
        return cdf

    def _normalized_laplacian(self, b):
        xs = np.arange(self.codebook_size)
        dist = 1/b * np.exp(-xs/b)
        total = np.sum(dist)
        dist_norm = np.round(dist/total * self.block_size)
        dist_norm = np.where(dist_norm == 0, 1, dist_norm)
        diff = self.block_size - np.sum(dist_norm)
        dist_norm[0] += diff

        return dist_norm

    def encode(self, magnitudes: np.ndarray, b: float) -> tuple:
        """Encode unsigned magnitudes using Laplacian b.
        Returns (PackedBits of range-coded stream, list of raw escaped magnitude values)."""
        pb = PackedBits()
        pb.Size(len(magnitudes) * 4)  # generous upper bound
        codebook = self._normalized_laplacian(b)
        cdf = self._cdf(codebook)
        total_freq = int(cdf[-1])
        low = 0
        rng = 0x100000000
        escaped = []

        cache = -1
        ffNum = 0
        
        def shift_out_byte(b):
            nonlocal cache, ffNum
            if b < 0xFF:
                if cache >= 0:
                    pb.WriteBits(cache, 8)
                for _ in range(ffNum):
                    pb.WriteBits(0xFF, 8)
                cache = b
                ffNum = 0
            elif b == 0xFF:
                ffNum += 1
            else:
                # Carry!
                if cache >= 0:
                    pb.WriteBits(cache + 1, 8)
                for _ in range(ffNum):
                    pb.WriteBits(0x00, 8)
                cache = b & 0xFF
                ffNum = 0

        for idx_enc, value in enumerate(magnitudes):
            value = int(value)
            if value <= self.max_value:
                idx = value
            else:
                idx = self.escape_idx
                escaped.append(value)

            step = rng // total_freq
            low += int(cdf[idx]) * step
            rng = int(cdf[idx + 1] - cdf[idx]) * step
            
            while rng < 0x01000000:
                shift_out_byte(low >> 24)
                low = (low << 8) & 0xFFFFFFFF
                rng <<= 8

        # Flush 4 bytes
        for _ in range(4):
            shift_out_byte(low >> 24)
            low = (low << 8) & 0xFFFFFFFF
            
        if cache >= 0:
            pb.WriteBits(cache, 8)
        for _ in range(ffNum):
            pb.WriteBits(0xFF, 8)

        return pb, escaped

    def decode(self, pb: PackedBits, n_values: int, b: float) -> tuple:
        """Read range-coded stream from pb.
        Returns (magnitudes_with_escape_placeholders, n_escaped).
        Escape placeholders are left as self.escape_idx; caller substitutes real values."""
        codebook = self._normalized_laplacian(b)
        cdf = self._cdf(codebook)
        total_freq = int(cdf[-1])

        low = 0
        rng = 0x100000000
        code = 0
        trace = []
        for _ in range(4):
            byte = pb.ReadBits(8)
            trace.append(byte)
            code = (code << 8) | byte

        magnitudes = np.empty(n_values, dtype=np.int64)
        n_escaped = 0

        bits_read = 32  # 4 init bytes

        for i in range(n_values):
            step = rng // total_freq
            if step == 0:
                break
            scaled = ((code - low) & 0xFFFFFFFF) // step
            symbol = int(np.searchsorted(cdf[1:], scaled, side='right'))
            symbol = min(symbol, self.codebook_size - 1)

            freq = int(cdf[symbol + 1] - cdf[symbol])

            low = (low + int(cdf[symbol]) * step) & 0xFFFFFFFF
            rng = freq * step
            magnitudes[i] = symbol

            if symbol == self.escape_idx:
                n_escaped += 1

            while rng < 0x01000000:
                low = (low << 8) & 0xFFFFFFFF
                rng <<= 8
                byte = pb.ReadBits(8)
                code = ((code << 8) | byte) & 0xFFFFFFFF
                bits_read += 8

        return magnitudes, n_escaped


class BlockEntropyCoder:
    N_LAP_BITS = 3  # ceil(log2(len(LAPLACIAN_MAP))) = 3

    def __init__(self, block_order: int, codebook_order: int = 5):
        self.coder = RangeCoder(block_order, codebook_order)

    def encode_block(
        self,
        mantissas: np.ndarray,
        bitAlloc: np.ndarray,
        sfBands,
    ) -> PackedBits:
        signs, magnitudes = self._split_mants(mantissas, bitAlloc, sfBands)

        # Select best Laplacian by brute-forcing encode() for every b,
        # comparing coded_pb.nBits (signs + escaped overhead is identical for all b)
        best_idx = 0
        best_pb = None
        best_escaped = None
        for i, b in enumerate(LAPLACIAN_MAP):
            candidate_pb, candidate_escaped = self.coder.encode(magnitudes, b)
            if best_pb is None or candidate_pb.nBits < best_pb.nBits:
                best_idx, best_pb, best_escaped = i, candidate_pb, candidate_escaped
        coded_pb, escaped = best_pb, best_escaped
        print(f"Used distribution {best_idx}")
        escaped_offsets = [v - self.coder.max_value - 1 for v in escaped]

        # Compute escaped bits for sizing
        eg_bits = sum(2 * (k + 1).bit_length() - 1 for k in escaped_offsets) if escaped_offsets else 0
        n_total_bits = self.N_LAP_BITS + len(signs) + coded_pb.nBits + eg_bits
        n_bytes = (n_total_bits + 7) // 8


        pb = PackedBits()
        pb.Size(n_bytes)
        pb.WriteBits(best_idx, self.N_LAP_BITS)
        pb.WriteBitsArray(signs, 1)
        pb.WriteBits(coded_pb.buffer, coded_pb.nBits)
        for k in escaped_offsets:
            ExpGolombCoder.write(pb, k)

        return pb

    def decode_block(
        self,
        pb: PackedBits,
        bitAlloc: np.ndarray,
        sfBands,
        nMDCTLines: int,
    ) -> np.ndarray:
        b_idx = pb.ReadBits(self.N_LAP_BITS)
        b = LAPLACIAN_MAP[b_idx]

        n_alloc = sum(
            sfBands.nLines[iBand]
            for iBand in range(sfBands.nBands)
            if bitAlloc[iBand]
        )

        signs = pb.ReadBitsArray(1, n_alloc).astype(np.int64)
        magnitudes, n_escaped = self.coder.decode(pb, n_alloc, b)

        # Substitute escaped values
        escape_iter = (
            ExpGolombCoder.read(pb) + self.coder.max_value + 1
            for _ in range(n_escaped)
        )
        for i in range(len(magnitudes)):
            if magnitudes[i] == self.coder.escape_idx:
                magnitudes[i] = next(escape_iter)

        return self._combine_mants(magnitudes, signs, bitAlloc, sfBands, nMDCTLines)

    # --- helpers ---

    def _split_mants(self, mantissas, bitAlloc, sfBands):
        """Separate folded-binary mantissas into sign bits and unsigned magnitudes."""
        signs = []
        magnitudes = []
        iMant = 0
        for iBand in range(sfBands.nBands):
            nLines = sfBands.nLines[iBand]
            if bitAlloc[iBand]:
                ba = bitAlloc[iBand]
                sign_pos = ba - 1
                band_m = mantissas[iMant:iMant + nLines]
                signs.extend(int(m) >> sign_pos for m in band_m)
                magnitudes.extend(int(m) & ((1 << sign_pos) - 1) for m in band_m)
                iMant += nLines
        return np.array(signs, dtype=np.uint32), np.array(magnitudes, dtype=np.uint32)

    def _combine_mants(self, magnitudes, signs, bitAlloc, sfBands, nMDCTLines):
        """Recombine signs and magnitudes into full-length mantissa array (nMDCTLines)."""
        mantissa = np.zeros(nMDCTLines, dtype=np.int32)
        iSrc = 0
        for iBand in range(sfBands.nBands):
            nLines = sfBands.nLines[iBand]
            if bitAlloc[iBand]:
                ba = bitAlloc[iBand]
                sign_pos = ba - 1
                band_mags = magnitudes[iSrc:iSrc + nLines]
                band_signs = signs[iSrc:iSrc + nLines]
                mantissa[sfBands.lowerLine[iBand]:sfBands.upperLine[iBand] + 1] = (band_signs << sign_pos) | band_mags
                iSrc += nLines
        return mantissa


if __name__ == "__main__":
    block_order = 14
    coder = RangeCoder(block_order, 5)
    laplacian = coder._normalized_laplacian(1)
    assert np.sum(laplacian) == 1 << block_order
    for l in LAPLACIAN_MAP:
        print(coder._normalized_laplacian(l))
