"""
Context-adaptive range coder for entropy coding blocks of quantized MDCT coefficients.
Switches CDF per ba (bit allocation) group so each group gets a codebook sized to its
exact magnitude range, eliminating wasted probability mass.
"""
import numpy as np
from bitpack import *

# 8 zero-concentration levels: fraction of total probability assigned to symbol 0.
# Remaining mass is distributed geometrically across symbols 1..cb_size-1.
# Each ba-group trial-encodes with all 8 and picks the one giving fewest bits.
ZERO_CONC = [0.97, 0.90, 0.80, 0.65, 0.50, 0.35, 0.20, 0.05]
N_DIST_OPTIONS = len(ZERO_CONC)


class RangeCoder:
    """Core arithmetic coder with adaptive (per-symbol CDF) encode/decode."""

    def __init__(self, block_order: int):
        self.block_order = block_order
        self.block_size = 1 << block_order
        # Pre-compute CDFs for every (ba, dist_idx) pair.
        # ba=k → magnitude range 0..2^(k-1)-1 → codebook_size = 2^(k-1)
        # ba=1 → codebook_size=1 (magnitude always 0) — skip in range coding
        self._cdf_cache = {}  # (ba, dist_idx) → (cdf, total_freq)
        for ba in range(2, 17):  # ba 2..16 covers all practical allocations
            cb_size = 1 << (ba - 1)
            for d_idx in range(N_DIST_OPTIONS):
                cdf, total = self._make_cdf(cb_size, ZERO_CONC[d_idx])
                self._cdf_cache[(ba, d_idx)] = (cdf, total)

    def _make_cdf(self, codebook_size, zero_frac):
        """Build a Laplacian CDF with `codebook_size` symbols, scaled so symbol 0
        gets approximately `zero_frac` of the total probability mass.
        Uses binary search to find the Laplacian b parameter that achieves target zero_frac."""
        total_freq = max(self.block_size, codebook_size * 2)

        if codebook_size == 1:
            cdf = np.array([0, total_freq], dtype=np.uint64)
            return cdf, total_freq
        if codebook_size == 2:
            c0 = max(1, int(round(zero_frac * total_freq)))
            c1 = max(1, total_freq - c0)
            cdf = np.array([0, c0, c0 + c1], dtype=np.uint64)
            return cdf, c0 + c1

        # Binary search for Laplacian b that gives ~zero_frac mass at symbol 0
        xs = np.arange(codebook_size, dtype=np.float64)
        b_lo, b_hi = 0.001, 100.0
        for _ in range(50):
            b_mid = (b_lo + b_hi) / 2
            dist = np.exp(-xs / b_mid)
            frac0 = dist[0] / np.sum(dist)
            if frac0 > zero_frac:
                b_lo = b_mid
            else:
                b_hi = b_mid
        b = (b_lo + b_hi) / 2
        dist = np.exp(-xs / b)
        dist_sum = np.sum(dist)
        counts = np.maximum(1, np.round(dist / dist_sum * total_freq).astype(np.int64))
        # Adjust symbol 0 to hit exact total
        diff = total_freq - int(np.sum(counts))
        counts[0] = max(1, counts[0] + diff)
        actual_total = int(np.sum(counts))

        cdf = np.zeros(codebook_size + 1, dtype=np.uint64)
        cdf[1:] = np.cumsum(counts)
        return cdf, actual_total

    def encode_adaptive(self, symbols, cdf_sequence):
        """Encode a sequence of symbols where each symbol has its own (cdf, total_freq, cb_size).
        Returns a PackedBits with the range-coded stream."""
        pb = PackedBits()
        pb.Size(len(symbols) * 4)  # generous upper bound
        low = 0
        rng = 0x100000000
        cache = -1
        ffNum = 0

        def shift_out_byte(byte):
            nonlocal cache, ffNum
            if byte < 0xFF:
                if cache >= 0:
                    pb.WriteBits(cache, 8)
                for _ in range(ffNum):
                    pb.WriteBits(0xFF, 8)
                cache = byte
                ffNum = 0
            elif byte == 0xFF:
                ffNum += 1
            else:
                if cache >= 0:
                    pb.WriteBits(cache + 1, 8)
                for _ in range(ffNum):
                    pb.WriteBits(0x00, 8)
                cache = byte & 0xFF
                ffNum = 0

        for i, sym in enumerate(symbols):
            cdf, total_freq, cb_size = cdf_sequence[i]
            sym = min(int(sym), cb_size - 1)
            step = rng // total_freq
            low += int(cdf[sym]) * step
            rng = int(cdf[sym + 1] - cdf[sym]) * step
            while rng < 0x01000000:
                shift_out_byte(low >> 24)
                low = (low << 8) & 0xFFFFFFFF
                rng <<= 8

        for _ in range(4):
            shift_out_byte(low >> 24)
            low = (low << 8) & 0xFFFFFFFF
        if cache >= 0:
            pb.WriteBits(cache, 8)
        for _ in range(ffNum):
            pb.WriteBits(0xFF, 8)
        return pb

    def decode_adaptive(self, pb, n_values, cdf_sequence):
        """Decode n_values symbols from pb, each with its own (cdf, total_freq, cb_size)."""
        low = 0
        rng = 0x100000000
        code = 0
        for _ in range(4):
            code = (code << 8) | pb.ReadBits(8)

        magnitudes = np.empty(n_values, dtype=np.int64)
        for i in range(n_values):
            cdf, total_freq, cb_size = cdf_sequence[i]
            step = rng // total_freq
            if step == 0:
                magnitudes[i] = 0
                continue
            scaled = ((code - low) & 0xFFFFFFFF) // step
            symbol = int(np.searchsorted(cdf[1:], scaled, side='right'))
            symbol = min(symbol, cb_size - 1)
            low = (low + int(cdf[symbol]) * step) & 0xFFFFFFFF
            rng = int(cdf[symbol + 1] - cdf[symbol]) * step
            magnitudes[i] = symbol
            while rng < 0x01000000:
                low = (low << 8) & 0xFFFFFFFF
                rng <<= 8
                code = ((code << 8) | pb.ReadBits(8)) & 0xFFFFFFFF

        return magnitudes


def _ba_groups(bitAlloc, sfBands):
    """Group bands by ba value. Returns sorted list of (ba, [(iBand, nLines), ...]).
    Deterministic ordering: sorted by ba, then by iBand within each group.
    Same ba → same codebook size → single CDF per group, which is why grouping helps."""
    groups = {}
    for iBand in range(sfBands.nBands):
        ba = int(bitAlloc[iBand])
        if ba == 0:
            continue
        if ba not in groups:
            groups[ba] = []
        groups[ba].append((iBand, int(sfBands.nLines[iBand])))
    return sorted(groups.items())


class BlockEntropyCoder:
    N_DIST_BITS = 3  # ceil(log2(N_DIST_OPTIONS)) = 3

    def __init__(self, block_order: int, codebook_order: int = 8):
        # codebook_order kept in signature for compatibility but ignored
        self.coder = RangeCoder(block_order)

    def encode_block(
        self,
        mantissas: np.ndarray,
        bitAlloc: np.ndarray,
        sfBands,
    ) -> PackedBits:
        signs, magnitudes = self._split_mants(mantissas, bitAlloc, sfBands)
        groups = _ba_groups(bitAlloc, sfBands)

        # Build per-group magnitudes in ba-group order and pick best Laplacian per group
        group_mags = []   # list of np.array per group
        group_b_idx = []  # best Laplacian index per group
        mag_offset = 0
        # Map from band order to ba-group order
        band_to_group_offset = {}
        iMant = 0
        for iBand in range(sfBands.nBands):
            nLines = sfBands.nLines[iBand]
            if bitAlloc[iBand]:
                band_to_group_offset[iBand] = iMant
                iMant += nLines

        for ba, bands in groups:
            if ba == 1:
                # ba=1: magnitudes are always 0, no range coding needed
                n_lines = sum(nl for _, nl in bands)
                group_mags.append(np.zeros(n_lines, dtype=np.uint32))
                group_b_idx.append(0)  # placeholder, won't be written
                continue

            # Collect magnitudes for this group in band order
            g_mags = []
            for iBand, nLines in bands:
                offset = band_to_group_offset[iBand]
                g_mags.append(magnitudes[offset:offset + nLines])
            g_mags = np.concatenate(g_mags)
            group_mags.append(g_mags)

            # Brute-force best distribution for this group
            best_idx = 0
            best_bits = None
            cb_size = 1 << (ba - 1)
            for d_idx in range(N_DIST_OPTIONS):
                cdf_entry = self.coder._cdf_cache.get((ba, d_idx))
                if cdf_entry is None:
                    continue
                cdf, total_freq = cdf_entry
                seq = [(cdf, total_freq, cb_size)] * len(g_mags)
                cpb = self.coder.encode_adaptive(g_mags, seq)
                if best_bits is None or cpb.nBits < best_bits:
                    best_bits = cpb.nBits
                    best_idx = d_idx
            group_b_idx.append(best_idx)

        # Build flat CDF sequence and symbol list in ba-group order (skip ba=1)
        # Output bitstream: [3b × n_active_groups | 1b × n_alloc signs | range stream]
        all_symbols = []
        cdf_sequence = []
        for (ba, bands), g_mags, b_idx in zip(groups, group_mags, group_b_idx):
            if ba == 1:
                continue
            cdf, total_freq = self.coder._cdf_cache[(ba, b_idx)]
            cb_size = 1 << (ba - 1)
            entry = (cdf, total_freq, cb_size)
            for s in g_mags:
                all_symbols.append(int(s))
                cdf_sequence.append(entry)

        # Encode the range-coded stream
        if all_symbols:
            coded_pb = self.coder.encode_adaptive(all_symbols, cdf_sequence)
        else:
            coded_pb = PackedBits()
            coded_pb.Size(0)

        # Reorder signs to ba-group order
        signs_grouped = []
        for ba, bands in groups:
            for iBand, nLines in bands:
                offset = band_to_group_offset[iBand]
                signs_grouped.extend(signs[offset:offset + nLines])
        signs_grouped = np.array(signs_grouped, dtype=np.uint32)

        # Count active ba groups (ba > 1) for Laplacian index overhead
        active_groups = [(ba, b_idx) for (ba, _), b_idx in zip(groups, group_b_idx) if ba > 1]
        n_lap_bits = self.N_DIST_BITS * len(active_groups)
        n_total_bits = n_lap_bits + len(signs_grouped) + coded_pb.nBits
        n_bytes = (n_total_bits + 7) // 8

        pb = PackedBits()
        pb.Size(n_bytes)
        # Write Laplacian indices for active groups
        for ba, b_idx in active_groups:
            pb.WriteBits(b_idx, self.N_DIST_BITS)
        # Write signs in ba-group order
        pb.WriteBitsArray(signs_grouped, 1)
        # Write range-coded stream
        if coded_pb.nBits > 0:
            pb.WriteBits(coded_pb.buffer, coded_pb.nBits)

        return pb

    def decode_block(
        self,
        pb: PackedBits,
        bitAlloc: np.ndarray,
        sfBands,
        nMDCTLines: int,
    ) -> np.ndarray:
        groups = _ba_groups(bitAlloc, sfBands)

        # Read per-group Laplacian indices (skip ba=1)
        group_b_idx = []
        for ba, bands in groups:
            if ba == 1:
                group_b_idx.append(0)
            else:
                group_b_idx.append(pb.ReadBits(self.N_DIST_BITS))

        # Total allocated lines
        n_alloc = sum(nLines for ba, bands in groups for _, nLines in bands)

        # Read signs in ba-group order
        signs_grouped = pb.ReadBitsArray(1, n_alloc).astype(np.int64)

        # Build CDF sequence for range decoding (skip ba=1)
        cdf_sequence = []
        n_range_symbols = 0
        group_n_lines = []  # lines per group for ba>1
        for (ba, bands), b_idx in zip(groups, group_b_idx):
            n_lines = sum(nl for _, nl in bands)
            if ba == 1:
                group_n_lines.append(0)
                continue
            group_n_lines.append(n_lines)
            cdf, total_freq = self.coder._cdf_cache[(ba, b_idx)]
            cb_size = 1 << (ba - 1)
            entry = (cdf, total_freq, cb_size)
            for _ in range(n_lines):
                cdf_sequence.append(entry)
                n_range_symbols += 1

        # Decode range-coded magnitudes
        if n_range_symbols > 0:
            range_mags = self.coder.decode_adaptive(pb, n_range_symbols, cdf_sequence)
        else:
            range_mags = np.array([], dtype=np.int64)

        # Reconstruct full magnitude array in ba-group order
        mags_grouped = np.empty(n_alloc, dtype=np.int64)
        group_offset = 0
        range_offset = 0
        for (ba, bands), n_lines in zip(groups, group_n_lines):
            total_lines = sum(nl for _, nl in bands)
            if ba == 1:
                mags_grouped[group_offset:group_offset + total_lines] = 0
            else:
                mags_grouped[group_offset:group_offset + total_lines] = \
                    range_mags[range_offset:range_offset + n_lines]
                range_offset += n_lines
            group_offset += total_lines

        # Un-shuffle: map ba-group order back to band order
        magnitudes = np.empty(n_alloc, dtype=np.int64)
        signs = np.empty(n_alloc, dtype=np.int64)
        # Build mapping: for each band in group order, what's its position in band order?
        band_order_offset = {}
        offset = 0
        for iBand in range(sfBands.nBands):
            if bitAlloc[iBand]:
                band_order_offset[iBand] = offset
                offset += int(sfBands.nLines[iBand])

        group_src = 0
        for ba, bands in groups:
            for iBand, nLines in bands:
                dst = band_order_offset[iBand]
                magnitudes[dst:dst + nLines] = mags_grouped[group_src:group_src + nLines]
                signs[dst:dst + nLines] = signs_grouped[group_src:group_src + nLines]
                group_src += nLines

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
    coder = RangeCoder(block_order)
    print("CDF cache entries:", len(coder._cdf_cache))
    for d_idx, zc in enumerate(ZERO_CONC):
        cdf, total = coder._cdf_cache[(4, d_idx)]
        print(f"  ba=4 zc={zc:.0%}: codebook_size=8, total_freq={total}, cdf[1]={cdf[1]}")
    # Verify large codebooks don't break
    for ba in [14, 15, 16]:
        cdf, total = coder._cdf_cache[(ba, 0)]
        cb = 1 << (ba - 1)
        assert np.all(np.diff(cdf) >= 1), f"ba={ba}: CDF not monotonic!"
        print(f"  ba={ba}: cb_size={cb}, total_freq={total} OK")
