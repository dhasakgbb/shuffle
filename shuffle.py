#!/usr/bin/env python3
"""
shuffle_dashboard_with_metrics.py

High-performance shuffle+detect dashboard with additional mixing metrics:
  • Fused single‑pass stats & initial‑deck detection
  • Table‑driven poker via device‑memory LUT
  • Warp‑level reductions for hist/poker
  • GPU kernel also collects inversion & run‑length histograms
  • Asynchronous compute‑UI pipeline with lockless ring buffer

Press Ctrl‑C to quit.
"""

import os
import numpy as np
import numba
from numba import cuda, njit
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float64
from numba.cuda.cudadrv.driver import CudaAPIError
import multiprocessing as mp
from time import perf_counter, sleep
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from colorsys import hsv_to_rgb
from dataclasses import dataclass
from ring_buffer import MetricsBuffer
import math

# ─────────── Configuration Constants ──────────────────────────────────────
# Core configuration
DECK_SIZE = 52
HIST_BINS = numba.int32(101)  # Mark as Numba constant
POKER_CAT_BITS = 4
POKER_CAT_MASK = 0xF
POKER_CATS_PER_U32 = 8  # 32 bits / 4 bits per category
HIST_WIDTH = 35
INV_MAX = DECK_SIZE * (DECK_SIZE-1) // 2  # 1326 possible inversions
RUN_BINS = DECK_SIZE  # run lengths 1..52
INV_BUCKETS = numba.int32(64)  # Mark as Numba constant
INV_BUCKETS = 64  # Number of buckets for inversion histogram
RUN_BUCKETS = 16  # Number of buckets for run length histogram

# Performance tuning
UI_FPS = 60
CPU_CHUNK_SIZE = 50_000
GPU_SHUFFLES_THR = 5000  # ~128×256×5k ≈ 163M shuffles per kernel launch
WARP_SIZE = 32
INV_SAMPLE_SIZE = 1000  # Sample 1000 pairs for higher precision
CPU_WORKERS = max(1, mp.cpu_count() - 1)

# Poker hand masks
ROYAL_MASK = (1<<0)|(1<<9)|(1<<10)|(1<<11)|(1<<12)
CONSEC5 = 0b11111

# Xoshiro256++ constants
XOSHIRO256_MULT = np.uint64(0x82A2B175229D6A5B)
XOSHIRO256_ROT1 = np.uint64(17)
XOSHIRO256_ROT2 = np.uint64(45)
XOSHIRO256_ROT3 = np.uint64(23)

# Precompute bar color
_r,_g,_b = hsv_to_rgb(0.66,1,1)
BAR_COLOR = f"rgb({int(_r*255)},{int(_g*255)},{int(_b*255)})"

@dataclass(frozen=True)
class Config:
    # Core configuration
    DECK_SIZE: int = DECK_SIZE
    HIST_BINS: int = HIST_BINS
    POKER_CAT_BITS: int = POKER_CAT_BITS
    POKER_CAT_MASK: int = POKER_CAT_MASK
    POKER_CATS_PER_U32: int = POKER_CATS_PER_U32
    HIST_WIDTH: int = HIST_WIDTH
    INV_MAX: int = INV_MAX
    RUN_BINS: int = RUN_BINS
    
    # Performance tuning
    UI_FPS: int = UI_FPS
    CPU_CHUNK_SIZE: int = CPU_CHUNK_SIZE
    GPU_SHUFFLES_THR: int = GPU_SHUFFLES_THR
    WARP_SIZE: int = WARP_SIZE
    INV_SAMPLE_SIZE: int = INV_SAMPLE_SIZE
    CPU_WORKERS: int = CPU_WORKERS
    
    # Poker hand masks
    ROYAL_MASK: int = ROYAL_MASK
    CONSEC5: int = CONSEC5
    
    # Xoshiro256++ constants
    XOSHIRO256_MULT: np.uint64 = XOSHIRO256_MULT
    XOSHIRO256_ROT1: np.uint64 = XOSHIRO256_ROT1
    XOSHIRO256_ROT2: np.uint64 = XOSHIRO256_ROT2
    XOSHIRO256_ROT3: np.uint64 = XOSHIRO256_ROT3

cfg = Config()

# ─────────── Utility Functions ────────────────────────────────────────────
@njit(nogil=True, fastmath=True, cache=True)
def xoshiro256pp_next(state):
    """Generate next random number using Xoshiro256++."""
    result = np.uint64(0)
    t = state[1] << np.uint64(17)
    
    state[2] ^= state[0]
    state[3] ^= state[1]
    state[1] ^= state[2]
    state[0] ^= state[3]
    
    state[2] ^= t
    state[3] = (state[3] << np.uint64(45)) | (state[3] >> np.uint64(19))
    
    result = (state[0] + state[3]) >> np.uint64(23)
    result ^= state[0] + state[3]
    
    return result

@njit(nogil=True, fastmath=True, cache=True)
def xoshiro256pp_jump(state):
    """Jump function for Xoshiro256++."""
    jump = np.array([0x180ec6d33cfd0aba, 0xd5a61266f0c9392c,
                     0xa9582618e03fc9aa, 0x39abdc4529b1661c], dtype=np.uint64)
    
    s0 = np.uint64(0)
    s1 = np.uint64(0)
    s2 = np.uint64(0)
    s3 = np.uint64(0)
    
    for i in range(4):
        for b in range(64):
            if jump[i] & (np.uint64(1) << np.uint64(b)):
                s0 ^= state[0]
                s1 ^= state[1]
                s2 ^= state[2]
                s3 ^= state[3]
            xoshiro256pp_next(state)
    
    state[0] = s0
    state[1] = s1
    state[2] = s2
    state[3] = s3

def init_rng_state():
    """Initialize a Xoshiro256++ state."""
    state = np.zeros(4, dtype=np.uint64)
    for i in range(4):
        state[i] = int.from_bytes(os.urandom(8), 'little')
    return state

# ─────────── Poker Hand Categorization ────────────────────────────────────
def build_poker_lut():
    """Build the poker hand lookup table using efficient vectorized operations."""
    # Create arrays for all possible rank masks
    masks = np.arange(1<<13, dtype=np.uint16)
    
    # Count bits in each mask
    bit_counts = np.array([m.bit_count() for m in masks])
    valid_masks = masks[bit_counts == 5]
    
    # Initialize LUTs
    rank_lut = np.zeros(1<<13, dtype=np.uint8)
    flush_lut = np.zeros(1<<13, dtype=np.uint8)
    
    # Check for straights and royal flushes
    straight_masks = np.array([CONSEC5 << i for i in range(9)], dtype=np.uint16)
    is_straight = np.any((valid_masks[:, None] & straight_masks) == straight_masks, axis=1)
    is_royal = (valid_masks & ROYAL_MASK) == ROYAL_MASK
    
    # Determine flush categories (4 bits)
    cat_flush = np.where(is_royal, 8, np.where(is_straight, 8, 5))
    flush_lut[valid_masks] = cat_flush
    
    # Determine non-flush categories (4 bits)
    cat_non_flush = np.where(is_royal | is_straight, 4, 0)  # Start with straights
    rank_lut[valid_masks] = cat_non_flush
    
    # Process remaining categories (pairs, trips, quads)
    for mask in valid_masks:
        # Count pairs, trips, quads
        pairs = 0
        trips = 0
        quads = 0
        for j in range(13):
            if (mask >> j) & 1:
                if (mask >> (j+1)) & 1:
                    pairs += 1
                    if (mask >> (j+2)) & 1:
                        trips += 1
                        if (mask >> (j+3)) & 1:
                            quads += 1
        
        # Update category if not a straight
        if not (is_royal[mask == valid_masks][0] or is_straight[mask == valid_masks][0]):
            if quads:
                rank_lut[mask] = 7  # Four of a kind
            elif trips and pairs:
                rank_lut[mask] = 6  # Full house
            elif trips:
                rank_lut[mask] = 3  # Three of a kind
            elif pairs == 2:
                rank_lut[mask] = 2  # Two pair
            elif pairs == 1:
                rank_lut[mask] = 1  # One pair
    
    # Pack LUTs into 32-bit words (8 categories per word)
    packed_size = (1<<14 + POKER_CATS_PER_U32 - 1) // POKER_CATS_PER_U32
    packed_lut = np.zeros(packed_size, dtype=np.uint32)
    
    # Pack rank LUT (first 13 bits)
    for i in range(1<<13):
        word_idx = i // POKER_CATS_PER_U32
        bit_shift = (i % POKER_CATS_PER_U32) * POKER_CAT_BITS
        packed_lut[word_idx] |= (rank_lut[i] & POKER_CAT_MASK) << bit_shift
    
    # Pack flush LUT (next 13 bits)
    for i in range(1<<13):
        word_idx = (i + (1<<13)) // POKER_CATS_PER_U32
        bit_shift = ((i + (1<<13)) % POKER_CATS_PER_U32) * POKER_CAT_BITS
        packed_lut[word_idx] |= (flush_lut[i] & POKER_CAT_MASK) << bit_shift
    
    return packed_lut

# Build the packed poker LUT once at module load
lut_packed = build_poker_lut()

# Precompute triangular mapping arrays
pre_i = np.zeros(INV_MAX, dtype=np.uint8)
pre_j = np.zeros(INV_MAX, dtype=np.uint8)
idx = 0
for i in range(DECK_SIZE):
    for j in range(i + 1, DECK_SIZE):
        pre_i[idx] = i
        pre_j[idx] = j
        idx += 1

# Initialize device arrays
initial_d = cuda.to_device(np.arange(DECK_SIZE, dtype=np.int8))
lut_d = cuda.to_device(lut_packed)
const_indices_i = cuda.to_device(pre_i)
const_indices_j = cuda.to_device(pre_j)

@cuda.jit(device=True, inline=True)
def get_poker_cat(mask, flush):
    """Get poker category from packed LUT using vectorized loads."""
    # Calculate index in packed LUT
    idx = mask | (flush << 13)
    word_idx = idx // POKER_CATS_PER_U32
    bit_shift = (idx % POKER_CATS_PER_U32) * POKER_CAT_BITS
    
    # Load packed word and extract category
    packed_word = cuda.ldg(lut_d, word_idx)
    return (packed_word >> bit_shift) & POKER_CAT_MASK

@cuda.jit(device=True, inline=True)
def compute_run_lengths(deck, block_run_hist):
    """Compute run lengths and update histogram."""
    run_len = 1
    prev = deck[0]
    
    for i in range(1, DECK_SIZE):
        if deck[i] > prev:
            run_len += 1
        else:
            # Bucket (length-1) to match CPU code
            bucket = min(RUN_BUCKETS-1, (run_len-1) // (DECK_SIZE // RUN_BUCKETS))
            cuda.atomic.add(block_run_hist, bucket, 1)
            run_len = 1
        prev = deck[i]
    
    # Handle the last run
    bucket = min(RUN_BUCKETS-1, (run_len-1) // (DECK_SIZE // RUN_BUCKETS))
    cuda.atomic.add(block_run_hist, bucket, 1)

@cuda.jit(device=True, inline=True)
def compute_inversions(deck, rng_states, gid, lid):
    """Compute inversion counts using high-precision sampling and warp-level reduction."""
    samples_per_thread = INV_SAMPLE_SIZE // WARP_SIZE
    val = 0  # Local count for this thread
    
    for s in range(samples_per_thread):
        # Sample without replacement using precomputed indices
        pair = int(xoroshiro128p_uniform_float64(rng_states, gid) * INV_MAX)
        val += (deck[const_indices_j[pair]] > deck[const_indices_i[pair]])
    
    # Warp-level reduction using shuffle down
    lane_id = lid % WARP_SIZE
    # shuffle down by power-of-2 offsets with sync mask
    for offset in (16, 8, 4, 2, 1):
        val = cuda.shfl_down_sync(0xffffffff, val, offset)
    
    # Lane 0 of each warp now holds its sum
    if lane_id == 0:
        # Scale the sample count to estimate total inversions
        # With without-replacement sampling, we sample from N*(N-1)/2 pairs
        total_inv = (val * INV_MAX) // INV_SAMPLE_SIZE
        # Scale and down-bucket the estimated count
        bucket = min(INV_BUCKETS-1, total_inv // (INV_MAX // INV_BUCKETS))
        return bucket
    return 0

@cuda.jit(device=True, inline=True)
def compute_deck_metrics(deck, initial_const, block_hist, block_r):
    """Compute deck-level metrics (simc, runs, duplicates)."""
    simc = 0
    runs = 1
    prev = deck[0]
    match = True
    
    for k in range(DECK_SIZE):
        c = deck[k]
        match &= (c == initial_const[k])
        simc += (c == k)
        if k > 0:
            runs += (c < prev)
        prev = c
    
    if match: cuda.atomic.add(block_r, 1, 1)  # Increment block-level duplicate counter
    cuda.atomic.add(block_hist, simc, 1)
    cuda.atomic.add(block_r, 0, runs)

@cuda.jit(device=True, inline=True)
def compute_poker_mask(deck, block_poker):
    """Compute poker hand mask and update histogram."""
    mask = 0
    s0 = deck[0] // 13
    flush = 1
    
    # Unroll poker mask and flush check
    for i in range(5):
        c = deck[i]
        mask |= 1 << (c % 13)
        flush &= (c // 13 == s0)
    
    cuda.atomic.add(block_poker, get_poker_cat(mask, flush), 1)

@cuda.jit(device=True, inline=True)
def reduce_block_histograms_child(block_hist, block_poker, block_inv_hist, block_r, block_run_hist,
                                hist, poker, inv_hist, run_hist, rises, total, dup):
    """Child kernel for final reduction of block histograms."""
    tx = cuda.threadIdx.x
    bdx = cuda.blockDim.x
    
    # Each thread handles a portion of the histograms
    for i in range(tx, HIST_BINS, bdx):
        if block_hist[i] > 0:
            cuda.atomic.add(hist, i, block_hist[i])
    
    for i in range(tx, POKER_CATS_PER_U32, bdx):
        if block_poker[i] > 0:
            cuda.atomic.add(poker, i, block_poker[i])
    
    for i in range(tx, INV_BUCKETS, bdx):
        if block_inv_hist[i] > 0:
            cuda.atomic.add(inv_hist, i, block_inv_hist[i])
    
    for i in range(tx, RUN_BUCKETS, bdx):
        if block_run_hist[i] > 0:
            cuda.atomic.add(run_hist, i, block_run_hist[i])
    
    # Single thread updates global counters
    if tx == 0:
        cuda.atomic.add(rises, 0, block_r[0])
        cuda.atomic.add(total, 0, bdx * GPU_SHUFFLES_THR)
        cuda.atomic.add(dup, 0, block_r[1])

@cuda.jit(device=True, inline=True)
def shuffle_deck(deck, rng_states, gid):
    """Optimized Fisher-Yates shuffle with unrolled final iterations and vectorized loads."""
    # Unroll final 16 iterations in registers
    regs = cuda.local.array(16, dtype=numba.int8)
    for i in range(16):
        regs[i] = deck[DECK_SIZE-16+i]
    
    # Shuffle final 16 elements in registers
    for i in range(15, 0, -1):
        j = int(xoroshiro128p_uniform_float64(rng_states, gid) * (i+1))
        if j > i: j = i
        regs[i], regs[j] = regs[j], regs[i]
    
    # Vectorized shuffle of first half using uint2 loads
    for i in range(0, DECK_SIZE-16, 2):
        # Load two elements at once
        val = cuda.ldg(deck, i)
        val2 = cuda.ldg(deck, i+1)
        
        # Generate two random indices
        j = int(xoroshiro128p_uniform_float64(rng_states, gid) * (DECK_SIZE-16))
        j2 = int(xoroshiro128p_uniform_float64(rng_states, gid) * (DECK_SIZE-16))
        
        # Swap with vectorized store
        deck[i] = deck[j]
        deck[i+1] = deck[j2]
        deck[j] = val
        deck[j2] = val2
    
    # Store final 16 elements from registers
    for i in range(16):
        deck[DECK_SIZE-16+i] = regs[i]

@cuda.jit(max_registers=32, inline=True)
def shuffle_kernel(rng_states,
                  hist, poker, total, rises, dup,
                  inv_hist, run_hist):
    """GPU kernel for shuffling and metrics collection."""
    # Shared memory for local histograms
    block_hist = cuda.shared.array(shape=(HIST_BINS,), dtype=numba.int32)
    block_poker = cuda.shared.array(shape=(POKER_CATS_PER_U32,), dtype=numba.int32)
    block_inv_hist = cuda.shared.array(shape=(INV_BUCKETS,), dtype=numba.int32)
    block_run_hist = cuda.shared.array(shape=(RUN_BUCKETS,), dtype=numba.int32)
    block_r = cuda.shared.array(shape=(2,), dtype=numba.int32)  # [0] for runs, [1] for duplicates
    
    tx = cuda.threadIdx.x
    bdx = cuda.blockDim.x
    lid = tx % WARP_SIZE
    wid = tx // WARP_SIZE
    mask = 0xffffffff  # All lanes active

    # Initialize shared memory
    if lid == 0:
        for i in range(HIST_BINS):
            block_hist[i] = 0
        for i in range(POKER_CATS_PER_U32):
            block_poker[i] = 0
        for i in range(INV_BUCKETS):
            block_inv_hist[i] = 0
        for i in range(RUN_BUCKETS):
            block_run_hist[i] = 0
        block_r[0] = 0
        block_r[1] = 0
    cuda.syncthreads()

    # Only allocate deck array for first thread in warp
    deck = cuda.local.array(shape=(DECK_SIZE,), dtype=numba.int8) if lid == 0 else None
    gid = cuda.grid(1)

    for _ in range(GPU_SHUFFLES_THR):
        if lid == 0:
            # reset & shuffle
            for i in range(DECK_SIZE): deck[i]=i
            shuffle_deck(deck, rng_states, gid)

            # Compute metrics using bitboard operations
            simc = count_similarity(deck, initial_d)
            runs = count_runs(deck)
            
            # Check for exact match (duplicate)
            match = (simc == DECK_SIZE)
            if match: cuda.atomic.add(block_r, 1, 1)
            
            # Update histograms using warp-level ballot
            warp_histogram_add(simc, mask, block_hist)
            warp_histogram_add(runs, mask, block_run_hist)
            
            # Compute inversions and update histogram
            bucket = compute_inversions(deck, rng_states, gid, lid)
            if lid == 0:
                warp_histogram_add(bucket, mask, block_inv_hist)
            
            # Compute poker hand and update histogram
            mask = 0
            s0 = deck[0] // 13
            flush = 1
            for i in range(5):
                c = deck[i]
                mask |= 1 << (c % 13)
                flush &= (c // 13 == s0)
            cat = get_poker_cat(mask, flush)
            warp_histogram_add(cat, mask, block_poker)

    # Synchronize all threads before final updates
    cuda.syncthreads()

    # Single thread per block spawns child kernel for final reduction
    if tx == 0:
        # Launch child kernel with 128 threads (4 warps)
        child_threads = 128
        child_blocks = 1
        
        # Copy block histograms to device memory for child kernel
        block_hist_d = cuda.device_array(HIST_BINS, dtype=numba.int32)
        block_poker_d = cuda.device_array(POKER_CATS_PER_U32, dtype=numba.int32)
        block_inv_hist_d = cuda.device_array(INV_BUCKETS, dtype=numba.int32)
        block_run_hist_d = cuda.device_array(RUN_BUCKETS, dtype=numba.int32)
        block_r_d = cuda.device_array(2, dtype=numba.int32)
        
        # Copy data to device arrays
        for i in range(HIST_BINS):
            block_hist_d[i] = block_hist[i]
        for i in range(POKER_CATS_PER_U32):
            block_poker_d[i] = block_poker[i]
        for i in range(INV_BUCKETS):
            block_inv_hist_d[i] = block_inv_hist[i]
        for i in range(RUN_BUCKETS):
            block_run_hist_d[i] = block_run_hist[i]
        block_r_d[0] = block_r[0]
        block_r_d[1] = block_r[1]
        
        # Launch child kernel
        reduce_block_histograms_child[child_blocks, child_threads](
            block_hist_d, block_poker_d, block_inv_hist_d, block_r_d, block_run_hist_d,
            hist, poker, inv_hist, run_hist, rises, total, dup
        )
        
        # Free device arrays
        block_hist_d.free()
        block_poker_d.free()
        block_inv_hist_d.free()
        block_run_hist_d.free()
        block_r_d.free()

@cuda.jit(device=True, inline=True)
def compute_run_buckets(deck):
    """Compute run length buckets for histogram."""
    run_len = 1
    prev = deck[0]
    buckets = []
    
    for i in range(1, DECK_SIZE):
        if deck[i] > prev:
            run_len += 1
        else:
            bucket = min(RUN_BUCKETS-1, (run_len-1) // (DECK_SIZE // RUN_BUCKETS))
            buckets.append(bucket)
            run_len = 1
        prev = deck[i]
    
    bucket = min(RUN_BUCKETS-1, (run_len-1) // (DECK_SIZE // RUN_BUCKETS))
    buckets.append(bucket)
    return buckets

# Consolidate poker logic into a single function
@njit(nogil=True, fastmath=True, cache=True)
def categorize_poker_hand(deck):
    """Categorize a 5-card poker hand using the LUT.
    Returns category index (0-8) for the hand.
    """
    if deck.shape[0] != 5:
        return 0  # High card for invalid hands
        
    mask = 0
    s0 = deck[0] // 13
    flush = 1
    
    for i in range(5):
        c = deck[i]
        mask |= 1 << (c % 13)
        if c // 13 != s0:
            flush = 0
            
    return lut_packed[mask | (flush << 13)]

# ─────────── Dashboard State Abstraction ───────────────────────────────────────
class DashboardState:
    def __init__(self, mode, hist_data, total_data, rises_data, poker_data,
                 inv_hist_data=None, run_hist_data=None, dup_data=None):
        self.mode = mode
        self._hist_data = hist_data
        self._total_data = total_data
        self._rises_data = rises_data
        self._poker_data = poker_data
        # GPU only metrics
        self._inv_hist_data = inv_hist_data
        self._run_hist_data = run_hist_data
        self._dup_data = dup_data

        # Pre-calculate buffer views for CPU mode for efficiency
        if self.mode == "CPU":
            # Create stacked arrays for efficient summing
            self._hist_stack = np.stack([np.frombuffer(b, dtype=np.int32) for b in self._hist_data])
            self._total_stack = np.stack([np.frombuffer(b, dtype=np.int32) for b in self._total_data])
            self._rises_stack = np.stack([np.frombuffer(b, dtype=np.int32) for b in self._rises_data])
            self._poker_stack = np.stack([np.frombuffer(b, dtype=np.int32) for b in self._poker_data])
            
            # Initialize master buffers for efficient summing
            self._master_hist = np.zeros(HIST_BINS, dtype=np.int32)
            self._master_total = np.zeros(1, dtype=np.int32)
            self._master_rises = np.zeros(1, dtype=np.int32)
            self._master_poker = np.zeros(POKER_CATS_PER_U32, dtype=np.int32)
            self._last_sum_time = perf_counter()
            
            # Track histogram changes
            self._prev_hist = np.zeros(HIST_BINS, dtype=np.int32)
            self._changed_bins = set()
        else:  # GPU mode
            # Unwrap single-element lists into direct buffer views
            self._hist_view = np.frombuffer(self._hist_data[0], dtype=np.int32)
            self._total_view = np.frombuffer(self._total_data[0], dtype=np.int32)
            self._rises_view = np.frombuffer(self._rises_data[0], dtype=np.int32)
            self._poker_view = np.frombuffer(self._poker_data[0], dtype=np.int32)
            if self._inv_hist_data:
                self._inv_hist_view = np.frombuffer(self._inv_hist_data[0], dtype=np.int32)
            if self._run_hist_data:
                self._run_hist_view = np.frombuffer(self._run_hist_data[0], dtype=np.int32)
            if self._dup_data:
                self._dup_view = np.frombuffer(self._dup_data[0], dtype=np.int32)
            
            # Track histogram changes
            self._prev_hist = np.zeros(HIST_BINS, dtype=np.int32)
            self._changed_bins = set()

    def _update_master_buffers(self):
        """Update master buffers with latest worker data and track changes."""
        current_time = perf_counter()
        if current_time - self._last_sum_time >= 1.0:  # Update every second
            if self.mode == "CPU":
                new_hist = np.sum(self._hist_stack, axis=0)
                self._master_total[0] = np.sum(self._total_stack)
                self._master_rises[0] = np.sum(self._rises_stack)
                self._master_poker[:] = np.sum(self._poker_stack, axis=0)
            else:
                new_hist = self._hist_view.copy()
            
            # Track changed bins
            self._changed_bins.clear()
            for i in range(HIST_BINS):
                if new_hist[i] != self._prev_hist[i]:
                    self._changed_bins.add(i)
            
            # Update previous histogram
            self._prev_hist[:] = new_hist
            self._last_sum_time = current_time

    @property
    def hist(self):
        if self.mode == "GPU":
            return self._hist_view
        else:
            self._update_master_buffers()
            return self._master_hist

    @property
    def changed_bins(self):
        """Return the set of bins that changed since last update."""
        self._update_master_buffers()
        return self._changed_bins

    @property
    def total(self):
        if self.mode == "GPU":
            return int(self._total_view[0])
        else:
            self._update_master_buffers()
            return int(self._master_total[0])

    @property
    def rises(self):
        if self.mode == "GPU":
            return int(self._rises_view[0])
        else:
            self._update_master_buffers()
            return int(self._master_rises[0])

    @property
    def poker(self):
        if self.mode == "GPU":
            return self._poker_view
        else:
            self._update_master_buffers()
            return self._master_poker

    @property
    def inv_hist(self):
        if self.mode == "GPU" and hasattr(self, '_inv_hist_view'):
            return self._inv_hist_view
        return None

    @property
    def run_hist(self):
        if self.mode == "GPU" and hasattr(self, '_run_hist_view'):
            return self._run_hist_view
        return None

    @property
    def duplicates(self) -> int:
        if self.mode == "GPU" and hasattr(self, '_dup_view'):
            return int(self._dup_view[0])
        return 0

def build_layout():
    """Create and return the UI layout."""
    layout = Layout()
    layout.split_row(
        Layout(name="hist", ratio=2),
        Layout(name="side", ratio=1)
    )
    layout["side"].split_column(
        Layout(name="stats", ratio=1),
        Layout(name="poker", ratio=1),
        Layout(name="runs", ratio=1)
    )
    return layout

def _init_gpu_buffers():
    """Initialize GPU buffers with zero-copy memory."""
    cuda.select_device(0)
    dev = cuda.get_current_device()
    
    # Calculate optimal block size based on device properties
    max_threads = dev.MAX_THREADS_PER_BLOCK
    # Try different thread counts to find optimal occupancy
    # 128 threads = 4 warps/block, good for register-heavy kernels
    # 256 threads = 8 warps/block, balanced
    # 512 threads = 16 warps/block, good for memory-bound kernels
    threads_per_block = 128  # Start with 128 for better register usage
    blocks_per_grid = (dev.MULTIPROCESSOR_COUNT * 4)  # Increase blocks to compensate for smaller thread count
    
    # Create streams for double buffering
    streams = [cuda.stream(), cuda.stream()]
    
    # Initialize zero-copy buffers for each metric
    hist_bufs = []
    total_bufs = []
    rises_bufs = []
    poker_bufs = []
    inv_hist_bufs = []
    run_hist_bufs = []
    dup_bufs = []
    
    for _ in range(2):
        # Create true zero-copy mapped arrays (host and device share memory)
        hist_dev = cuda.mapped_array(HIST_BINS, dtype=np.int32)
        total_dev = cuda.mapped_array(1, dtype=np.int32)
        rises_dev = cuda.mapped_array(1, dtype=np.int32)
        poker_dev = cuda.mapped_array(POKER_CATS_PER_U32, dtype=np.int32)
        inv_hist_dev = cuda.mapped_array(INV_BUCKETS, dtype=np.int32)
        run_hist_dev = cuda.mapped_array(RUN_BUCKETS, dtype=np.int32)
        dup_dev = cuda.mapped_array(1, dtype=np.int32)
        
        # Host and device arrays are the same memory
        hist_host = hist_dev
        total_host = total_dev
        rises_host = rises_dev
        poker_host = poker_dev
        inv_hist_host = inv_hist_dev
        run_hist_host = run_hist_dev
        dup_host = dup_dev
        
        # Store both host and device arrays (they're the same memory)
        hist_bufs.append((hist_host, hist_dev))
        total_bufs.append((total_host, total_dev))
        rises_bufs.append((rises_host, rises_dev))
        poker_bufs.append((poker_host, poker_dev))
        inv_hist_bufs.append((inv_hist_host, inv_hist_dev))
        run_hist_bufs.append((run_hist_host, run_hist_dev))
        dup_bufs.append((dup_host, dup_dev))
    
    # Initialize RNG states
    rng_d = cuda.to_device(create_xoroshiro128p_states(
        threads_per_block * blocks_per_grid,
        int.from_bytes(os.urandom(8), 'little')
    ))
    
    return (hist_bufs, poker_bufs, total_bufs, rises_bufs, dup_bufs, inv_hist_bufs, run_hist_bufs,
            None, None, rng_d, streams, threads_per_block, blocks_per_grid)

def simulation_worker(mode: str, metrics_buffer: MetricsBuffer, stop_evt: mp.Event):
    """Worker process that runs the simulation and writes metrics to the ring buffer."""
    try:
        if mode == "CPU":
            # Initialize CPU buffers
            (hist_bufs,
             total_bufs,
             rises_bufs,
             poker_bufs,
             rng_bufs,
             run_hist_bufs,
             _,          # inv_hist (unused)
             _           # dup (unused)
            ) = init_buffers(mode)
            
            # Start CPU workers
            workers = []
            for i in range(CPU_WORKERS):
                p = mp.Process(
                    target=shuffle_worker_cpu,
                    args=(
                        hist_bufs[i],
                        total_bufs[i],
                        rises_bufs[i],
                        poker_bufs[i],
                        run_hist_bufs[i],
                        rng_bufs[i],
                        stop_evt
                    )
                )
                p.start()
                workers.append(p)
            
            # Main simulation loop
            while not stop_evt.is_set():
                # Collect metrics from all workers
                hist = np.sum([np.frombuffer(b, dtype=np.int32) for b in hist_bufs], axis=0)
                total = sum(np.frombuffer(b, dtype=np.int32)[0] for b in total_bufs)
                rises = sum(np.frombuffer(b, dtype=np.int32)[0] for b in rises_bufs)
                poker = np.sum([np.frombuffer(b, dtype=np.int32) for b in poker_bufs], axis=0)
                run_hist = np.sum([np.frombuffer(b, dtype=np.int32) for b in run_hist_bufs], axis=0)
                
                # Write metrics to ring buffer
                metrics_buffer.write_metrics({
                    'hist': hist,
                    'total': total,
                    'rises': rises,
                    'poker': poker,
                    'run_hist': run_hist,
                    'timestamp': perf_counter()
                })
                
                sleep(1.0 / cfg.UI_FPS)  # Match UI refresh rate
            
            # Clean up
            for worker in workers:
                worker.join()
                
        else:  # GPU mode
            # Initialize GPU buffers
            (hist_bufs,
             poker_bufs,
             total_bufs,
             rises_bufs,
             dup_bufs,
             inv_hist_bufs,
             run_hist_bufs,
             _,          # init_d (unused, now in constant memory)
             _,          # lut_d (unused, now in constant memory)
             rng_d,
             streams,
             threads_per_block,
             blocks_per_grid
            ) = init_buffers(mode)
            
            # Main simulation loop
            buf_idx = 0
            while not stop_evt.is_set():
                s = streams[buf_idx]
                
                try:
                    # Get current buffer pair
                    hist_host, hist_dev = hist_bufs[buf_idx]
                    poker_host, poker_dev = poker_bufs[buf_idx]
                    total_host, total_dev = total_bufs[buf_idx]
                    rises_host, rises_dev = rises_bufs[buf_idx]
                    dup_host, dup_dev = dup_bufs[buf_idx]
                    inv_hist_host, inv_hist_dev = inv_hist_bufs[buf_idx]
                    run_hist_host, run_hist_dev = run_hist_bufs[buf_idx]
                    
                    # Clear host buffers asynchronously
                    hist_host.fill(0)
                    poker_host.fill(0)
                    total_host.fill(0)
                    rises_host.fill(0)
                    dup_host.fill(0)
                    inv_hist_host.fill(0)
                    run_hist_host.fill(0)
                    
                    # Run kernel with updated signature (no initial or lut args)
                    shuffle_kernel[blocks_per_grid, threads_per_block, s](
                        rng_d,
                        hist_dev, poker_dev, total_dev,
                        rises_dev, dup_dev, inv_hist_dev, run_hist_dev
                    )
                    
                    # Synchronize stream to ensure kernel completion
                    s.synchronize()
                    
                    # Write metrics to ring buffer
                    metrics_buffer.write_metrics({
                        'hist': hist_host,
                        'total': int(total_host[0]),
                        'rises': int(rises_host[0]),
                        'poker': poker_host,
                        'inv_hist': inv_hist_host,
                        'run_hist': run_hist_host,
                        'duplicates': int(dup_host[0]),
                        'timestamp': perf_counter()
                    })
                    
                    # Switch buffer
                    buf_idx = 1 - buf_idx
                    
                except CudaAPIError as e:
                    print(f"GPU kernel error: {e}", flush=True)
                    break
                
                sleep(1.0 / cfg.UI_FPS)  # Match UI refresh rate
            
            # Clean up
            cuda.close()
            
    except Exception as e:
        print(f"Simulation error: {e}", flush=True)
    finally:
        metrics_buffer.close()

def ui_worker(metrics_buffer: MetricsBuffer, stop_evt: mp.Event):
    """Worker process that handles UI rendering with partial updates."""
    try:
        # Create UI layout
        layout = build_layout()
        t0 = perf_counter()
        
        # Create a dictionary to store individual line panels
        hist_lines = {}
        for i in range(HIST_BINS):
            hist_lines[i] = Panel("", height=1)
        
        # Create the main histogram panel with all line panels
        hist_panel = Panel("\n".join(hist_lines[i].renderable for i in range(HIST_BINS)),
                          title="Positional Similarity", border_style="cyan")
        layout["hist"].update(hist_panel)
        
        with Live(layout, refresh_per_second=cfg.UI_FPS, screen=True) as live:
            while not stop_evt.is_set():
                # Read latest metrics from ring buffer
                metrics = metrics_buffer.read_metrics()
                if metrics is None:
                    sleep(1.0 / cfg.UI_FPS)
                    continue
                
                # Update UI
                elapsed = metrics['timestamp'] - t0
                
                # Update only changed histogram lines
                for i, line in render_hist(metrics):
                    hist_lines[i].update(line)
                
                # Update other panels
                layout["side"]["stats"].update(render_stats(metrics, elapsed))
                layout["side"]["poker"].update(render_poker(metrics))
                layout["side"]["runs"].update(render_runs(metrics))
                
                sleep(1.0 / cfg.UI_FPS)
                
    except KeyboardInterrupt:
        print("\033[2J\033[H", end="", flush=True)
    except Exception as e:
        print(f"UI error: {e}", flush=True)
    finally:
        stop_evt.set()

def main():
    print("🚀 Starting Dashboard with Metrics — Ctrl‑C to quit", flush=True)
    
    stop_evt = None
    metrics_buffer = None
    
    try:
        # Determine execution mode
        use_gpu = cuda.is_available()
        mode = "GPU" if use_gpu else "CPU"
        print(f"Using {mode} mode for simulation", flush=True)
        
        # Create shared metrics buffer
        if os.name == 'nt':  # Windows
            metrics_buffer = MetricsBuffer("shuffle_metrics", use_windows=True)
        else:  # Linux/Unix
            metrics_buffer = MetricsBuffer("shuffle_metrics")
        
        # Create stop event
        stop_evt = mp.Event()
        
        # Start simulation and UI processes
        sim_proc = mp.Process(target=simulation_worker, args=(mode, metrics_buffer, stop_evt))
        ui_proc = mp.Process(target=ui_worker, args=(metrics_buffer, stop_evt))
        
        sim_proc.start()
        ui_proc.start()
        
        # Wait for processes to finish
        sim_proc.join()
        ui_proc.join()
        
    except Exception as e:
        print(f"\nError: {e}", flush=True)
    finally:
        if stop_evt is not None:
            stop_evt.set()
        if metrics_buffer is not None:
            metrics_buffer.close()

if __name__ == "__main__":
    main()

# Add bitboard utilities
@cuda.jit(device=True, inline=True)
def deck_to_bitboards(deck):
    """Convert deck to four 64-bit bitboards (one per quarter)."""
    b0 = np.uint64(0)
    b1 = np.uint64(0)
    b2 = np.uint64(0)
    b3 = np.uint64(0)
    
    for i in range(DECK_SIZE):
        if i < 16:
            b0 |= np.uint64(1) << (deck[i] * 4)
        elif i < 32:
            b1 |= np.uint64(1) << (deck[i] * 4)
        elif i < 48:
            b2 |= np.uint64(1) << (deck[i] * 4)
        else:
            b3 |= np.uint64(1) << (deck[i] * 4)
    return b0, b1, b2, b3

@cuda.jit(device=True, inline=True)
def count_similarity(deck, initial_d):
    """Count similar cards using bitboard operations."""
    b0, b1, b2, b3 = deck_to_bitboards(deck)
    # Use __ldg for constant memory access
    i0 = cuda.ldg(initial_d, 0)
    i1 = cuda.ldg(initial_d, 1)
    i2 = cuda.ldg(initial_d, 2)
    i3 = cuda.ldg(initial_d, 3)
    
    # Count matches in each quarter
    same0 = ~(b0 ^ i0)
    same1 = ~(b1 ^ i1)
    same2 = ~(b2 ^ i2)
    same3 = ~(b3 ^ i3)
    
    # Sum popcounts
    return (cuda.popc(same0) + cuda.popc(same1) + 
            cuda.popc(same2) + cuda.popc(same3)) // 4

@cuda.jit(device=True, inline=True)
def count_runs(deck):
    """Count runs using bitboard operations."""
    b0, b1, b2, b3 = deck_to_bitboards(deck)
    
    # Shift and mask to find rising edges
    rising0 = (b0 >> 4) & ~b0
    rising1 = (b1 >> 4) & ~b1
    rising2 = (b2 >> 4) & ~b2
    rising3 = (b3 >> 4) & ~b3
    
    # Sum rising edges
    return (cuda.popc(rising0) + cuda.popc(rising1) + 
            cuda.popc(rising2) + cuda.popc(rising3)) + 1

@cuda.jit(device=True, inline=True)
def warp_histogram_add(val, mask, hist):
    """Add a value to histogram using warp-level ballot."""
    # Get lane mask for this value
    lane_mask = cuda.ballot_sync(mask, val)
    
    # Only first active lane for each value does the atomic add
    if lane_mask & (1 << cuda.laneid):
        count = cuda.popc(lane_mask)
        cuda.atomic.add(hist, val, count)
