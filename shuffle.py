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
import math
import ctypes
from ctypes import c_int64, c_uint64, c_uint8, Structure, c_int32
from multiprocessing import shared_memory, Lock
import atexit # For cleanup
import sys # For platform check

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
INV_BUCKETS = numba.int32(64)  # Number of buckets for inversion histogram
RUN_BUCKETS = 16  # Number of buckets for run length histogram

# Performance tuning
UI_FPS = 60
GPU_SHUFFLES_THR = 5000  # ~128×256×5k ≈ 163M shuffles per kernel launch
WARP_SIZE = 32
INV_SAMPLE_SIZE = 1000  # Sample 1000 pairs for higher precision

# Poker hand masks
ROYAL_MASK = (1<<0)|(1<<9)|(1<<10)|(1<<11)|(1<<12)
CONSEC5 = 0b11111

# Precompute bar color
_r,_g,_b = hsv_to_rgb(0.66,1,1)
BAR_COLOR = f"rgb({int(_r*255)},{int(_g*255)},{int(_b*255)})"

# Add CPU specific constants if needed
CPU_CHUNK_SIZE = 10_000 # Number of shuffles per chunk on CPU

@dataclass(frozen=True)
class Config:
    # Core configuration
    DECK_SIZE: int = DECK_SIZE
    HIST_BINS: int = HIST_BINS
    POKER_CATEGORIES: int = 10 # Added
    HIST_WIDTH: int = HIST_WIDTH
    INV_MAX: int = INV_MAX
    RUN_BINS: int = RUN_BINS
    INV_BUCKETS: int = INV_BUCKETS
    RUN_BUCKETS: int = RUN_BUCKETS
    
    # Performance tuning
    UI_FPS: int = UI_FPS
    GPU_SHUFFLES_THR: int = GPU_SHUFFLES_THR
    WARP_SIZE: int = WARP_SIZE
    INV_SAMPLE_SIZE: int = INV_SAMPLE_SIZE
    
    # Poker hand masks
    ROYAL_MASK: int = ROYAL_MASK
    CONSEC5: int = CONSEC5
    CPU_CHUNK_SIZE: int = CPU_CHUNK_SIZE # Add CPU chunk size

cfg = Config()

# ─────────── Shared Metrics Structure ───────────────────────────────────

# Define the layout for the *data* part of the shared memory
class MetricsDataStruct(Structure):
    _fields_ = [
        # Data arrays
        ('hist_data', c_int32 * cfg.HIST_BINS),
        ('poker_data', c_int32 * cfg.POKER_CATEGORIES), # Use POKER_CATEGORIES
        ('run_length_data', c_int32 * cfg.RUN_BUCKETS),
        ('inversion_data', c_int32 * cfg.INV_BUCKETS),
        # Scalar metrics
        ('total', c_int64),
        ('rises', c_int64),
        ('dup', c_int64),
        # Timestamp (optional, can be added by writer or reader)
        # ('timestamp', c_double) # Example if adding timestamp
    ]

# Define the layout for the *control block* in shared memory
class ControlBlockStruct(Structure):
     _fields_ = [
         ('head', c_int64), # Sequence number for writes
         ('tail', c_int64)  # Sequence number for reads
     ]

# Calculate sizes
METRICS_DATA_SIZE = ctypes.sizeof(MetricsDataStruct)
CONTROL_BLOCK_SIZE = ctypes.sizeof(ControlBlockStruct)
TOTAL_SHM_SIZE = CONTROL_BLOCK_SIZE + METRICS_DATA_SIZE

# Name for the shared memory block (unique)
# Using PID is good practice for avoiding collisions from stale runs
SHM_NAME = f"shuffle_metrics_shm_{os.getpid()}"

class SafeMetricsBuffer:
    """Process-safe buffer for simulation metrics using shared_memory and Lock."""

    def __init__(self, create=False, name=SHM_NAME):
        self.name = name
        self.created = create
        self.shm = None
        self.lock = Lock()
        # Initialize view attributes to None
        self.control_block = None
        self.metrics_data = None

        try:
            if create:
                self.shm = shared_memory.SharedMemory(name=self.name, create=True, size=TOTAL_SHM_SIZE)
                print(f"Created shared memory: {self.name} ({TOTAL_SHM_SIZE} bytes)")
                control_block_view = ControlBlockStruct.from_buffer(self.shm.buf)
                control_block_view.head = 0
                control_block_view.tail = 0
                del control_block_view # Release temp view
            else:
                self.shm = shared_memory.SharedMemory(name=self.name, create=False)
                print(f"Attached to shared memory: {self.name}")

            # Get persistent views
            self.control_block = ControlBlockStruct.from_buffer(self.shm.buf)
            # Use slice for offset
            self.metrics_data = MetricsDataStruct.from_buffer(self.shm.buf[CONTROL_BLOCK_SIZE:])

            if create:
                 # Register unlink for cleanup, unlink calls close internally
                 atexit.register(self.unlink)

        except FileNotFoundError:
             print(f"Error: Shared memory block '{self.name}' not found. Was the creating process started?")
             raise
        except Exception as e:
             print(f"Error initializing SafeMetricsBuffer ('{self.name}'): {e}")
             # Ensure close is called during failed init cleanup
             self.close()
             raise

    def write_metrics(self, metrics_dict):
        """Write metrics to the shared buffer. Assumes only one writer."""
        with self.lock:
            try:
                if 'hist' in metrics_dict:
                     np.copyto(np.frombuffer(self.metrics_data.hist_data, dtype=np.int32), metrics_dict['hist'])
                if 'poker' in metrics_dict:
                     np.copyto(np.frombuffer(self.metrics_data.poker_data, dtype=np.int32), metrics_dict['poker'])
                if 'run_hist' in metrics_dict:
                     np.copyto(np.frombuffer(self.metrics_data.run_length_data, dtype=np.int32), metrics_dict['run_hist'])
                if 'inv_hist' in metrics_dict:
                     np.copyto(np.frombuffer(self.metrics_data.inversion_data, dtype=np.int32), metrics_dict['inv_hist'])
                self.metrics_data.total = metrics_dict.get('total', 0)
                self.metrics_data.rises = metrics_dict.get('rises', 0)
                self.metrics_data.dup = metrics_dict.get('duplicates', 0)
                self.control_block.head += 1
            except AttributeError:
                print("SafeMetricsBuffer write error: Buffer likely closed or views not initialized.")
            except KeyError as e:
                print(f"SafeMetricsBuffer write error: Missing key {e}")
            except Exception as e:
                print(f"SafeMetricsBuffer write error: {e}")

    def read_metrics(self):
        """Read the latest metrics data if available. Assumes one reader."""
        with self.lock:
            try:
                head = self.control_block.head
                tail = self.control_block.tail
                if head == tail: return None
                # Ensure correct dtypes when reading from buffer
                metrics_dict = {
                    'hist': np.copy(np.frombuffer(self.metrics_data.hist_data, dtype=np.int32)),
                    'poker': np.copy(np.frombuffer(self.metrics_data.poker_data, dtype=np.int32)),
                    'run_hist': np.copy(np.frombuffer(self.metrics_data.run_length_data, dtype=np.int32)),
                    'inv_hist': np.copy(np.frombuffer(self.metrics_data.inversion_data, dtype=np.int32)),
                    'total': self.metrics_data.total,      # Already int64 from ctypes
                    'rises': self.metrics_data.rises,      # Already int64 from ctypes
                    'duplicates': self.metrics_data.dup,  # Already int64 from ctypes
                    'timestamp': perf_counter()
                }
                self.control_block.tail = head
                return metrics_dict
            except AttributeError:
                 print("SafeMetricsBuffer read error: Buffer likely closed or views not initialized.")
                 return None
            except Exception as e:
                 print(f"SafeMetricsBuffer read error: {e}")
                 return None

    def close(self):
        """Close the shared memory view, releasing internal pointers."""
        print(f"Closing shared memory view for: {self.name}")
        # --- Release ctypes views FIRST ---
        if hasattr(self, 'control_block') and self.control_block is not None:
            # print(f"Deleting control_block view for {self.name}")
            del self.control_block
            self.control_block = None
        if hasattr(self, 'metrics_data') and self.metrics_data is not None:
            # print(f"Deleting metrics_data view for {self.name}")
            del self.metrics_data
            self.metrics_data = None
        # ----------------------------------
        
        # Now close the shared memory object itself
        if self.shm:
            try:
                 self.shm.close()
                 # print(f"shm.close() called for {self.name}")
            except Exception as e:
                 print(f"Error closing shared memory object {self.name}: {e}")
            # Prevent further use
            self.shm = None

    def unlink(self):
        """Remove the shared memory block (called by creator)."""
        print(f"Unlinking shared memory: {self.name}")
        # Ensure the view is closed first (close handles releasing pointers)
        self.close()
        # Now attempt to unlink
        try:
            # Need a temporary handle to unlink if self.shm was already closed/nulled
            temp_shm = shared_memory.SharedMemory(name=self.name)
            temp_shm.unlink()
            print(f"Successfully unlinked {self.name}")
        except FileNotFoundError:
            print(f"Shared memory {self.name} already unlinked or never created.")
        except Exception as e:
            print(f"Error unlinking shared memory {self.name}: {e}")

    def __del__(self):
        # Optional: Ensure close is called if the object is garbage collected
        # Can be problematic with process cleanup, rely on explicit close/unlink
        # print(f"SafeMetricsBuffer __del__ called for {self.name}")
        # self.close()
        pass

# ─────────── Utility Functions ────────────────────────────────────────────


# ─────────── Poker Hand Categorization (Shared CPU/GPU Logic) ───────────
# Poker categories mapping
POKER_CATEGORIES_MAP = {
    0: "High Card", 1: "One Pair", 2: "Two Pair", 3: "3 of a Kind", 4: "Straight",
    5: "Flush", 6: "Full House", 7: "4 of a Kind", 8: "Straight Flush", 9: "Royal Flush"
}

# ─────────── Poker Hand Categorization (GPU Device Function) ──────────────
@cuda.jit(device=True, fastmath=True)
def compute_poker_category_device(deck5):
    """Categorize a 5-card poker hand directly on the GPU."""
    # Categories: 0:HighCard, 1:Pair, 2:TwoPair, 3:Trips, 4:Straight,
    #             5:Flush, 6:FullHouse, 7:Quads, 8:StraightFlush, 9:RoyalFlush

    ranks = cuda.local.array(5, dtype=numba.int8)
    suits = cuda.local.array(5, dtype=numba.int8)
    rank_counts = cuda.local.array(13, dtype=numba.int8) # Counts for ranks 0-12 (2-Ace)
    for i in range(13): rank_counts[i] = 0

    flush = True
    first_suit = deck5[0] // 13

    for i in range(5):
        card = deck5[i]
        rank = card % 13 # 0=2, ..., 8=10, 9=J, 10=Q, 11=K, 12=A
        suit = card // 13
        ranks[i] = rank
        suits[i] = suit
        rank_counts[rank] += 1
        if suit != first_suit: flush = False

    # Sort ranks for straight check (simple insertion sort for 5 elements)
    for i in range(1, 5):
        key = ranks[i]
        j = i - 1
        while j >= 0 and ranks[j] > key:
            ranks[j + 1] = ranks[j]
            j -= 1
        ranks[j + 1] = key

    # Check for straight
    straight = True
    # Ace-low straight check (A, 2, 3, 4, 5 -> ranks 12, 0, 1, 2, 3)
    is_ace_low = (ranks[0]==0 and ranks[1]==1 and ranks[2]==2 and ranks[3]==3 and ranks[4]==12)
    if not is_ace_low:
        for i in range(4):
            if ranks[i+1] != ranks[i] + 1:
                straight = False
                break
    else:
         straight = True # Ace-low counts as straight

    # Check for Royal Flush (Ace-high straight + flush)
    is_royal = straight and flush and ranks[4] == 12 and ranks[0]==8 # Ranks 10,J,Q,K,A -> 8,9,10,11,12

    if is_royal:
        return 9
    if straight and flush: # Includes Ace-low straight flush
        return 8

    # Check rank counts for pairs, trips, quads, full house
    has_quads = False
    has_trips = False
    pairs = 0
    for i in range(13):
        if rank_counts[i] == 4: has_quads = True
        if rank_counts[i] == 3: has_trips = True
        if rank_counts[i] == 2: pairs += 1

    if has_quads:
        return 7
    if has_trips and pairs == 1:
        return 6 # Full House
    if flush:
        return 5
    if straight:
        return 4
    if has_trips:
        return 3
    if pairs == 2:
        return 2
    if pairs == 1:
        return 1

    return 0 # High Card

# ─────────── Poker Hand Categorization (CPU JIT Function) ───────────────
@njit(nogil=True, cache=True)
def compute_poker_category_cpu(deck5):
    """Categorize a 5-card poker hand on the CPU (Numba JIT)."""
    # Logic mirrors the GPU version but uses standard operations
    ranks = np.empty(5, dtype=np.int8)
    suits = np.empty(5, dtype=np.int8)
    rank_counts = np.zeros(13, dtype=np.int8)

    flush = True
    first_suit = deck5[0] // 13

    for i in range(5):
        card = deck5[i]
        rank = card % 13
        suit = card // 13
        ranks[i] = rank
        suits[i] = suit
        rank_counts[rank] += 1
        if suit != first_suit: flush = False

    # Sort ranks (use np.sort which is JIT-compatible)
    ranks.sort()

    # Check for straight
    straight = True
    is_ace_low = (ranks[0]==0 and ranks[1]==1 and ranks[2]==2 and ranks[3]==3 and ranks[4]==12)
    if not is_ace_low:
        for i in range(4):
            if ranks[i+1] != ranks[i] + 1:
                straight = False
                break
    else:
        straight = True

    # Royal Flush
    is_royal = straight and flush and ranks[4] == 12 and ranks[0]==8
    if is_royal: return 9
    if straight and flush: return 8

    # Rank counts
    has_quads = False; has_trips = False; pairs = 0
    for i in range(13):
        count = rank_counts[i]
        if count == 4: has_quads = True
        if count == 3: has_trips = True
        if count == 2: pairs += 1

    if has_quads: return 7
    if has_trips and pairs == 1: return 6 # Full House
    if flush: return 5
    if straight: return 4
    if has_trips: return 3
    if pairs == 2: return 2
    if pairs == 1: return 1
    return 0 # High Card

# ─────────── CPU Simulation Logic ─────────────────────────────────────── 
@njit(nogil=True, cache=True)
def shuffle_deck_cpu(deck, rng_state):
    """Fisher-Yates shuffle for CPU (Numba JIT). Uses np.random."""
    n = len(deck)
    for i in range(n - 1, 0, -1):
        j = rng_state.integers(0, i + 1) # Equivalent to int(rng() * (i + 1))
        deck[i], deck[j] = deck[j], deck[i]

@njit(nogil=True, cache=True)
def compute_metrics_cpu(deck, deck_size):
    """Calculate similarity and runs for a deck (CPU JIT)."""
    simc = 0
    runs = 1
    prev_card = deck[0]
    exact_match = True
    initial_val = np.int8(0)

    for k in range(deck_size):
        c = deck[k]
        # Similarity to sorted 0..N-1
        simc += (c == k)
        # Exact match to initial state (requires initial state, pass or recreate)
        # For simplicity, assume initial state is always 0..N-1 here
        exact_match &= (c == initial_val)
        initial_val += 1
        # Runs
        if k > 0:
            runs += (c < prev_card)
        prev_card = c
        
    return simc, runs, exact_match

@njit(nogil=True, cache=True)
def compute_exact_inversions_cpu(deck):
    """Calculate the exact number of inversions (CPU JIT O(N^2))."""
    count = 0
    n = len(deck)
    for i in range(n):
        for j in range(i + 1, n):
            if deck[i] > deck[j]:
                count += 1
    return count

@njit(nogil=True, cache=True)
def compute_run_buckets_cpu(deck):
    """Calculate run length histogram buckets (CPU JIT)."""
    run_hist = np.zeros(cfg.RUN_BUCKETS, dtype=np.int32)
    run_len = 1
    prev = deck[0]
    bucket_width_run = cfg.DECK_SIZE // cfg.RUN_BUCKETS or 1

    for i in range(1, cfg.DECK_SIZE):
        if deck[i] > prev:
            run_len += 1
        else:
            bucket = min(cfg.RUN_BUCKETS - 1, (run_len - 1) // bucket_width_run)
            run_hist[bucket] += 1
            run_len = 1
        prev = deck[i]
    # Last run
    bucket = min(cfg.RUN_BUCKETS - 1, (run_len - 1) // bucket_width_run)
    run_hist[bucket] += 1
    return run_hist

# ─────────── CUDA Kernels (Split) ───────────────────────────────────────

# --- Device Helper Functions ---
@cuda.jit(device=True, inline=True)
def compute_inversions(deck, rng_states, gid, lid, 
                       const_indices_i_d, const_indices_j_d):
    """Compute inversion counts using sampling and Cooperative Groups warp reduction."""
    samples_per_thread = cfg.INV_SAMPLE_SIZE // cfg.WARP_SIZE
    if samples_per_thread <= 0: samples_per_thread = 1
    val = 0
    for s in range(samples_per_thread):
        pair_idx_f = xoroshiro128p_uniform_float64(rng_states, gid)
        pair = int(pair_idx_f * cfg.INV_MAX)
        if pair >= cfg.INV_MAX: pair = cfg.INV_MAX - 1
        # Use passed device arrays
        idx_i = const_indices_i_d[pair]
        idx_j = const_indices_j_d[pair]
        val += (deck[idx_j] > deck[idx_i])
    warp_group = cuda.cg.coalesced_group()
    warp_sum = warp_group.reduce_add(val)
    total_inv_est = (warp_sum * cfg.INV_MAX) // cfg.INV_SAMPLE_SIZE if cfg.INV_SAMPLE_SIZE > 0 else 0
    bucket_width = cfg.INV_MAX // cfg.INV_BUCKETS or 1
    bucket = total_inv_est // bucket_width
    if bucket >= cfg.INV_BUCKETS: bucket = cfg.INV_BUCKETS - 1
    return bucket

@cuda.jit(device=True, inline=True)
def shuffle_deck(deck, rng_states, gid):
    for i in range(cfg.DECK_SIZE - 1, 0, -1):
         j_f = xoroshiro128p_uniform_float64(rng_states, gid)
         j = int(j_f * (i + 1))
         if j > i: j = i
         deck[i], deck[j] = deck[j], deck[i]

# --- Kernel Phase 1: Shuffle and Basic Stats ---
@cuda.jit
def shuffle_kernel_phase1(rng_states, total, dup, shuffled_decks_d, 
                          initial_d_k):
    """Kernel Phase 1: Shuffle decks, check duplicates, write to buffer."""
    tx = cuda.threadIdx.x
    bdx = cuda.blockDim.x
    bid = cuda.blockIdx.x
    gid = bid * bdx + tx

    deck = cuda.local.array(shape=(cfg.DECK_SIZE,), dtype=numba.int8)

    # Reset deck from passed initial_d_k
    for i in range(cfg.DECK_SIZE):
        deck[i] = initial_d_k[i]

    shuffle_deck(deck, rng_states, gid)

    # Check for exact match against passed initial_d_k
    exact_match = True
    for k in range(cfg.DECK_SIZE):
        if deck[k] != initial_d_k[k]:
            exact_match = False
            break

    deck_offset = gid * cfg.DECK_SIZE
    for i in range(cfg.DECK_SIZE):
        shuffled_decks_d[deck_offset + i] = deck[i]

    cuda.atomic.add(total, 0, 1)
    if exact_match:
        cuda.atomic.add(dup, 0, 1)

# --- Kernel Phase 2: Analyze Decks ---
@cuda.jit(max_registers=64)
def analyze_kernel_phase2(shuffled_decks_d, hist, poker, rises, inv_hist, run_hist, rng_states,
                          const_indices_i_d, const_indices_j_d):
    """Kernel Phase 2: Analyze shuffled decks from buffer."""
    tx = cuda.threadIdx.x
    bdx = cuda.blockDim.x
    bid = cuda.blockIdx.x
    gid = bid * bdx + tx
    lid = tx % 32

    deck = cuda.local.array(shape=(cfg.DECK_SIZE,), dtype=numba.int8)
    deck5 = cuda.local.array(5, dtype=numba.int8)

    deck_offset = gid * cfg.DECK_SIZE
    for i in range(cfg.DECK_SIZE):
        deck[i] = shuffled_decks_d[deck_offset + i]

    # Perform Analysis
    simc = 0; runs = 1; prev_card = deck[0]
    for k in range(cfg.DECK_SIZE):
        c = deck[k]
        simc += (c == k)
        if k > 0: runs += (c < prev_card)
        prev_card = c
        if k < 5: deck5[k] = c

    # Update Histograms/Counters
    cuda.atomic.add(hist, simc, 1)
    cuda.atomic.add(rises, 0, runs)

    # Run Length
    run_len = 1; prev = deck[0]
    for i in range(1, cfg.DECK_SIZE):
        if deck[i] > prev:
            run_len += 1
        else:
            bucket_width_run = cfg.DECK_SIZE // cfg.RUN_BUCKETS or 1
            bucket = min(cfg.RUN_BUCKETS - 1, (run_len - 1) // bucket_width_run)
            cuda.atomic.add(run_hist, bucket, 1)
            run_len = 1
        prev = deck[i]
    bucket_width_run = cfg.DECK_SIZE // cfg.RUN_BUCKETS or 1
    bucket = min(cfg.RUN_BUCKETS - 1, (run_len - 1) // bucket_width_run)
    cuda.atomic.add(run_hist, bucket, 1)

    # Poker
    cat = compute_poker_category_device(deck5)
    cuda.atomic.add(poker, cat, 1)

    # Inversions - pass needed device arrays
    inv_bucket = compute_inversions(deck, rng_states, gid, tx, 
                                    const_indices_i_d, const_indices_j_d)
    if lid == 0:
        cuda.atomic.add(inv_hist, inv_bucket, 1)

# ─────────── UI Setup & Workers ──────────────────────────────────────────

def build_layout():
    # ... (build_layout remains the same) ...
    layout = Layout()
    layout.split_row(
        Layout(name="hist", ratio=2),
        Layout(name="side", ratio=1)
    )
    layout["side"].split_column(
        Layout(name="stats", ratio=1),
        Layout(name="poker", ratio=1),
        Layout(name="runs", ratio=1),
        Layout(name="inv", ratio=1)
    )
    return layout

def _init_gpu_buffers(total_threads):
    """Initialize GPU buffers, including device constants."""
    try:
        cuda.select_device(0)
        dev = cuda.get_current_device()
    except cuda.cudadrv.driver.CudaAPIError as e:
        print(f"CUDA Error: {e}")
        return None

    stream = cuda.stream()

    try:
        # --- Initialize Device Constants Here ---
        pre_i_host = np.zeros(cfg.INV_MAX, dtype=np.uint8)
        pre_j_host = np.zeros(cfg.INV_MAX, dtype=np.uint8)
        idx = 0
        for i in range(cfg.DECK_SIZE):
            for j in range(i + 1, cfg.DECK_SIZE):
                pre_i_host[idx] = i
                pre_j_host[idx] = j
                idx += 1
        
        initial_d_dev = cuda.to_device(np.arange(cfg.DECK_SIZE, dtype=np.int8), stream=stream)
        const_indices_i_dev = cuda.to_device(pre_i_host, stream=stream)
        const_indices_j_dev = cuda.to_device(pre_j_host, stream=stream)
        # ---------------------------------------

        # Metric buffers (device side)
        hist_dev = cuda.device_array(cfg.HIST_BINS, dtype=np.int32)
        poker_dev = cuda.device_array(cfg.POKER_CATEGORIES, dtype=np.int32)
        total_dev = cuda.device_array(1, dtype=np.int64)
        rises_dev = cuda.device_array(1, dtype=np.int64)
        dup_dev = cuda.device_array(1, dtype=np.int64)
        inv_hist_dev = cuda.device_array(cfg.INV_BUCKETS, dtype=np.int32)
        run_hist_dev = cuda.device_array(cfg.RUN_BUCKETS, dtype=np.int32)
        shuffled_decks_dev = cuda.device_array(total_threads * cfg.DECK_SIZE, dtype=np.int8)

        # Host arrays for results
        hist_host = np.zeros(cfg.HIST_BINS, dtype=np.int32)
        poker_host = np.zeros(cfg.POKER_CATEGORIES, dtype=np.int32)
        total_host = np.zeros(1, dtype=np.int64)
        rises_host = np.zeros(1, dtype=np.int64)
        dup_host = np.zeros(1, dtype=np.int64)
        inv_hist_host = np.zeros(cfg.INV_BUCKETS, dtype=np.int32)
        run_hist_host = np.zeros(cfg.RUN_BUCKETS, dtype=np.int32)

        metric_bufs = {
            "hist": (hist_host, hist_dev),
            "poker": (poker_host, poker_dev),
            "total": (total_host, total_dev),
            "rises": (rises_host, rises_dev),
            "dup": (dup_host, dup_dev),
            "inv_hist": (inv_hist_host, inv_hist_dev),
            "run_hist": (run_hist_host, run_hist_dev),
            "shuffled_decks": (None, shuffled_decks_dev)
        }
        
        # Consolidate device arrays needed by kernels
        kernel_consts = {
            "initial_d": initial_d_dev,
            "const_indices_i": const_indices_i_dev,
            "const_indices_j": const_indices_j_dev
        }

    except CudaAPIError as e:
        print(f"Failed to allocate CUDA device memory: {e}")
        return None

    try:
        rng_d = create_xoroshiro128p_states(total_threads, seed=int.from_bytes(os.urandom(8), 'little'))
    except CudaAPIError as e:
        print(f"Failed to create RNG states on device: {e}")
        return None
    
    stream.synchronize() # Ensure device constants are ready before returning
    return (metric_bufs, kernel_consts, rng_d, stream)

def simulation_worker_gpu(metrics_buffer: SafeMetricsBuffer, stop_evt: mp.Event):
    """GPU Worker process launching split kernels."""
    sim_init_data = None
    try:
        cuda.select_device(0)
        dev = cuda.get_current_device()
        threads_per_block = 128
        min_blocks = dev.MULTIPROCESSOR_COUNT * 4
        blocks_per_grid = max(min_blocks, (dev.MULTIPROCESSOR_COUNT * dev.MAX_THREADS_PER_MULTIPROCESSOR) // threads_per_block)
        blocks_per_grid = min(blocks_per_grid, 1024)
        total_threads = threads_per_block * blocks_per_grid

        sim_init_data = _init_gpu_buffers(total_threads)
        if sim_init_data is None:
             raise RuntimeError("Failed to initialize GPU resources.")

        (metric_bufs, kernel_consts, rng_d, stream) = sim_init_data

        # Get device pointers and host arrays from metric_bufs
        hist_dev, poker_dev, total_dev, rises_dev, dup_dev, inv_hist_dev, run_hist_dev, shuffled_decks_dev = (
            metric_bufs["hist"][1], metric_bufs["poker"][1], metric_bufs["total"][1],
            metric_bufs["rises"][1], metric_bufs["dup"][1], metric_bufs["inv_hist"][1],
            metric_bufs["run_hist"][1], metric_bufs["shuffled_decks"][1]
        )
        hist_host, poker_host, total_host, rises_host, dup_host, inv_hist_host, run_hist_host = (
             metric_bufs["hist"][0], metric_bufs["poker"][0], metric_bufs["total"][0],
             metric_bufs["rises"][0], metric_bufs["dup"][0], metric_bufs["inv_hist"][0],
             metric_bufs["run_hist"][0]
        )
        
        # Get device constants from kernel_consts
        initial_d_dev = kernel_consts["initial_d"]
        const_indices_i_dev = kernel_consts["const_indices_i"]
        const_indices_j_dev = kernel_consts["const_indices_j"]

        while not stop_evt.is_set():
             try:
                # Zero metric buffers
                stream.synchronize()
                cuda.driver.device_memset_d32(hist_dev.device_ctypes_pointer, 0, hist_dev.size)
                cuda.driver.device_memset_d32(poker_dev.device_ctypes_pointer, 0, poker_dev.size)
                cuda.driver.device_memset_d64(total_dev.device_ctypes_pointer, 0, total_dev.size)
                cuda.driver.device_memset_d64(rises_dev.device_ctypes_pointer, 0, rises_dev.size)
                cuda.driver.device_memset_d64(dup_dev.device_ctypes_pointer, 0, dup_dev.size)
                cuda.driver.device_memset_d32(inv_hist_dev.device_ctypes_pointer, 0, inv_hist_dev.size)
                cuda.driver.device_memset_d32(run_hist_dev.device_ctypes_pointer, 0, run_hist_dev.size)

                # Launch Kernels - Pass device constants
                shuffle_kernel_phase1[blocks_per_grid, threads_per_block, stream](
                    rng_d, total_dev, dup_dev, shuffled_decks_dev, 
                    initial_d_dev
                )
                analyze_kernel_phase2[blocks_per_grid, threads_per_block, stream](
                    shuffled_decks_dev, hist_dev, poker_dev, rises_dev,
                    inv_hist_dev, run_hist_dev, rng_d,
                    const_indices_i_dev, const_indices_j_dev
                )

                # Copy results back
                hist_dev.copy_to_host(hist_host, stream=stream)
                poker_dev.copy_to_host(poker_host, stream=stream)
                total_dev.copy_to_host(total_host, stream=stream)
                rises_dev.copy_to_host(rises_host, stream=stream)
                dup_dev.copy_to_host(dup_host, stream=stream)
                inv_hist_dev.copy_to_host(inv_hist_host, stream=stream)
                run_hist_dev.copy_to_host(run_hist_host, stream=stream)
                stream.synchronize()

                # Write metrics
                metrics_to_write = {
                    'hist': hist_host,
                    'total': int(total_host[0]),
                    'rises': int(rises_host[0]),
                    'poker': poker_host,
                    'inv_hist': inv_hist_host,
                    'run_hist': run_hist_host,
                    'duplicates': int(dup_host[0])
                }
                metrics_buffer.write_metrics(metrics_to_write)
                sleep(0.001)

             except CudaAPIError as e:
                print(f"GPU Error: {e}", flush=True)
                stop_evt.set(); break
             except Exception as e:
                print(f"Sim loop Error: {e}", flush=True)
                stop_evt.set(); break
    except Exception as e:
        print(f"Sim worker Error: {e}", flush=True)
        stop_evt.set()
    finally:
        print("GPU Simulation worker exiting.")

def simulation_worker_cpu(metrics_buffer: SafeMetricsBuffer, stop_evt: mp.Event):
    """CPU Worker process performing simulation in chunks."""
    try:
        rng = np.random.default_rng()
        deck = np.arange(cfg.DECK_SIZE, dtype=np.int8)
        deck5 = np.empty(5, dtype=np.int8)

        inv_bucket_width = cfg.INV_MAX // cfg.INV_BUCKETS or 1

        while not stop_evt.is_set():
            chunk_hist = np.zeros(cfg.HIST_BINS, dtype=np.int32)
            chunk_poker = np.zeros(cfg.POKER_CATEGORIES, dtype=np.int32)
            chunk_inv = np.zeros(cfg.INV_BUCKETS, dtype=np.int32)
            chunk_run = np.zeros(cfg.RUN_BUCKETS, dtype=np.int32)
            chunk_total = 0
            chunk_rises = 0
            chunk_dup = 0

            for _ in range(cfg.CPU_CHUNK_SIZE):
                np.copyto(deck, np.arange(cfg.DECK_SIZE, dtype=np.int8))
                shuffle_deck_cpu(deck, rng)
                
                simc, runs, exact_match = compute_metrics_cpu(deck, cfg.DECK_SIZE)
                np.copyto(deck5, deck[:5])
                cat = compute_poker_category_cpu(deck5)
                inversions = compute_exact_inversions_cpu(deck)
                run_hist_delta = compute_run_buckets_cpu(deck)

                chunk_hist[simc] += 1
                chunk_poker[cat] += 1
                inv_bucket = min(cfg.INV_BUCKETS - 1, inversions // inv_bucket_width)
                chunk_inv[inv_bucket] += 1
                chunk_run += run_hist_delta
                chunk_total += 1
                chunk_rises += runs
                if exact_match: chunk_dup += 1

            metrics_to_write = {
                'hist': chunk_hist,
                'total': chunk_total,
                'rises': chunk_rises,
                'poker': chunk_poker,
                'inv_hist': chunk_inv,
                'run_hist': chunk_run,
                'duplicates': chunk_dup
            }
            metrics_buffer.write_metrics(metrics_to_write)
            sleep(0.01)

    except Exception as e:
        print(f"CPU Sim worker Error: {e}", flush=True)
        stop_evt.set()
    finally:
        print("CPU Simulation worker exiting.")

def ui_worker(metrics_buffer: SafeMetricsBuffer, stop_evt: mp.Event):
    try:
        print("UI Worker: Initializing layout...", flush=True)
        layout = build_layout()
        t0 = perf_counter()
        
        def render_stats(metrics, elapsed):
            table = Table(title="Stats")
            table.add_column("Metric"); table.add_column("Value")
            total_shuffles = metrics.get('total', 0)
            rate = total_shuffles / elapsed if elapsed > 0 else 0
            table.add_row("Elapsed", f"{elapsed:.2f} s")
            table.add_row("Shuffles", f"{total_shuffles:,d}")
            table.add_row("Rate (k/s)", f"{rate / 1e3:.2f}")
            table.add_row("Duplicates", f"{metrics.get('duplicates', 0):,d}")
            return Panel(table, title="Statistics", border_style="green")
            
        def render_poker(metrics):
            poker_cats = list(POKER_CATEGORIES_MAP.values())
            poker_hist = metrics.get('poker', np.zeros(cfg.POKER_CATEGORIES, dtype=np.int64))
            total_hands = poker_hist.sum()
            table = Table(title="Poker Hands (First 5 Cards)")
            table.add_column("Category"); table.add_column("Count"); table.add_column("%")
            for i, cat in enumerate(poker_cats):
                count = poker_hist[i] if i < len(poker_hist) else 0
                percent = (count / total_hands * 100) if total_hands > 0 else 0
                table.add_row(cat, f"{int(count):,d}", f"{percent:.4f}%")
            return Panel(table, title="Poker Frequencies", border_style="magenta")
            
        def render_runs(metrics):
            run_hist = metrics.get('run_hist', np.zeros(cfg.RUN_BUCKETS, dtype=np.int64))
            total_runs = run_hist.sum(); max_val = run_hist.max() if total_runs > 0 else 1
            lines = []
            bucket_width = cfg.DECK_SIZE // cfg.RUN_BUCKETS or 1
            for i in range(cfg.RUN_BUCKETS):
                count = run_hist[i]
                bar_len = int((count / max_val) * cfg.HIST_WIDTH) if max_val > 0 else 0
                bar = '█' * bar_len
                len_start = i * bucket_width + 1; len_end = (i + 1) * bucket_width
                label = f"{len_start}-{len_end}" if bucket_width > 1 else f"{len_start}"
                lines.append(f"{label:>5s} │ {bar} [{int(count):,d}]") 
            return Panel("\n".join(lines), title=f"Run Lengths (Buckets: {cfg.RUN_BUCKETS})", border_style="yellow")
            
        def render_inv(metrics):
            inv_hist = metrics.get('inv_hist', np.zeros(cfg.INV_BUCKETS, dtype=np.int64))
            total_inv = inv_hist.sum(); max_val = inv_hist.max() if total_inv > 0 else 1
            lines = []
            bucket_width = cfg.INV_MAX // cfg.INV_BUCKETS or 1
            for i in range(cfg.INV_BUCKETS):
                count = inv_hist[i]
                bar_len = int((count / max_val) * cfg.HIST_WIDTH) if max_val > 0 else 0
                bar = '█' * bar_len
                inv_start = i * bucket_width; inv_end = (i + 1) * bucket_width -1
                label = f"{inv_start}-{inv_end}"
                lines.append(f"{label:>9s} │ {bar} [{int(count):,d}]")
            return Panel("\n".join(lines), title=f"Inversions (Buckets: {cfg.INV_BUCKETS}, Max: {cfg.INV_MAX})", border_style="blue")
            
        def render_hist(metrics):
            hist = metrics.get('hist', np.zeros(cfg.HIST_BINS, dtype=np.int64))
            total = hist.sum(); max_val = hist.max() if total > 0 else 1
            lines = []
            for i in range(cfg.HIST_BINS):
                if i >= len(hist): continue
                count = hist[i]
                bar_len = int((count / max_val) * cfg.HIST_WIDTH) if max_val > 0 else 0
                bar = '█' * bar_len
                percent = (count / total * 100) if total > 0 else 0
                lines.append(f"{i:3d} │ {bar:<{cfg.HIST_WIDTH}}│ {percent:5.1f}% ({int(count):,d})")
            return "\n".join(lines)
        
        print("UI Worker: Setting up initial layout...", flush=True)
        # Initial layout setup
        layout["hist"].update(Panel(render_hist({}), title="Positional Similarity", border_style="cyan"))
        layout["side"]["stats"].update(render_stats({}, 0))
        layout["side"]["poker"].update(render_poker({}))
        layout["side"]["runs"].update(render_runs({}))
        layout["side"]["inv"].update(render_inv({}))
        
        print("UI Worker: Configuring terminal...", flush=True)
        from rich.console import Console
        from rich.terminal_theme import TerminalTheme
        from rich.theme import Theme
        
        # Create a custom theme that's more compatible
        theme = Theme({
            "info": "dim cyan",
            "warning": "yellow",
            "danger": "bold red"
        })
        
        # Configure console with explicit settings
        console = Console(
            theme=theme,
            force_terminal=True,
            force_interactive=True,
            color_system="auto",
            width=None,  # Let it auto-detect
            height=None,  # Let it auto-detect
            record=True
        )
        
        print("UI Worker: Terminal configuration complete", flush=True)
        print(f"Terminal size: {console.size}", flush=True)
        print(f"Terminal is interactive: {console.is_interactive}", flush=True)
        print(f"Terminal supports color: {console.color_system}", flush=True)
        
        print("UI Worker: Entering Live context...", flush=True)
        try:
            # First try without screen mode
            with Live(
                layout,
                refresh_per_second=cfg.UI_FPS,
                screen=False,  # Changed from True to False
                transient=False,  # Changed from True to False
                console=console,
                auto_refresh=True,
                vertical_overflow="visible"
            ) as live:
                print("UI Worker: Live context entered successfully", flush=True)
                # Initialize accumulator with correct dtypes matching MetricsDataStruct
                total_metrics = { 
                    'hist': np.zeros(cfg.HIST_BINS, dtype=np.int32),
                    'poker': np.zeros(cfg.POKER_CATEGORIES, dtype=np.int32),
                    'run_hist': np.zeros(cfg.RUN_BUCKETS, dtype=np.int32),
                    'inv_hist': np.zeros(cfg.INV_BUCKETS, dtype=np.int32),
                    'total': np.int64(0),
                    'rises': np.int64(0),
                    'duplicates': np.int64(0)
                }
                last_update_time = t0
                print("UI Worker: Starting main loop...", flush=True)
                while not stop_evt.is_set():
                    try:
                        metrics_batch = metrics_buffer.read_metrics()
                        if metrics_batch is None:
                            sleep(0.01)
                            continue
                        
                        print("UI Worker: Updating metrics...", flush=True)
                        # Accumulate metrics - Ensure types are compatible
                        total_metrics['hist'] += metrics_batch['hist'].astype(np.int32)
                        total_metrics['poker'] += metrics_batch['poker'].astype(np.int32)
                        total_metrics['run_hist'] += metrics_batch['run_hist'].astype(np.int32)
                        total_metrics['inv_hist'] += metrics_batch['inv_hist'].astype(np.int32)
                        total_metrics['total'] += np.int64(metrics_batch['total'])
                        total_metrics['rises'] += np.int64(metrics_batch['rises'])
                        total_metrics['duplicates'] += np.int64(metrics_batch['duplicates'])
                        
                        current_time = perf_counter()
                        elapsed_total = current_time - t0
                        
                        print("UI Worker: Updating layout...", flush=True)
                        # Update layout components 
                        layout["hist"].update(Panel(render_hist(total_metrics), title="Positional Similarity", border_style="cyan"))
                        layout["side"]["stats"].update(render_stats(total_metrics, elapsed_total))
                        layout["side"]["poker"].update(render_poker(total_metrics))
                        layout["side"]["runs"].update(render_runs(total_metrics))
                        layout["side"]["inv"].update(render_inv(total_metrics))
                        
                        last_update_time = current_time
                    except Exception as e:
                        print(f"UI Worker: Error in main loop: {e}", flush=True)
                        import traceback
                        traceback.print_exc()
                        break
        except Exception as e:
            print(f"UI Worker: Error in Live context: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise
                    
    except KeyboardInterrupt:
        print("\nUI Worker Interrupted.", flush=True)
    except Exception as e:
        print(f"\nUI error: {e}", flush=True)
        import traceback
        traceback.print_exc()
    finally:
        print("UI Worker exiting.")
        stop_evt.set()

def main():
    print("🚀 Starting Dashboard with Metrics — Ctrl‑C to quit", flush=True)

    stop_evt = None
    safe_buffer_instance = None
    sim_proc = None
    ui_proc = None
    is_shm_creator = False

    try:
        # Determine execution mode
        try:
            use_gpu = cuda.is_available()
            if use_gpu:
                cuda.select_device(0) # Check if context can be created
        except Exception as e:
            print(f"CUDA Check failed: {e}")
            use_gpu = False
            
        mode = "GPU" if use_gpu else "CPU"
        print(f"Using {mode} mode for simulation", flush=True)

        # Create shared buffer
        is_shm_creator = True
        safe_buffer_instance = SafeMetricsBuffer(create=True, name=SHM_NAME)
        stop_evt = mp.Event()

        # Select target worker function based on mode
        if mode == "GPU":
            target_worker = simulation_worker_gpu_process_wrapper
            print("Targeting GPU worker...")
        else:
            target_worker = cpu_simulation_worker_process_wrapper
            print("Targeting CPU worker...")

        # Start simulation and UI processes
        sim_proc = mp.Process(target=target_worker, args=(SHM_NAME, stop_evt), daemon=True)
        ui_proc = mp.Process(target=ui_worker_process_wrapper, args=(SHM_NAME, stop_evt), daemon=True)

        print("Starting simulation process...")
        sim_proc.start()
        print("Starting UI process...")
        ui_proc.start()

        ui_proc.join()
        print("UI process finished.")

    except KeyboardInterrupt:
        print("\nCtrl+C detected, shutting down...", flush=True)
    except Exception as e:
        print(f"\nMain process error: {e}", flush=True)
        import traceback
        traceback.print_exc()
    finally:
        print("Initiating cleanup...")
        if stop_evt:
            print("Signaling processes to stop...")
            stop_evt.set()
        sleep(0.5)
        if sim_proc and sim_proc.is_alive():
            print("Terminating simulation process...")
            sim_proc.terminate(); sim_proc.join(timeout=1)
        if ui_proc and ui_proc.is_alive():
            print("Terminating UI process...")
            ui_proc.terminate(); ui_proc.join(timeout=1)
        if is_shm_creator and safe_buffer_instance:
            print(f"Cleaning up shared memory '{SHM_NAME}'...")
            safe_buffer_instance.unlink()
            safe_buffer_instance = None
            print("Shared memory cleaned up.")
        elif safe_buffer_instance:
             safe_buffer_instance.close()
             safe_buffer_instance = None
        print("Cleanup complete. Exiting.")

# Wrapper for GPU simulation process
def simulation_worker_gpu_process_wrapper(shm_name, stop_evt):
    buffer_instance = None
    try:
        buffer_instance = SafeMetricsBuffer(create=False, name=shm_name)
        simulation_worker_gpu(buffer_instance, stop_evt)
    except Exception as e:
        print(f"[GPU Sim Wrapper] Error: {e}")
        stop_evt.set()
    finally:
        if buffer_instance: buffer_instance.close()
        print("[GPU Sim Wrapper] Exiting.")

# Wrapper for CPU simulation process
def cpu_simulation_worker_process_wrapper(shm_name, stop_evt):
    buffer_instance = None
    try:
        buffer_instance = SafeMetricsBuffer(create=False, name=shm_name)
        simulation_worker_cpu(buffer_instance, stop_evt)
    except Exception as e:
        print(f"[CPU Sim Wrapper] Error: {e}")
        stop_evt.set()
    finally:
        if buffer_instance: buffer_instance.close()
        print("[CPU Sim Wrapper] Exiting.")

# Wrapper for UI process
def ui_worker_process_wrapper(shm_name, stop_evt):
    buffer_instance = None
    try:
        buffer_instance = SafeMetricsBuffer(create=False, name=shm_name)
        ui_worker(buffer_instance, stop_evt)
    except Exception as e:
        print(f"[UI Wrapper] Error: {e}")
        stop_evt.set()
    finally:
        if buffer_instance: buffer_instance.close()
        print("[UI Wrapper] Exiting.")

if __name__ == "__main__":
    if sys.platform.startswith('win'):
        mp.freeze_support()
    main()
