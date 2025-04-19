#!/usr/bin/env python3
"""
High-performance shuffle+detect dashboard with metrics:
  • Combined single-pass shuffle and analysis
  • GPU-accelerated poker hand evaluation
  • Real-time metrics visualization
  • Process-safe shared memory communication
"""

import os
import sys
import signal
from typing import ClassVar
import numpy as np
from numba import cuda, njit
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float64
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
from time import perf_counter, sleep
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.console import Console
from dataclasses import dataclass

# Configuration
@dataclass(frozen=True)
class Config:
    """Dashboard configuration parameters."""
    DECK_SIZE: ClassVar[int] = 52
    HIST_BINS: ClassVar[int] = 53  # 0 to 52 matches inclusive
    POKER_CATEGORIES: ClassVar[int] = 10
    HIST_WIDTH: ClassVar[int] = 50
    UI_FPS: ClassVar[int] = 10
    CPU_CHUNK_SIZE: ClassVar[int] = 1000
    MAX_THREADS: ClassVar[int] = 1024  # Maximum concurrent GPU threads

cfg = Config()

# Shared memory configuration
SHM_NAME = f"shuffle_metrics_{os.getpid()}"

# Poker hand categories
POKER_CATEGORIES_MAP = {
    1: "One Pair", 2: "Two Pair", 3: "3 of a Kind", 4: "Straight",
    5: "Flush", 6: "Full House", 7: "4 of a Kind", 8: "Straight Flush", 
    9: "Royal Flush"
}

class MetricsDataStruct:
    """Shared memory data structure for metrics.
    
    Contains:
    - Similarity histogram (52 bins)
    - Poker hand frequencies (10 categories)
    - Total shuffles counter
    - Rising sequences counter
    - Duplicate shuffles counter
    """
    def __init__(self):
        self.hist = np.zeros(cfg.HIST_BINS, dtype=np.int32)
        self.poker = np.zeros(cfg.POKER_CATEGORIES, dtype=np.int32)
        self.total = np.int64(0)
        self.rises = np.int64(0)
        self.duplicates = np.int64(0)
        
    @property
    def nbytes(self):
        """Total bytes needed for all metrics."""
        return (
            self.hist.nbytes +
            self.poker.nbytes +
            self.total.nbytes +
            self.rises.nbytes +
            self.duplicates.nbytes
        )

class SafeMetricsBuffer:
    """Thread-safe metrics buffer using shared memory and structured arrays."""
    
    # Define the metrics dtype
    metrics_dtype = np.dtype([
        ('hist', np.int32, (cfg.HIST_BINS,)),
        ('poker', np.int32, (cfg.POKER_CATEGORIES,)),
        ('total', np.int64),
        ('rises', np.int64),
        ('duplicates', np.int64),
        ('sequence', np.int64)  # Control sequence number
    ])
    
    def __init__(self, create=False, name=SHM_NAME):
        """Initialize the metrics buffer."""
        self.name = name
        
        if create:
            self.shm = SharedMemory(
                name=name,
                create=True,
                size=self.metrics_dtype.itemsize
            )
            # Initialize the structured array
            self.metrics = np.ndarray(
                shape=(),
                dtype=self.metrics_dtype,
                buffer=self.shm.buf
            )
            self.metrics.fill(0)
        else:
            self.shm = SharedMemory(name=name)
            self.metrics = np.ndarray(
                shape=(),
                dtype=self.metrics_dtype,
                buffer=self.shm.buf
            )
    
    def write_metrics(self, metrics_dict):
        """Write metrics to the buffer."""
        # Copy metrics
        self.metrics['hist'] = metrics_dict['hist']
        self.metrics['poker'] = metrics_dict['poker']
        self.metrics['total'] = metrics_dict['total']
        self.metrics['rises'] = metrics_dict['rises']
        self.metrics['duplicates'] = metrics_dict['duplicates']
        # Update sequence number with wrap-around
        self.metrics['sequence'] = (self.metrics['sequence'] + 1) & 0x7FFFFFFFFFFFFFFF
    
    def read_metrics(self):
        """Read metrics from the buffer."""
        # Check sequence number
        seq = self.metrics['sequence']
        if seq == getattr(self, '_last_seq', -1):
            return None
            
        # Copy metrics to dict
        metrics = {
            'hist': self.metrics['hist'].copy(),
            'poker': self.metrics['poker'].copy(),
            'total': int(self.metrics['total']),
            'rises': int(self.metrics['rises']),
            'duplicates': int(self.metrics['duplicates'])
        }
        
        self._last_seq = seq
        return metrics
    
    def close(self):
        """Close the shared memory segment."""
        self.shm.close()
    
    def unlink(self):
        """Unlink the shared memory segment."""
        try:
            self.shm.unlink()
        except Exception:
            pass

@contextlib.contextmanager
def metrics_shm(name):
    """Context manager for shared memory buffer."""
    buf = None
    try:
        buf = SafeMetricsBuffer(create=True, name=name)
        yield buf
    finally:
        if buf is not None:
            buf.close()
            buf.unlink()

# ─────────── Simulation Code ────────────────────────────────────────────

@cuda.jit
def shuffle_kernel(deck_in, deck_out, hist, poker, num_shuffles, rng_states):
    """Combined shuffle and analyze kernel."""
    tx = cuda.threadIdx.x
    bdx = cuda.blockDim.x
    bid = cuda.blockIdx.x
    gid = bid * bdx + tx
    
    if gid >= num_shuffles:
        return
        
    # Local arrays
    deck = cuda.local.array(cfg.DECK_SIZE, dtype=numba.int32)
    deck5 = cuda.local.array(5, dtype=numba.int32)
    
    # Copy input deck
    for i in range(cfg.DECK_SIZE):
        deck[i] = deck_in[i]
    
    # Shuffle (Fisher-Yates)
    for i in range(cfg.DECK_SIZE - 1, 0, -1):
        j = int(xoroshiro128p_uniform_float64(rng_states, gid) * (i + 1))
        if j > i: j = i
        deck[i], deck[j] = deck[j], deck[i]
    
    # Copy to output buffer
    for i in range(cfg.DECK_SIZE):
        deck_out[gid * cfg.DECK_SIZE + i] = deck[i]
    
    # Analyze shuffle
    simc = 0
    runs = 1
    prev_card = deck[0]
    
    for k in range(cfg.DECK_SIZE):
        c = deck[k]
        simc += (c == k)
        if k > 0:
            runs += (c < prev_card)
        if k < 5:
            deck5[k] = c
    
    # Update metrics atomically
    cuda.atomic.add(hist, simc, 1)
    
    # Poker hand analysis
    cat = compute_poker_category_device(deck5)
    cuda.atomic.add(poker, cat, 1)

class BaseSimulator:
    """Base class for shuffle simulators."""
    
    def __init__(self, deck_size=cfg.DECK_SIZE):
        self.deck_size = deck_size
        
    def run(self, num_shuffles):
        """Run simulation for specified number of shuffles."""
        raise NotImplementedError
        
    def close(self):
        """Clean up resources."""
        pass

class CPUSimulator(BaseSimulator):
    """CPU-based shuffle simulation."""
    
    def __init__(self, deck_size=cfg.DECK_SIZE):
        super().__init__(deck_size)
        self.deck = np.arange(deck_size, dtype=np.int32)
        self.hist = np.zeros(cfg.HIST_BINS, dtype=np.int32)
        self.poker = np.zeros(cfg.POKER_CATEGORIES, dtype=np.int32)
        
    def run(self, num_shuffles):
        """Run simulation for specified number of shuffles."""
        # Reset metrics
        self.hist.fill(0)
        self.poker.fill(0)
        rises = 0
        duplicates = 0
        
        # Run shuffles in chunks
        for i in range(0, num_shuffles, cfg.CPU_CHUNK_SIZE):
            chunk_size = min(cfg.CPU_CHUNK_SIZE, num_shuffles - i)
            chunk_metrics = simulate_cpu_chunk(
                self.deck,
                chunk_size,
                self.hist,
                self.poker
            )
            rises += chunk_metrics['rises']
            duplicates += chunk_metrics['duplicates']
            
        return {
            'hist': self.hist,
            'poker': self.poker,
            'total': num_shuffles,
            'rises': rises,
            'duplicates': duplicates
        }

class GPUSimulator(BaseSimulator):
    """GPU-based shuffle simulation."""
    
    def __init__(self, deck_size=cfg.DECK_SIZE):
        super().__init__(deck_size)
        self.stream = cuda.stream()
        
        # Configure kernel launch parameters
        self.threadsperblock = 256
        self.max_blocks = (cfg.MAX_THREADS + self.threadsperblock - 1) // self.threadsperblock
        self.max_shuffles = self.max_blocks * self.threadsperblock
        
        # Allocate fixed-size buffers
        self.d_deck = cuda.device_array((deck_size,), dtype=np.int32, stream=self.stream)
        self.d_hist = cuda.device_array((cfg.HIST_BINS,), dtype=np.int32, stream=self.stream)
        self.d_poker = cuda.device_array((cfg.POKER_CATEGORIES,), dtype=np.int32, stream=self.stream)
        
        # Create RNG states
        self.rng_states = create_xoroshiro128p_states(
            self.max_shuffles, 
            seed=np.random.randint(2**32)
        )
        
        # Host buffers for results
        self.h_hist = np.zeros(cfg.HIST_BINS, dtype=np.int32)
        self.h_poker = np.zeros(cfg.POKER_CATEGORIES, dtype=np.int32)
        
        # Initialize deck
        h_deck = np.arange(deck_size, dtype=np.int32)
        self.d_deck.copy_to_device(h_deck, stream=self.stream)
        
    def run(self, num_shuffles):
        """Run simulation for specified number of shuffles."""
        # Limit number of shuffles to maximum GPU capacity
        num_shuffles = min(num_shuffles, self.max_shuffles)
        blockspergrid = (num_shuffles + self.threadsperblock - 1) // self.threadsperblock
        
        # Reset metrics
        self.d_hist.copy_to_device(np.zeros_like(self.h_hist), stream=self.stream)
        self.d_poker.copy_to_device(np.zeros_like(self.h_poker), stream=self.stream)
        
        # Allocate temporary buffer for shuffled decks
        d_shuffled = cuda.device_array((num_shuffles * self.deck_size,), dtype=np.int32, stream=self.stream)
        
        # Launch kernel
        shuffle_kernel[blockspergrid, self.threadsperblock, self.stream](
            self.d_deck,
            d_shuffled,
            self.d_hist,
            self.d_poker,
            num_shuffles,
            self.rng_states
        )
        
        # Copy results back to host
        self.d_hist.copy_to_host(self.h_hist, stream=self.stream)
        self.d_poker.copy_to_host(self.h_poker, stream=self.stream)
        self.stream.synchronize()
        
        # Free temporary buffer
        d_shuffled.free()
        
        return {
            'hist': self.h_hist,
            'poker': self.h_poker,
            'total': num_shuffles,
            'rises': 0,  # Not tracked in GPU version
            'duplicates': 0  # Not tracked in GPU version
        }
        
    def close(self):
        """Free GPU resources."""
        self.stream.synchronize()
        # Free device arrays
        self.d_deck.free()
        self.d_hist.free()
        self.d_poker.free()
        # Free RNG states
        cuda.random.free_rng_states(self.rng_states)
        # Free stream
        self.stream.free()

@njit(nogil=True, cache=True)
def shuffle_deck_cpu(deck, rng_state):
    """Fisher-Yates shuffle for CPU (Numba JIT)."""
    n = len(deck)
    for i in range(n - 1, 0, -1):
        j = rng_state.integers(0, i + 1)
        deck[i], deck[j] = deck[j], deck[i]

@njit(nogil=True, cache=True)
def compute_metrics_cpu(deck, deck_size):
    """Calculate similarity and runs for a deck (CPU JIT)."""
    simc = 0
    runs = 1
    prev_card = deck[0]
    exact_match = True
    initial_val = np.int32(0)
    
    for k in range(deck_size):
        c = deck[k]
        simc += (c == k)
        exact_match &= (c == initial_val)
        initial_val += 1
        if k > 0:
            runs += (c < prev_card)
        prev_card = c
        
    return simc, runs, exact_match

@njit(nogil=True, cache=True)
def simulate_cpu_chunk(deck, chunk_size, hist, poker):
    """Run a chunk of CPU simulations."""
    rises = 0
    duplicates = 0
    deck5 = np.empty(5, dtype=np.int32)
    rng = np.random.default_rng()
    
    for _ in range(chunk_size):
        # Reset and shuffle deck
        np.copyto(deck, np.arange(len(deck), dtype=np.int32))
        shuffle_deck_cpu(deck, rng)
        
        # Compute metrics
        simc, run_count, exact = compute_metrics_cpu(deck, len(deck))
        np.copyto(deck5, deck[:5])
        cat = compute_poker_category_cpu(deck5)
        
        # Update metrics (ensure simc is in bounds)
        if 0 <= simc < cfg.HIST_BINS:
            hist[simc] += 1
        if 0 <= cat < cfg.POKER_CATEGORIES:
            poker[cat] += 1
            
        rises += run_count
        duplicates += exact
        
    return {
        'rises': rises,
        'duplicates': duplicates
    }

def check_gpu_available():
    """Check if CUDA GPU is available."""
    try:
        cuda.select_device(0)
        return True
    except Exception:
        return False

def simulation_worker(use_gpu, shm_name, stop_event):
    """Worker process for running simulations."""
    simulator = None
    metrics_buffer = None
    
    try:
        # Auto-detect GPU if not explicitly specified
        if use_gpu is None:
            use_gpu = check_gpu_available()
            
        # Initialize simulator
        try:
            simulator = GPUSimulator() if use_gpu else CPUSimulator()
            print(f"Starting {'GPU' if use_gpu else 'CPU'} simulation worker")
        except Exception as e:
            print(f"GPU initialization failed: {e}")
            print("Falling back to CPU simulation")
            simulator = CPUSimulator()
            
        metrics_buffer = SafeMetricsBuffer(create=False, name=shm_name)
        
        while not stop_event.is_set():
            # Run simulation chunk
            metrics = simulator.run(cfg.CPU_CHUNK_SIZE)
            metrics_buffer.write_metrics(metrics)
            sleep(1.0 / cfg.UI_FPS)  # Match UI update rate
            
    except Exception as e:
        print(f"Simulation worker error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if simulator is not None:
            simulator.close()
        if metrics_buffer is not None:
            metrics_buffer.close()

# ─────────── UI Code ────────────────────────────────────────────────

class DashboardRenderer:
    """Rich dashboard renderer."""
    
    def __init__(self):
        self.console = Console()
        self.layout = Layout()
        self.setup_layout()
        
        # Initialize with zeros
        self.initial_metrics = {
            'hist': np.zeros(cfg.HIST_BINS, dtype=np.int32),
            'poker': np.zeros(cfg.POKER_CATEGORIES, dtype=np.int32),
            'total': 0,
            'rises': 0,
            'duplicates': 0
        }
        self.update(self.initial_metrics)
        
    def setup_layout(self):
        """Initialize dashboard layout."""
        self.layout.split(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3)
        )
        
        self.layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        self.layout["left"].split(
            Layout(name="histogram", ratio=2),
            Layout(name="poker")
        )
        
        self.layout["right"].split(
            Layout(name="stats"),
            Layout(name="info")
        )
        
    def _make_panel(self, title, rows):
        """Create a panel with a table of rows."""
        table = Table.grid()
        table.add_column(justify="right")
        for row in rows:
            table.add_row(row)
        return Panel(table, title=title)
        
    def _make_histogram_rows(self, hist_data):
        """Generate histogram rows."""
        max_count = max(hist_data)
        return [
            f"{i:2d} | {'█' * int(count * cfg.HIST_WIDTH / max_count) if max_count > 0 else ''}"
            for i, count in enumerate(hist_data)
        ]
        
    def _make_poker_rows(self, poker_data):
        """Generate poker hand rows."""
        total = sum(poker_data)
        return [
            f"{POKER_CATEGORIES_MAP.get(i, 'Invalid')}: {count:,d} ({count * 100.0 / total:.1f}%)"
            for i, count in enumerate(poker_data)
            if total > 0
        ]
        
    def _make_stats_rows(self, hist_data):
        """Generate statistics rows."""
        return [
            f"Mean: {np.mean(hist_data):.2f}",
            f"Std Dev: {np.std(hist_data):.2f}",
            f"Min: {np.min(hist_data)}",
            f"Max: {np.max(hist_data)}"
        ]
        
    def update(self, metrics):
        """Update all dashboard components with new metrics."""
        # Header
        header_text = (
            f"Total Shuffles: {metrics['total']:,d} | "
            f"Rising Sequences: {metrics['rises']:,d} | "
            f"Duplicates: {metrics['duplicates']:,d}"
        )
        self.layout["header"].update(Panel(header_text, title="Shuffle Dashboard"))
        
        # Main panels
        self.layout["histogram"].update(
            self._make_panel("Similarity Distribution", self._make_histogram_rows(metrics['hist']))
        )
        self.layout["poker"].update(
            self._make_panel("Poker Hand Distribution", self._make_poker_rows(metrics['poker']))
        )
        self.layout["stats"].update(
            self._make_panel("Statistics", self._make_stats_rows(metrics['hist']))
        )
        self.layout["info"].update(
            Panel("Press Ctrl+C to exit\nGPU mode available", title="Info")
        )

def ui_worker(shm_name, stop_event):
    """Worker process for UI updates."""
    try:
        print("Starting UI worker")
        metrics_buffer = SafeMetricsBuffer(create=False, name=shm_name)
        dashboard = DashboardRenderer()
        
        # Wait for initial metrics with timeout
        start = perf_counter()
        while perf_counter() - start < 2:
            metrics = metrics_buffer.read_metrics()
            if metrics is not None:
                break
            if stop_event.is_set():
                return
            sleep(0.1)
        
        with Live(dashboard.layout, refresh_per_second=cfg.UI_FPS) as live:
            while not stop_event.is_set():
                new_metrics = metrics_buffer.read_metrics()
                if new_metrics:
                    dashboard.update(new_metrics)
                sleep(1.0 / cfg.UI_FPS)
                
    except Exception as e:
        print(f"UI worker error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if 'metrics_buffer' in locals():
            metrics_buffer.close()

def _poker_logic(ranks, suits, rank_counts):
    """Core poker hand evaluation logic shared between CPU and GPU."""
    # Sort ranks in-place (insertion sort)
    for i in range(1, 5):
        j = i
        while j > 0 and ranks[j-1] > ranks[j]:
            ranks[j-1], ranks[j] = ranks[j], ranks[j-1]
            suits[j-1], suits[j] = suits[j], suits[j-1]
            j -= 1
    
    # Count rank frequencies
    for i in range(5):
        rank_counts[ranks[i]] += 1
        
    # Check for flush
    flush = True
    for i in range(1, 5):
        if suits[i] != suits[0]:
            flush = False
            break
            
    # Check for straight
    straight = True
    for i in range(1, 5):
        if ranks[i] != ranks[i-1] + 1:
            straight = False
            break
            
    # Special case: Ace-low straight
    if not straight and ranks[0] == 0 and ranks[1] == 9 and ranks[2] == 10 and ranks[3] == 11 and ranks[4] == 12:
        straight = True
        
    # Find highest count
    max_count = 0
    second_count = 0
    for i in range(13):
        if rank_counts[i] > max_count:
            second_count = max_count
            max_count = rank_counts[i]
        elif rank_counts[i] > second_count:
            second_count = rank_counts[i]
            
    # Determine hand category
    if straight and flush:
        if ranks[4] == 12 and ranks[0] == 8:  # Royal flush
            return 9
        return 8  # Straight flush
    elif max_count == 4:
        return 7  # Four of a kind
    elif max_count == 3 and second_count == 2:
        return 6  # Full house
    elif flush:
        return 5  # Flush
    elif straight:
        return 4  # Straight
    elif max_count == 3:
        return 3  # Three of a kind
    elif max_count == 2 and second_count == 2:
        return 2  # Two pair
    elif max_count == 2:
        return 1  # One pair
    return 0  # High card

@njit(nogil=True, cache=True)
def compute_poker_category_cpu(deck5):
    """CPU version of poker hand evaluation."""
    ranks = np.empty(5, dtype=np.int32)
    suits = np.empty(5, dtype=np.int32)
    rank_counts = np.zeros(13, dtype=np.int32)
    
    # Extract ranks and suits
    for i in range(5):
        ranks[i] = deck5[i] % 13
        suits[i] = deck5[i] // 13
        
    return _poker_logic(ranks, suits, rank_counts)

@cuda.jit(device=True, fastmath=True)
def compute_poker_category_device(deck5):
    """GPU version of poker hand evaluation."""
    ranks = cuda.local.array(5, numba.int32)
    suits = cuda.local.array(5, numba.int32)
    rank_counts = cuda.local.array(13, numba.int32)
    
    # Initialize rank_counts to zero
    for i in range(13):
        rank_counts[i] = 0
        
    # Extract ranks and suits
    for i in range(5):
        ranks[i] = deck5[i] % 13
        suits[i] = deck5[i] // 13
        
    return _poker_logic(ranks, suits, rank_counts)

def main():
    """Main entry point."""
    # Set up signal handling
    stop_event = mp.Event()
    
    def signal_handler(signum, frame):
        print("\nShutting down gracefully...")
        stop_event.set()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Create shared memory buffer
        with metrics_shm(SHM_NAME) as metrics_buffer:
            # Start simulation worker
            use_gpu = None  # Auto-detect
            if '--gpu' in sys.argv:
                use_gpu = True
            elif '--cpu' in sys.argv:
                use_gpu = False
                
            sim_proc = mp.Process(target=simulation_worker, args=(use_gpu, SHM_NAME, stop_event))
            sim_proc.start()
            print("Started simulation process")
            
            # Wait for simulator to initialize
            sleep(2)
            
            # Start UI worker
            ui_proc = mp.Process(target=ui_worker, args=(SHM_NAME, stop_event))
            ui_proc.start()
            print("Started UI process")
            
            # Wait for processes to finish or stop event
            while not stop_event.is_set() and sim_proc.is_alive() and ui_proc.is_alive():
                sleep(0.1)
                
            # Clean up processes
            if sim_proc.is_alive():
                sim_proc.terminate()
            if ui_proc.is_alive():
                ui_proc.terminate()
                
            sim_proc.join()
            ui_proc.join()
            
    except Exception as e:
        print(f"Main process error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    # Run main
    main()
