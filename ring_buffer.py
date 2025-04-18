#!/usr/bin/env python3
"""
ring_buffer.py

Lockless ring buffer implementation using mmap for inter-process communication.
"""

import mmap
import os
import struct
import numpy as np
from typing import Tuple, Optional
import ctypes
from ctypes import c_int64, c_uint64, c_uint8, Structure, c_int32
import tempfile

# Define the metrics struct layout
class MetricsStruct(Structure):
    _fields_ = [
        ('head', c_int64),
        ('tail', c_int64),
        ('hist_data', c_int32 * 101),  # HIST_BINS
        ('poker_data', c_int32 * 8),   # POKER_CATS_PER_U32
        ('run_length_data', c_int32 * 16),  # RUN_BUCKETS
        ('inversion_data', c_int32 * 64),   # INV_BUCKETS
        ('scalar_metrics', c_int64 * 3)     # total, rises, dup
    ]

# Calculate total size of metrics struct
METRICS_SIZE = ctypes.sizeof(MetricsStruct)
RING_SIZE = METRICS_SIZE * 2  # Double buffering

class RingBuffer:
    """Lockless ring buffer for inter-process communication using mmap."""
    
    def __init__(self, name: str, size: int, dtype: np.dtype):
        """Initialize a ring buffer with the given name and size.
        
        Args:
            name: Unique identifier for the shared memory region
            size: Number of elements in the buffer
            dtype: NumPy dtype for the buffer elements
        """
        self.name = name
        self.size = size
        self.dtype = dtype
        self.element_size = dtype.itemsize
        
        # Calculate total size including header (2 uint64s for head and tail)
        self.total_size = 16 + size * self.element_size
        
        # Create or open shared memory region
        self.fd = os.open(f"/dev/shm/{name}", os.O_CREAT | os.O_RDWR)
        os.ftruncate(self.fd, self.total_size)
        
        # Map the memory region
        self.mmap = mmap.mmap(self.fd, self.total_size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        
        # Initialize header if this is the first process
        if self.mmap.read(1) == b'\x00':
            self.mmap.seek(0)
            self.mmap.write(struct.pack('QQ', 0, 0))  # head = 0, tail = 0
        
        # Create NumPy views for header and data
        self.mmap.seek(0)
        self.head = np.frombuffer(self.mmap.read(8), dtype=np.uint64)[0]
        self.tail = np.frombuffer(self.mmap.read(8), dtype=np.uint64)[0]
        self.data = np.frombuffer(self.mmap.read(size * self.element_size), dtype=dtype)
    
    def write(self, data: np.ndarray) -> bool:
        """Write data to the ring buffer.
        
        Args:
            data: NumPy array to write
            
        Returns:
            bool: True if write was successful, False if buffer was full
        """
        if len(data) > self.size:
            raise ValueError(f"Data size {len(data)} exceeds buffer size {self.size}")
        
        # Calculate available space
        head = np.frombuffer(self.mmap[0:8], dtype=np.uint64)[0]
        tail = np.frombuffer(self.mmap[8:16], dtype=np.uint64)[0]
        available = self.size - ((head - tail) % self.size)
        
        if available < len(data):
            return False
        
        # Write data
        start = head % self.size
        end = (head + len(data)) % self.size
        
        if end > start:
            self.data[start:end] = data
        else:
            self.data[start:] = data[:self.size - start]
            self.data[:end] = data[self.size - start:]
        
        # Update head atomically
        new_head = (head + len(data)) % self.size
        self.mmap[0:8] = np.array([new_head], dtype=np.uint64).tobytes()
        
        return True
    
    def read(self) -> Optional[np.ndarray]:
        """Read the latest data from the ring buffer.
        
        Returns:
            Optional[np.ndarray]: Latest data if available, None if buffer is empty
        """
        head = np.frombuffer(self.mmap[0:8], dtype=np.uint64)[0]
        tail = np.frombuffer(self.mmap[8:16], dtype=np.uint64)[0]
        
        if head == tail:
            return None
        
        # Calculate size of latest data
        size = (head - tail) % self.size
        if size == 0:
            return None
        
        # Read data
        start = tail % self.size
        end = (tail + size) % self.size
        
        if end > start:
            data = self.data[start:end].copy()
        else:
            data = np.empty(size, dtype=self.dtype)
            data[:self.size - start] = self.data[start:]
            data[self.size - start:] = self.data[:end]
        
        # Update tail atomically
        new_tail = (tail + size) % self.size
        self.mmap[8:16] = np.array([new_tail], dtype=np.uint64).tobytes()
        
        return data
    
    def close(self):
        """Close the ring buffer and clean up resources."""
        self.mmap.close()
        os.close(self.fd)
        try:
            os.unlink(f"/dev/shm/{self.name}")
        except FileNotFoundError:
            pass

class MetricsBuffer:
    """Ring buffer for simulation metrics."""
    
    def __init__(self, name, use_windows=False):
        self.use_windows = use_windows
        self.name = name
        self.size = RING_SIZE
        
        if use_windows:
            # Create a temporary file for Windows shared memory
            self.temp_file = tempfile.NamedTemporaryFile(delete=False)
            self.temp_file.truncate(self.size)
            self.temp_file.close()
            
            # Map the file into memory
            self.mmap = mmap.mmap(-1, self.size, tagname=name)
        else:
            # Linux shared memory
            self.mmap = mmap.mmap(-1, self.size, tagname=f"/dev/shm/{name}")
        
        # Initialize the buffer with zeros
        self.mmap.write(b'\x00' * self.size)
        self.mmap.seek(0)
        
        # Create numpy views for each field
        self.buffer = np.frombuffer(self.mmap, dtype=np.uint8)
        self.metrics = np.frombuffer(self.buffer, dtype=np.uint8).view(MetricsStruct)
        
        # Initialize head and tail
        self.metrics[0].head = 0
        self.metrics[0].tail = 0
        self.metrics[1].head = 0
        self.metrics[1].tail = 0
        
        # Ensure proper memory ordering
        os.fsync(self.mmap.fileno())

    def write_metrics(self, metrics):
        """Write metrics to the ring buffer."""
        # Get current buffer index
        idx = self.metrics[0].head & 1
        
        # Copy metrics data
        np.copyto(self.metrics[idx].hist_data, metrics['hist'])
        np.copyto(self.metrics[idx].poker_data, metrics['poker'])
        np.copyto(self.metrics[idx].run_length_data, metrics['run_hist'])
        np.copyto(self.metrics[idx].inversion_data, metrics['inv_hist'])
        self.metrics[idx].scalar_metrics[0] = metrics['total']
        self.metrics[idx].scalar_metrics[1] = metrics['rises']
        self.metrics[idx].scalar_metrics[2] = metrics['duplicates']
        
        # Update head with release fence
        self.metrics[0].head += 1
        os.fsync(self.mmap.fileno())

    def read_metrics(self):
        """Read metrics from the ring buffer."""
        # Get current buffer index
        idx = self.metrics[0].tail & 1
        
        # Read metrics data
        metrics = {
            'hist': np.array(self.metrics[idx].hist_data),
            'poker': np.array(self.metrics[idx].poker_data),
            'run_hist': np.array(self.metrics[idx].run_length_data),
            'inv_hist': np.array(self.metrics[idx].inversion_data),
            'total': self.metrics[idx].scalar_metrics[0],
            'rises': self.metrics[idx].scalar_metrics[1],
            'duplicates': self.metrics[idx].scalar_metrics[2]
        }
        
        # Update tail with acquire fence
        self.metrics[0].tail += 1
        os.fsync(self.mmap.fileno())
        
        return metrics

    def close(self):
        """Close the shared memory buffer."""
        if hasattr(self, 'mmap'):
            self.mmap.close()
        if self.use_windows and hasattr(self, 'temp_file'):
            try:
                os.unlink(self.temp_file.name)
            except:
                pass 