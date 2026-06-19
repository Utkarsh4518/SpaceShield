#!/usr/bin/env python3
"""
SpaceShield: High-Performance SDR Raw IQ Playback Player.
Author: Antigravity AI
Version: 1.0.0

This module implements a multi-threaded high-performance reader to play back raw
binary complex I/Q recordings (interleaved float32 or int16, standard .bin/.dat or .npy)
at exact physical sampling clock rates, integrated with the SpaceShield pipeline.
"""

import os
import sys
import time
import queue
import ctypes
import numpy as np

class BinaryFilePlayer:
    def __init__(self, file_path, fs=2e6, num_channels=4, sample_type='float32', endianness='little', chunk_size=8192):
        """
        Initializes the binary file player.

        Parameters:
          file_path (str): Path to the raw complex binary or .npy file.
          fs (float): Target physical sampling rate in Hz (e.g. 2.0e6 or 5.0e6).
          num_channels (int): Antenna channels (M=4).
          sample_type (str): Input format, 'float32' or 'int16'.
          endianness (str): Byte order, 'little', 'big', or 'native'.
          chunk_size (int): Block size per read channel chunk (default 8192).
        """
        self.file_path = file_path
        self.fs = fs
        self.num_channels = num_channels
        self.sample_type = sample_type.lower()
        self.endianness = endianness.lower()
        self.chunk_size = chunk_size
        self.running = False

        # Verify options
        if self.sample_type not in ['float32', 'int16']:
            raise ValueError("sample_type must be 'float32' or 'int16'")
        if self.endianness not in ['little', 'big', 'native']:
            raise ValueError("endianness must be 'little', 'big', or 'native'")

        # Bytes configuration
        # int16 complex sample: 2 * 2 bytes = 4 bytes
        # float32 complex sample: 2 * 4 bytes = 8 bytes
        self.bytes_per_sample = 4 if self.sample_type == 'int16' else 8
        self.bytes_per_chunk = self.chunk_size * self.num_channels * self.bytes_per_sample

        # Endianness prefix for numpy frombuffer
        self.dtype_prefix = '<' if self.endianness == 'little' else ('>' if self.endianness == 'big' else '=')
        
        if self.sample_type == 'int16':
            self.np_dtype = np.dtype(f"{self.dtype_prefix}i2")
        else:
            self.np_dtype = np.dtype(f"{self.dtype_prefix}f4")

        # Temporal duration of one chunk in seconds
        self.chunk_duration = self.chunk_size / self.fs

        # Memory mapping configuration for NumPy files
        self.is_npy = file_path.endswith('.npy')
        
        # Diagnostic performance counters
        self.total_bytes_read = 0
        self.read_latencies = []
        self.delivery_latencies = []

        # Pre-allocate read and processing buffers to guarantee zero-allocation
        self.raw_buffer = bytearray(self.bytes_per_chunk)
        self.temp_flat_float = np.zeros(self.chunk_size * self.num_channels * 2, dtype=np.float32)
        self.temp_complex = np.zeros((self.num_channels, self.chunk_size), dtype=np.complex64)

        # Windows timer resolution configuration
        self.win_timer_adjusted = False
        self.winmm = None
        if os.name == 'nt':
            try:
                self.winmm = ctypes.WinDLL('winmm')
            except Exception:
                pass

        if self.is_npy:
            if not os.path.exists(self.file_path):
                # Will fail on actual playback, but checked in stream_to_harness()
                self.npy_data = None
            else:
                # High-performance read using read-only memory map
                self.npy_data = np.load(self.file_path, mmap_mode='r')
                if self.npy_data.ndim != 2:
                    raise ValueError("NPY playback file must hold a 2D complex signal array.")
                
                # Check spatial matrix orientation
                if self.npy_data.shape[0] == self.num_channels:
                    self.npy_transpose = False
                    self.total_samples = self.npy_data.shape[1]
                elif self.npy_data.shape[1] == self.num_channels:
                    self.npy_transpose = True
                    self.total_samples = self.npy_data.shape[0]
                else:
                    raise ValueError(f"NPY dimension mismatch. Expected channel dimension: {self.num_channels}")
                self.npy_index = 0

    def read_next_chunk(self, f):
        """Reads and unpacks a multi-channel interleaved binary chunk from disk in a zero-allocation manner."""
        t_start = time.perf_counter_ns()

        # Zero-copy read directly into pre-allocated raw byte buffer
        bytes_read = f.readinto(self.raw_buffer)
        if bytes_read < self.bytes_per_chunk:
            return None # EOF or partial chunk

        # Unpack raw byte stream to flat array view (Real and Imag interleaved)
        expected_elements = self.chunk_size * self.num_channels * 2
        data_flat = np.frombuffer(self.raw_buffer, dtype=self.np_dtype, count=expected_elements)

        # Cast/normalize in-place into pre-allocated flat float array to prevent allocations
        if self.sample_type == 'int16':
            np.multiply(data_flat, 1.0 / 32768.0, out=self.temp_flat_float)
        else:
            # For float32, copy/convert (if endianness conversion is needed, copyto handles it)
            np.copyto(self.temp_flat_float, data_flat)

        # Reshape view (views are zero-allocation)
        data_reshaped = self.temp_flat_float.reshape(self.chunk_size, self.num_channels, 2)

        # Form complex array in-place within the pre-allocated temp_complex buffer
        self.temp_complex.real = data_reshaped[:, :, 0].T
        self.temp_complex.imag = data_reshaped[:, :, 1].T

        t_end = time.perf_counter_ns()
        self.read_latencies.append((t_end - t_start) / 1000.0)
        self.total_bytes_read += self.bytes_per_chunk

        return self.temp_complex

    def read_next_chunk_npy(self):
        """Reads a multi-channel chunk from the memory-mapped NPY file in a zero-allocation manner."""
        t_start = time.perf_counter_ns()

        if self.npy_index + self.chunk_size > self.total_samples:
            return None # EOF

        # In-place cast and copy from memory map directly into the pre-allocated complex buffer
        if self.npy_transpose:
            np.copyto(self.temp_complex, self.npy_data[self.npy_index : self.npy_index + self.chunk_size, :].T)
        else:
            np.copyto(self.temp_complex, self.npy_data[:, self.npy_index : self.npy_index + self.chunk_size])

        self.npy_index += self.chunk_size

        t_end = time.perf_counter_ns()
        self.read_latencies.append((t_end - t_start) / 1000.0)
        self.total_bytes_read += self.chunk_size * self.num_channels * 8 # complex64 is 8 bytes per sample

        return self.temp_complex

    def stream_to_harness(self, harness):
        """Streams samples directly into the spatial harness, matching physical clocks with high-precision timing."""
        print(f"[*] BinaryFilePlayer: Playback started for {self.file_path}")
        print(f"[*] Configuration: fs={self.fs / 1e6:.2f} MSPS | Channels={self.num_channels} | Format={self.sample_type} | Endianness={self.endianness}")
        
        # Enable high-precision Windows timer resolution if applicable
        if self.winmm:
            try:
                if self.winmm.timeBeginPeriod(1) == 0:
                    self.win_timer_adjusted = True
            except Exception as e:
                print(f"[!] Warning: Could not adjust Windows multimedia timer resolution: {e}")

        f = None
        if not self.is_npy:
            try:
                f = open(self.file_path, 'rb')
            except Exception as e:
                print(f"[-] BinaryFilePlayer: Failed to open binary capture: {e}")
                if self.win_timer_adjusted:
                    try:
                        self.winmm.timeEndPeriod(1)
                    except Exception:
                        pass
                return

        start_time = time.time()
        next_chunk_time = start_time
        chunks_streamed = 0
        self.running = True

        try:
            while self.running and harness.running:
                # 1. Fetch next complex block
                if self.is_npy:
                    iq_chunk = self.read_next_chunk_npy()
                else:
                    iq_chunk = self.read_next_chunk(f)

                if iq_chunk is None:
                    print("[*] BinaryFilePlayer: End of File (EOF) reached.")
                    break

                # 2. In-place memory copy into pre-allocated harness buffer pool
                # to maintain sub-millisecond zero-allocation requirements
                buffer_idx = harness.pool_index
                harness.pool_index = (harness.pool_index + 1) % harness.pool_size
                iq_signal = harness.buffer_pool[buffer_idx]
                np.copyto(iq_signal, iq_chunk)

                # Determine simulated threat scenario tag based on play duration
                elapsed = time.time() - start_time
                if elapsed < 10.0:
                    scenario = "normal"
                elif elapsed < 20.0:
                    scenario = "jamming"
                else:
                    scenario = "spoofing"

                t_deliver_start = time.perf_counter_ns()

                # 3. Enqueue block for worker threads
                try:
                    harness.iq_queue.put_nowait((iq_signal, scenario, time.perf_counter()))
                except queue.Full:
                    with harness.metrics_lock:
                        harness.dropped_blocks += 1

                t_deliver_end = time.perf_counter_ns()
                self.delivery_latencies.append((t_deliver_end - t_deliver_start) / 1000.0)

                chunks_streamed += 1

                # 4. Hybrid high-precision sleep & busy-wait clock throttling
                next_chunk_time += self.chunk_duration
                now = time.time()
                if next_chunk_time < now - self.chunk_duration:
                    # Thread fell behind significantly, re-anchor timing schedule
                    next_chunk_time = now
                else:
                    while True:
                        now = time.time()
                        sleep_time = next_chunk_time - now
                        if sleep_time <= 0:
                            break
                        if sleep_time > 0.0015:
                            time.sleep(0.001)
                        # busy-wait for sub-millisecond timing precision

        finally:
            if f:
                f.close()
            # Restore Windows timer resolution
            if self.win_timer_adjusted:
                try:
                    self.winmm.timeEndPeriod(1)
                except Exception:
                    pass
                self.win_timer_adjusted = False
            self.running = False
            self.print_profiling_report(chunks_streamed)

    def print_profiling_report(self, chunks_streamed):
        """Prints high-precision disk read throughput and block delivery latency diagnostics."""
        total_time = chunks_streamed * self.chunk_duration
        if total_time <= 0:
            print("[!] No metrics collected: zero chunks streamed.")
            return

        throughput_mb = (self.total_bytes_read / (1024 * 1024)) / total_time
        avg_read_us = np.mean(self.read_latencies) if self.read_latencies else 0.0
        avg_delivery_us = np.mean(self.delivery_latencies) if self.delivery_latencies else 0.0

        print("\n" + "=" * 60)
        print("          BINARY FILE PLAYER PROFILING REPORT           ")
        print("=" * 60)
        print(f"  Target Sampling Clock:     {self.fs / 1e6:.2f} MSPS")
        print(f"  Total Chunks Played:       {chunks_streamed:<6}")
        print(f"  Total Ingested Data Size:  {self.total_bytes_read:,} bytes")
        print(f"  Playback Duration:         {total_time:.3f} seconds")
        print(f"  Disk Read Throughput:      {throughput_mb:.2f} MB/s")
        print(f"  Avg Disk Read Latency:     {avg_read_us:.2f} µs")
        print(f"  Avg Queue Push Latency:    {avg_delivery_us:.2f} µs")
        print("=" * 60 + "\n")

def generate_dummy_binary_file(filename, num_channels=4, num_samples=100000, sample_type='float32'):
    """Utility function to generate synthetic over-the-air capture recordings for self-test."""
    print(f"[*] Generating dummy capture file for test: {filename} ({sample_type})...")
    
    t = np.arange(num_samples)
    data = np.zeros((num_samples, num_channels, 2), dtype=np.float32)
    
    for c in range(num_channels):
        # Insert simple phase shifted tones
        data[:, c, 0] = 0.5 * np.cos(2 * np.pi * (1000.0 * (c + 1)) * t / 2e6)
        data[:, c, 1] = 0.5 * np.sin(2 * np.pi * (1000.0 * (c + 1)) * t / 2e6)

    if filename.endswith('.npy'):
        complex_data = data[:, :, 0] + 1j * data[:, :, 1]
        np.save(filename, complex_data.T)
    else:
        if sample_type == 'int16':
            data_to_write = (data * 32767.0).astype(np.int16)
        else:
            data_to_write = data.astype(np.float32)
        with open(filename, 'wb') as f:
            f.write(data_to_write.tobytes())
            
    print(f"[+] Capture file generation complete: {filename}")

def main():
    """Independent diagnostic run of the binary player."""
    test_bin = "data/dummy_ota_capture.bin"
    test_npy = "data/dummy_ota_capture.npy"
    os.makedirs("data", exist_ok=True)
    
    # 1. Test binary raw float32
    generate_dummy_binary_file(test_bin, num_channels=4, num_samples=200000, sample_type='float32')
    
    # 2. Test memory-mapped npy
    generate_dummy_binary_file(test_npy, num_channels=4, num_samples=200000, sample_type='float32')

    # Instantiate player
    player = BinaryFilePlayer(
        file_path=test_bin,
        fs=2e6,
        num_channels=4,
        sample_type='float32',
        endianness='little',
        chunk_size=8192
    )

    # Simple mock harness for streaming self-test
    class MockHarness:
        def __init__(self):
            self.running = True
            self.iq_queue = queue.Queue(maxsize=100)
            self.pool_size = 150
            self.pool_index = 0
            self.buffer_pool = [np.zeros((4, 8192), dtype=np.complex64) for _ in range(self.pool_size)]
            self.metrics_lock = queue.threading.Lock()
            self.dropped_blocks = 0

    harness = MockHarness()
    
    print("[*] Starting self-test binary streaming playback...")
    player.stream_to_harness(harness)
    
    # Verify NPY mapped streaming
    print("[*] Starting self-test NPY memory-mapped playback...")
    player_npy = BinaryFilePlayer(
        file_path=test_npy,
        fs=2e6,
        num_channels=4,
        sample_type='float32',
        endianness='little',
        chunk_size=8192
    )
    player_npy.stream_to_harness(harness)

    # Clean up files
    try:
        os.remove(test_bin)
        os.remove(test_npy)
        print("[+] Diagnostic cleanup complete.")
    except Exception:
        pass

if __name__ == "__main__":
    main()
