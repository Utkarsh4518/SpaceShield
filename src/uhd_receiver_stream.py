#!/usr/bin/env python3
"""
SpaceShield: Real-Time USRP SDR Receiver Ingestion Pipeline.
Author: Antigravity AI
Version: 1.0.0

This module interfaces with a physical Ettus USRP device (e.g., B210) via the
official UHD Python API. It implements a multi-threaded architecture to
ingest raw complex64 IQ sample streams, handle hardware buffer overflows
defensively, and pipe data into the SpaceShield Edge anomaly detection engine.
"""

import os
import sys
import time
import queue
import threading
import numpy as np
import argparse

# Attempt to import UHD Python API
try:
    import uhd
except ImportError:
    print("[!] Warning: official 'uhd' library not found.")
    print("[!] Ensure UHD is installed and the Python bindings are in PYTHONPATH.")
    print("[!] Falling back to Simulated Ingestion Mode if no hardware is connected.")
    uhd = None

# Downstream feature extraction imports
try:
    from rf_threat_simulator import extract_features, classify_rf_threat, write_certin_compliance_log
except ImportError:
    # If run outside the src folder
    try:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from rf_threat_simulator import extract_features, classify_rf_threat, write_certin_compliance_log
    except ImportError:
        extract_features = None
        print("[!] Downstream classification engine imports failed. Running in ingest-only mode.")

class USRPStreamer:
    def __init__(self, args):
        self.args = args
        self.usrp = None
        self.streamer = None
        self.running = False
        self.sample_queue = queue.Queue(maxsize=100)
        self.receiver_thread = None
        self.processor_thread = None

        # Determine target frequency based on band selection
        if args.band == 'L5':
            self.center_freq = 1176.45e6  # 1176.45 MHz (NavIC L5)
        else:
            self.center_freq = 2492.028e6  # 2492.028 MHz (NavIC S-band)

        self.sample_rate = args.rate
        self.gain = args.gain
        self.chunk_size = args.chunk_size

    def initialize_hardware(self):
        """Initializes the USRP hardware device and configures radio parameters."""
        if uhd is None:
            print("[!] UHD API is unavailable. Cannot initialize hardware.")
            return False

        try:
            print(f"[*] Locating and initializing USRP device with args: '{self.args.args}'...")
            self.usrp = uhd.usrp.MultiUSRP(self.args.args)
            
            # Print device information
            info = self.usrp.get_usrp_info()
            print(f"[+] Connected to: {info.get('mboard_id', 'Unknown Board')} (S/N: {info.get('mboard_serial', 'N/A')})")

            # 1. Set Clock & Time Source
            if self.args.external_ref:
                self.usrp.set_clock_source("external")
                self.usrp.set_time_source("external")
                print("[*] Clock & Time reference set to: EXTERNAL (10 MHz/PPS)")
            else:
                self.usrp.set_clock_source("internal")
                self.usrp.set_time_source("internal")
                print("[*] Clock & Time reference set to: INTERNAL")

            # 2. Configure RF Channel parameters
            channel = 0
            
            # Set master clock rate if specified
            if self.args.master_clock_rate > 0:
                print(f"[*] Setting master clock rate to: {self.args.master_clock_rate / 1e6:.3f} MHz")
                self.usrp.set_master_clock_rate(self.args.master_clock_rate)

            print(f"[*] Configuring RX Channel {channel}:")
            print(f"    - Target Frequency: {self.center_freq / 1e6:.3f} MHz")
            print(f"    - Target Sampling Rate: {self.sample_rate / 1e6:.3f} MSPS")
            print(f"    - Hardware Gain: {self.gain} dB")

            self.usrp.set_rx_rate(self.sample_rate, channel)
            self.usrp.set_rx_freq(uhd.types.TuneRequest(self.center_freq), channel)
            self.usrp.set_rx_gain(self.gain, channel)

            # Read back settings to verify calibration
            actual_rate = self.usrp.get_rx_rate(channel)
            actual_freq = self.usrp.get_rx_freq(channel)
            actual_gain = self.usrp.get_rx_gain(channel)
            print(f"[+] Readback configuration:")
            print(f"    - Actual Rate: {actual_rate / 1e6:.3f} MSPS")
            print(f"    - Actual Freq: {actual_freq / 1e6:.3f} MHz")
            print(f"    - Actual Gain: {actual_gain} dB")

            # 3. Create high-performance stream
            stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
            stream_args.channels = [channel]
            
            # Configure receive frame size to minimize context switching overhead
            if self.args.recv_frame_size > 0:
                stream_args.args = f"recv_frame_size={self.args.recv_frame_size}"
                print(f"[*] Set streaming frame size to: {self.args.recv_frame_size} bytes")

            self.streamer = self.usrp.get_rx_stream(stream_args)
            
            return True

        except Exception as e:
            print(f"[-] Hardware initialization error: {e}")
            return False

    def receiver_loop(self):
        """Dedicated high-priority thread loop pulling raw I/Q samples from UHD."""
        print("[*] Starting USRP RX buffer ingestion thread...")
        
        # Configure streaming command
        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_continuous)
        stream_cmd.stream_now = True
        self.streamer.issue_stream_cmd(stream_cmd)

        # Allocate buffer for complex64 samples
        # MultiUSRP expects: (channels, chunk_size)
        buffer = np.zeros((1, self.chunk_size), dtype=np.complex64)
        metadata = uhd.types.RXMetadata()

        # Timeout configuration (3.0 seconds fallback)
        timeout = 3.0

        while self.running:
            try:
                # Receive samples from hardware
                num_samps = self.streamer.recv(buffer, metadata, timeout)

                if metadata.error_code == uhd.types.RXMetadataErrorCode.none:
                    if num_samps > 0:
                        # Extract the active batch channel array and copy to decouple buffer reference
                        batch = buffer[0, :num_samps].copy()
                        try:
                            # Push to processing queue without blocking the ingestion thread
                            self.sample_queue.put_nowait((batch, time.perf_counter()))
                        except queue.Full:
                            # Queue backed up; indicates downstream processing slowdown
                            print("[!] Queue Full: Dropping frame to prevent hardware lockup.")
                
                elif metadata.error_code == uhd.types.RXMetadataErrorCode.overflow:
                    print("[!] Hardware Alert: Ingestion buffer overflow detected (ERROR_CODE_OVERFLOW). Recovering...")
                    # Small sleep to clear hardware buffer congestion
                    time.sleep(0.01)
                    continue

                elif metadata.error_code == uhd.types.RXMetadataErrorCode.timeout:
                    print("[!] Warning: Ingest timeout (ERROR_CODE_TIMEOUT). Waiting for packet synchronization...")
                    continue

                elif hasattr(uhd.types.RXMetadataErrorCode, 'late_command') and metadata.error_code == uhd.types.RXMetadataErrorCode.late_command:
                    print("[!] Hardware Alert: Late command timing alignment error (ERROR_CODE_LATE_COMMAND). Re-syncing...")
                    continue

                else:
                    print(f"[!] Metadata Error Code: {metadata.strerror()}")
                    continue

            except Exception as e:
                print(f"[-] Fatal Exception in ingestion loop: {e}")
                self.running = False

        # Issue stop streaming command
        try:
            stop_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_continuous)
            self.streamer.issue_stream_cmd(stop_cmd)
            print("[*] Ingestion loop terminated gracefully.")
        except Exception as e:
            print(f"[!] Error stopping stream command: {e}")

    def processor_loop(self):
        """Processes queued raw I/Q samples and feeds them to the classification engine."""
        print("[*] Starting processing pipeline thread...")
        
        while self.running or not self.sample_queue.empty():
            try:
                try:
                    iq_data, capture_time = self.sample_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                proc_start = time.perf_counter()
                latency_ingest_ms = (proc_start - capture_time) * 1000.0

                # 1. Feature Extraction & Threat Classification
                if extract_features:
                    # Construct matching metadata for pipeline ingestion
                    meta = {
                        "scenario": "live",
                        "fs": self.sample_rate,
                        "carrier_freq": self.center_freq,
                        "description": "Live hardware stream from USRP B210 receiver."
                    }
                    features = extract_features(iq_data, self.sample_rate, meta)
                    result = classify_rf_threat(features)
                    
                    proc_end = time.perf_counter()
                    latency_processing_ms = (proc_end - proc_start) * 1000.0
                    total_latency_ms = latency_ingest_ms + latency_processing_ms

                    # Log verification logs if anomalies are triggered
                    if result["verdict"] != "NORMAL" and self.args.enable_compliance:
                        # Write logs inside the target data and compliance directories
                        write_certin_compliance_log("live", features, result, "compliance", "data")

                    # Print status monitoring line
                    print(f"[*] Ingest Latency: {latency_ingest_ms:.2f}ms | Proc Time: {latency_processing_ms:.2f}ms | "
                          f"Verdict: {result['verdict']:<8} | I/N: {features['in_ratio_db']:.2f}dB | GLR: {features['glr_statistic']:.1f}")
                else:
                    # Fallback monitoring if pipeline module not loaded
                    proc_end = time.perf_counter()
                    latency_processing_ms = (proc_end - proc_start) * 1000.0
                    print(f"[*] Ingest Latency: {latency_ingest_ms:.2f}ms | Samples: {len(iq_data)} (Analysis offline)")

                self.sample_queue.task_done()

            except Exception as e:
                print(f"[-] Exception in processing thread: {e}")

    def run_simulated(self):
        """Simulates sample stream processing if no USRP is connected."""
        print("[*] Running in Simulated Ingestion Mode...")
        self.running = True
        
        # Start processing thread
        self.processor_thread = threading.Thread(target=self.processor_loop)
        self.processor_thread.start()

        # Simulate USRP generator
        try:
            while self.running:
                # Generate artificial NavIC samples
                noise = np.random.normal(0, 0.1, self.chunk_size) + 1j * np.random.normal(0, 0.1, self.chunk_size)
                # Introduce occasional fake spoofing/jamming signals randomly
                rand = np.random.rand()
                if rand > 0.95:
                    # Fake jamming
                    signal = np.random.normal(0, 1.2, self.chunk_size) + 1j * np.random.normal(0, 1.2, self.chunk_size)
                elif rand < 0.05:
                    # Fake spoofer tone
                    t = np.arange(self.chunk_size) / self.sample_rate
                    signal = 0.8 * np.exp(1j * 2 * np.pi * 150 * t) # Offset tone
                else:
                    signal = 0.2 * np.exp(1j * 2 * np.pi * 5 * t if 't' in locals() else np.zeros(self.chunk_size))
                
                iq_data = (noise + signal).astype(np.complex64)
                
                try:
                    self.sample_queue.put((iq_data, time.perf_counter()), timeout=1.0)
                except queue.Full:
                    pass
                
                time.sleep(self.chunk_size / self.sample_rate)
        except KeyboardInterrupt:
            print("[*] Simulation interrupt received.")
        
        self.running = False
        self.processor_thread.join()

    def start(self):
        """Starts both receiver and processing loops."""
        self.running = True
        
        # 1. Start processing thread
        self.processor_thread = threading.Thread(target=self.processor_loop)
        self.processor_thread.start()

        # 2. Start receiver thread
        self.receiver_thread = threading.Thread(target=self.receiver_loop)
        self.receiver_thread.start()

        try:
            while self.running:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\n[*] Shutting down signal ingestion stream...")
        finally:
            self.running = False
            if self.receiver_thread:
                self.receiver_thread.join()
            if self.processor_thread:
                self.processor_thread.join()
            print("[+] SpaceShield stream shut down successfully.")

def main():
    parser = argparse.ArgumentParser(description="SpaceShield USRP Hardware stream ingestion agent.")
    parser.add_argument('--band', type=str, choices=['L5', 'S'], default='L5',
                        help="Target band: 'L5' (1176.45 MHz) or 'S' (2492.028 MHz)")
    parser.add_argument('--rate', type=float, default=2e6,
                        help="USRP ADC sampling rate in Hz (default: 2.0 MSPS)")
    parser.add_argument('--gain', type=float, default=45.0,
                        help="USRP hardware RX frontend gain in dB (default: 45.0)")
    parser.add_argument('--chunk-size', type=int, default=8192,
                        help="Buffer chunk size (default: 8192)")
    parser.add_argument('--master-clock-rate', type=float, default=0.0,
                        help="USRP Master Clock Rate in Hz (default: 0.0, use device default)")
    parser.add_argument('--recv-frame-size', type=int, default=0,
                        help="UHD receive frame size buffer configuration in bytes")
    parser.add_argument('--args', type=str, default="",
                        help="USRP board initialization args (e.g. 'type=b200')")
    parser.add_argument('--external-ref', action='store_true',
                        help="Force clock source to external 10MHz/PPS reference")
    parser.add_argument('--enable-compliance', action='store_true',
                        help="Write CERT-In compliance logs for detected live threat scenarios")
    parser.add_argument('--simulated', action='store_true',
                        help="Bypass USRP device detection and run simulated stream")
    args = parser.parse_args()

    streamer = USRPStreamer(args)

    if args.simulated or uhd is None:
        if uhd is None and not args.simulated:
            print("[!] Falling back to Simulated mode due to missing dependencies.")
        streamer.run_simulated()
    else:
        if streamer.initialize_hardware():
            streamer.start()
        else:
            print("[-] Hardware initialisation failed. Try executing with '--simulated'")
            sys.exit(1)

if __name__ == "__main__":
    main()
