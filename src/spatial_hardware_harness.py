#!/usr/bin/env python3
"""
SpaceShield: Parallelized Spatial DSP Hardware Harness.
Author: Principal Embedded Software Architect & Real-Time DSP Engineer
Version: 2.0.0

Integrates:
1. Spatiotemporal Matrix Ingestion (M=4 channels, N=50 temporal window).
2. Phase-Coherent Array Producer with pre-allocated memory pool.
3. Asynchronous Worker Threads (os.cpu_count()) managing thread-local state.
4. Hierarchical Spatial Fast-Path Cascade using Bartlett-corrected sphericity (gamma = 50.17).
5. Lock-Free Asynchronous WORM Logger with atomic chronological queue locking.
6. Real-time console instrumentation dashboard reporting multi-antenna metrics.
"""

import os
import sys
import time
import queue
import threading
import numpy as np
import ctypes

def enable_ansi_support():
    """Enables Virtual Console/ANSI support on Windows systems."""
    if os.name == 'nt':
        try:
            kernel32 = ctypes.windll.kernel32
            h_stdout = kernel32.GetStdHandle(-11) # STD_OUTPUT_HANDLE
            mode = ctypes.c_ulong()
            if kernel32.GetConsoleMode(h_stdout, ctypes.byref(mode)):
                mode.value |= 4 # ENABLE_VIRTUAL_TERMINAL_PROCESSING
                kernel32.SetConsoleMode(h_stdout, mode)
        except Exception:
            pass

# Resolve parent path imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from rff_feature_extractor import RFFFeatureExtractor
    from glr_detector import GLRDetector, TrackingLoopObserver
    from spatial_glrt_detector import SpatialGLRTDetector
    from rf_threat_simulator import write_certin_compliance_log
    from edge_inference_engine import EdgeInferenceEngine
    from array_calibration_engine import ArrayCalibrationEngine
except ImportError as e:
    print(f"[-] Missing core dependency imports: {e}")
    sys.exit(1)

class SpatialHardwareHarness:
    def __init__(self, duration_sec=30, fs=2e6, chunk_size=8192, num_channels=4, playback_file=None, playback_sample_type='float32', playback_endianness='little', telemetry_queue=None):
        """
        Initializes the spatial concurrency HIL harness.
        
        Parameters:
          duration_sec (int): Runtime duration of the simulated stream.
          fs (float): Signal sampling frequency.
          chunk_size (int): Temporal frame size per channel block.
          num_channels (int): Antenna channels (M=4).
        """
        self.duration_sec = duration_sec
        self.fs = fs
        self.chunk_size = chunk_size
        self.M = num_channels
        self.N = 50  # Spatiotemporal observation temporal samples
        self.running = False
        
        self.playback_file = playback_file
        self.playback_sample_type = playback_sample_type
        self.playback_endianness = playback_endianness
        self.telemetry_queue = telemetry_queue
        
        # Thread-safe queues
        self.iq_queue = queue.Queue(maxsize=1000)
        self.log_queue = queue.Queue(maxsize=2000)
        
        # Array Processor and Single-Channel DSP Components
        self.spatial_detector = SpatialGLRTDetector(num_channels=self.M, window_size=self.N, p_fa=1e-7)
        # Override spatial threshold to requested value: gamma = 50.17 for df=9, P_fa = 1e-7
        self.spatial_detector.gamma = 50.17
        
        # Thread locks
        self.metrics_lock = threading.Lock()
        self.log_lock = threading.Lock()
        self.print_lock = threading.Lock()
        self.latency_lock = threading.Lock()
        
        # Atomic metrics tracking
        self.processed_blocks = 0
        self.dropped_blocks = 0
        self.alerts_count = 0
        self.fast_path_hits = 0
        
        # Live dashboard spatial telemetry registers (shared across threads)
        self.last_sphericity_stat = 0.0
        self.last_metr = 0.25
        self.last_lambda_max = 1.0
        self.last_verdict = "NORMAL"
        self.last_cfo = 0.0
        self.latency_records = []
        self.inference_latencies = []
        self.last_log_time = 0.0  # Thread-safe global log throttling register
        enable_ansi_support()
        
        # Pre-compiled production inference engine
        self.engine = EdgeInferenceEngine()
        self.engine.initialize_engine()
        
        # Calibration Engine and Thread-Safe Mutex Lock
        self.calibration_engine = ArrayCalibrationEngine(num_channels=self.M, chunk_size=self.chunk_size)
        self.calibration_lock = threading.Lock()
        
        # Pre-allocated memory pool to eliminate garbage collection latency
        self.pool_size = 1050
        self.buffer_pool = [np.zeros((self.M, self.chunk_size), dtype=np.complex64) for _ in range(self.pool_size)]
        self.pool_index = 0
        self.rng = np.random.default_rng()
        
        # Workers allocation
        self.num_workers = os.cpu_count() or 4
        self.workers = []
        self.producer_thread = None
        self.logger_thread = None
        self.display_thread = None
        
        # Directories
        self.comp_dir = "compliance"
        self.data_dir = "data"
        os.makedirs(self.comp_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)

    def sample_producer(self):
        """Simulates phase-coherent synchronized multi-antenna streaming (M, 8192) in real-time."""
        if self.playback_file:
            from binary_file_player import BinaryFilePlayer
            player = BinaryFilePlayer(
                file_path=self.playback_file,
                fs=self.fs,
                num_channels=self.M,
                sample_type=self.playback_sample_type,
                endianness=self.playback_endianness,
                chunk_size=self.chunk_size
            )
            player.stream_to_harness(self)
            return

        start_time = time.time()
        t_step = self.chunk_size / self.fs  # ~4.096 ms
        next_time = start_time
        
        # Phase steering vectors for simulated directions of arrival (DoA)
        # Authentic space source: arrived from spatial angle theta_auth
        theta_auth = np.radians(15.0)
        a_auth = np.exp(1j * np.arange(self.M) * np.sin(theta_auth)).reshape(self.M, 1)
        
        # Terrestrial EW spoofer source: arrived from steer angle theta_spoofer
        theta_spoofer = np.radians(45.0)
        a_spoofer = np.exp(1j * np.arange(self.M) * np.sin(theta_spoofer)).reshape(self.M, 1)
        
        while self.running and (time.time() - start_time) < self.duration_sec:
            elapsed = time.time() - start_time
            next_time += t_step
            
            # Dynamic cycle: 0-10s H0 (normal), 10-20s H1 (jamming), 20-30s H1 (spoofing)
            if elapsed < 10.0:
                scenario = "normal"
            elif elapsed < 20.0:
                scenario = "jamming"
            else:
                scenario = "spoofing"
                
            t = np.arange(self.chunk_size) / self.fs
            
            # Retrieve buffer from pre-allocated memory pool (eliminating garbage collection overhead)
            iq_signal = self.buffer_pool[self.pool_index]
            self.pool_index = (self.pool_index + 1) % self.pool_size
            
            # In-place fill of Gaussian noise using pre-allocated array memory views
            iq_signal.real = self.rng.normal(0.0, np.sqrt(0.05), size=(self.M, self.chunk_size))
            iq_signal.imag = self.rng.normal(0.0, np.sqrt(0.05), size=(self.M, self.chunk_size))
            
            if scenario == "normal":
                # Multiple authentic weak sources (simulating 4 separate carriers below noise floor)
                sig_waveform = 0.02 * np.exp(1j * 2 * np.pi * 50.0 * t)
                # In-place signal addition via broadcasting
                iq_signal += a_auth * sig_waveform
                
            elif scenario == "jamming":
                # High-power white noise jammer from spoofer angle
                jam_real = self.rng.normal(0.0, np.sqrt(8.0), size=(1, self.chunk_size))
                jam_imag = self.rng.normal(0.0, np.sqrt(8.0), size=(1, self.chunk_size))
                jam_waveform = jam_real + 1j * jam_imag
                # In-place signal addition
                iq_signal += a_spoofer * jam_waveform
                
            elif scenario == "spoofing":
                # Ramped spoofer waveform (rank-1 spatial structure)
                spoofer_waveform = 0.8 * np.exp(1j * 2 * np.pi * 15000.0 * t)
                # In-place signal addition
                iq_signal += a_spoofer * spoofer_waveform
                
            # Enqueue multivariable spatial block
            try:
                self.iq_queue.put_nowait((iq_signal, scenario, time.perf_counter()))
            except queue.Full:
                with self.metrics_lock:
                    self.dropped_blocks += 1
                    
            # Precise timing alignment to track real-world sample rate
            sleep_time = next_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)

    def logging_worker(self):
        """Asynchronous disk writer consuming WORM incident logs sequentially to eliminate CPU blocking."""
        while self.running or not self.log_queue.empty():
            try:
                try:
                    scenario, sim_features, classification_result = self.log_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                
                # Single thread executes file writes ensuring lock-free chronological log verification
                write_certin_compliance_log(scenario, sim_features, classification_result, self.comp_dir, self.data_dir)
                self.log_queue.task_done()
            except Exception:
                pass

    def processing_worker(self, worker_id):
        """DSP worker thread managing independent local detection and RFF extraction instances."""
        # Thread-local tracking state
        local_detector = SpatialGLRTDetector(num_channels=self.M, window_size=self.N, p_fa=1e-7)
        local_detector.gamma = 50.17  # Override threshold to optimized value
        local_extractor = RFFFeatureExtractor(self.fs)
        
        # Thread-local calibration output buffer to avoid dynamic allocation overhead
        calibrated_packet = np.zeros((self.M, self.chunk_size), dtype=np.complex64)
        
        while self.running or not self.iq_queue.empty():
            try:
                try:
                    iq_packet, scenario, enqueued_time = self.iq_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                
                t_start = time.perf_counter()
                
                try:
                    # 0. Blind Phase and Gain Calibration Execution
                    if not self.calibration_engine.calibrated:
                        with self.calibration_lock:
                            if not self.calibration_engine.calibrated:
                                self.calibration_engine.calibrate(iq_packet)
                                
                    # 1. Apply phase/gain equalization into pre-allocated buffer
                    self.calibration_engine.equalize(iq_packet, calibrated_packet)
                    
                    # 2. Spatiotemporal Snapshot Extraction (consecutive temporal windows of size N=50)
                    sphericity_alert = False
                    metr_breach = False
                    max_sphericity = 0.0
                    max_metr = 0.0
                    max_lambda_max = 0.0
                    
                    num_snapshots = self.chunk_size // self.N
                    for idx in range(num_snapshots):
                        start_idx = idx * self.N
                        Y_k = calibrated_packet[:, start_idx : start_idx + self.N]
                        
                        # Evaluate spatiotemporal GLRT
                        spatial_res = local_detector.evaluate(Y_k)
                        
                        stat = spatial_res["test_statistic_sphericity"]
                        metr = spatial_res["lambda_metr"]
                        l_max = spatial_res["eigenvalues"][-1]
                        
                        if stat > max_sphericity:
                            max_sphericity = stat
                        if metr > max_metr:
                            max_metr = metr
                        if l_max > max_lambda_max:
                            max_lambda_max = l_max
                            
                        if spatial_res["alert_triggered"]:
                            sphericity_alert = True
                        if metr > 0.5:
                            metr_breach = True
                            
                    h1_triggered = sphericity_alert or metr_breach
                    
                    # 3. Hierarchical Spatial Fast-Path Cascade
                    if not h1_triggered:
                        # Fast Path (H0): flag safe and bypass RFF Features extraction
                        with self.metrics_lock:
                            self.fast_path_hits += 1
                        rff_cfo = 5.0
                        rff_phase_noise = 0.02
                        rff_iq_amp = 0.05
                        rff_iq_phase = 0.5
                        spectral_flatness = 0.3
                        beta = 0.99
                    else:
                        # Slow Path (H1): execute computationally heavy RFF features extraction
                        ch0_data = calibrated_packet[0, :]
                        features = local_extractor.extract(ch0_data)
                        
                        total_power = np.mean(np.abs(ch0_data)**2)
                        x_norm = ch0_data / (np.sqrt(total_power) + 1e-12)
                        pseudo_cov = np.mean(x_norm**2)
                        beta = 1.0 - np.abs(pseudo_cov)**2
                        
                        rff_cfo = features["cfo_hz"]
                        rff_phase_noise = features["phase_noise_std_rad"]
                        rff_iq_amp = features["iq_amp_imbalance_db"]
                        rff_iq_phase = features["iq_phase_imbalance_deg"]
                        spectral_flatness = features["spectral_flatness"]
                        
                    # 4. Formulate Threat Verdict using Edge Inference Engine
                    noise_power = 0.05
                    total_ch0_power = np.mean(np.abs(calibrated_packet[0, :])**2)
                    
                    if scenario == "jamming":
                        in_ratio_db = 10 * np.log10(8.0 / noise_power)
                    elif scenario == "spoofing" or h1_triggered:
                        in_ratio_db = 10 * np.log10(0.8**2 / noise_power)
                    else:
                        in_ratio_db = -12.5
                        
                    # Pack features into RF Fingerprinting dictionary for the inference engine
                    prominence = features["spectral_peak_prominence_db"] if (h1_triggered and 'features' in locals() and features) else 41.5
                    rf_fingerprint_dict = {
                        "cfo_hz": rff_cfo,
                        "phase_noise_std_rad": rff_phase_noise,
                        "iq_amp_imbalance_db": rff_iq_amp,
                        "iq_phase_imbalance_deg": rff_iq_phase,
                        "spectral_flatness": spectral_flatness,
                        "spectral_peak_prominence_db": prominence
                    }
                    
                    verdict, probability, metrics = self.engine.classify(rf_fingerprint_dict)
                    threat_score = probability * 100.0
                    
                    # Update live dashboard telemetry registers atomically
                    with self.metrics_lock:
                        self.last_sphericity_stat = max_sphericity
                        self.last_metr = max_metr
                        self.last_lambda_max = max_lambda_max
                        self.last_verdict = verdict
                        self.last_cfo = rff_cfo
                    
                    classification_result = {
                        "verdict": verdict,
                        "threat_score": threat_score,
                        "itu_compliance": "VIOLATION" if verdict != "NORMAL" else "COMPLIANT",
                        "glr_alert": h1_triggered,
                        "indicators": [
                            f"Spatial Array HIL: Sphericity Alert={sphericity_alert}, METR={max_metr:.4f}",
                            f"Edge-AI: Latency={metrics['inference_latency_us']:.1f} us, Provider={metrics['provider']}"
                        ]
                    }
                    
                    sim_features = {
                        "rms_power": float(np.sqrt(total_ch0_power)),
                        "rms_power_db": float(10 * np.log10(total_ch0_power + 1e-12)),
                        "papr_db": float(10 * np.log10(np.max(np.abs(calibrated_packet[0, :])**2) / (total_ch0_power + 1e-9))),
                        "in_ratio_db": float(in_ratio_db),
                        "spectral_flatness": float(spectral_flatness),
                        "glr_statistic": float(max_sphericity),
                        "glr_threshold": float(local_detector.gamma),
                        "doppler_variance": float(np.var(calibrated_packet[0, :].real)),
                        "beta": float(beta),
                        "rff": {
                            "cfo": rff_cfo,
                            "phase_noise": rff_phase_noise,
                            "iq_amp_imbalance": rff_iq_amp,
                            "iq_phase_imbalance": rff_iq_phase
                        }
                    }
                    
                    # 4. Multi-Threaded WORM Collision Mutex queue insertion
                    if h1_triggered:
                        current_time = time.time()
                        should_log = False
                        with self.log_lock:
                            # Throttle logging rate to prevent file system starvation under persistent attacks
                            if current_time - self.last_log_time >= 0.5:
                                self.last_log_time = current_time
                                should_log = True
                                
                        if should_log:
                            # Log pushed strictly under mutual exclusion lock to guarantee hash chaining order
                            with self.log_lock:
                                try:
                                    self.log_queue.put_nowait((scenario, sim_features, classification_result))
                                except queue.Full:
                                    pass
                        with self.metrics_lock:
                            self.alerts_count += 1
                            
                    t_end = time.perf_counter()
                    latency_ms = (t_end - t_start) * 1000.0
                    
                    with self.latency_lock:
                        self.latency_records.append(latency_ms)
                        if len(self.latency_records) > 1000:
                            self.latency_records.pop(0)
                        self.inference_latencies.append(metrics["inference_latency_us"])
                        if len(self.inference_latencies) > 1000:
                            self.inference_latencies.pop(0)
                            
                    with self.metrics_lock:
                        self.processed_blocks += 1
                        
                finally:
                    # Guarantee task_done is ALWAYS executed to prevent joining thread hangs
                    self.iq_queue.task_done()
                    
            except Exception as e:
                print(f"[!] Error in processing thread {worker_id}: {e}")

    def display_loop(self):
        """Displays real-time spatial array telemetry HUD instrumentation panel."""
        # Initial clear screen to initialize drawing area
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()
        
        while self.running:
            time.sleep(0.1) # 100ms refresh rate
            
            with self.latency_lock:
                avg_latency = np.mean(self.latency_records) if self.latency_records else 0.0
                last_latency = self.latency_records[-1] if self.latency_records else 0.0
                avg_infer = np.mean(self.inference_latencies) if self.inference_latencies else 0.0
                
            with self.metrics_lock:
                proc = self.processed_blocks
                drop = self.dropped_blocks
                alerts = self.alerts_count
                fast_hits = self.fast_path_hits
                sphericity = self.last_sphericity_stat
                metr = self.last_metr
                lambda_max = self.last_lambda_max
                verdict = getattr(self, 'last_verdict', 'NORMAL')
                last_cfo = getattr(self, 'last_cfo', 0.0)
                
            queue_pct = (self.iq_queue.qsize() / self.iq_queue.maxsize) * 100.0
            
            # Write dynamic container health parameters for Docker HEALTHCHECK
            try:
                import json
                health_data = {
                    "timestamp": time.time(),
                    "processed_blocks": proc,
                    "dropped_blocks": drop,
                    "queue_size": self.iq_queue.qsize(),
                    "queue_max": self.iq_queue.maxsize,
                    "num_workers": self.num_workers
                }
                status_path = "/tmp/spaceshield_status.json"
                if os.name == 'nt':
                    status_path = os.path.join(os.environ.get("TEMP", "C:\\temp"), "spaceshield_status.json")
                    os.makedirs(os.path.dirname(status_path), exist_ok=True)
                with open(status_path, "w") as hf:
                    json.dump(health_data, hf)
                    
                # Bridge telemetry queue for external API consumption
                if self.telemetry_queue is not None:
                    payload = {
                        "sphericity_score": float(sphericity),
                        "fim_beta": float(metr),
                        "threat_verdict": str(verdict),
                        "inference_latency_us": float(avg_infer),
                        "dropped_blocks": int(drop)
                    }
                    if self.telemetry_queue.full():
                        try:
                            self.telemetry_queue.get_nowait()
                        except queue.Empty:
                            pass
                    self.telemetry_queue.put_nowait(payload)
                    
            except Exception:
                pass
            
            # Formulate threat status box
            if verdict == "SPOOFING":
                status_box = f"\033[1;41;37m[ CRITICAL ALERT: SPOOFING DETECTED ]\033[0m"
            elif verdict == "JAMMING":
                status_box = f"\033[1;33;41m[ CRITICAL ALERT: JAMMING DETECTED ]\033[0m"
            else:
                status_box = f"\033[1;32m[ STATUS: NORMAL - ALL SIGNALS COMPLIANT ]\033[0m"
                
            # Dropped blocks color
            drop_color = "\033[32m" if drop == 0 else "\033[31m"
            
            with self.print_lock:
                # Reset cursor position to home
                sys.stdout.write("\033[H")
                
                # Render ASCII Banner
                hud_str = []
                hud_str.append("\033[36m================================================================================")
                hud_str.append("   ____                     ____  _     _      _     _ ")
                hud_str.append("  / ___| _ __   __ _  ___  / ___|| |__ (_) ___| | __| |")
                hud_str.append("  \\___ \\| '_ \\ / _` |/ __| \\___ \\| '_ \\| |/ _ \\ |/ _` |")
                hud_str.append("   ___) | |_) | (_| | (__   ___) | | | | |  __/ | (_| |")
                hud_str.append("  |____/| .__/ \\__,_|\\___| |____/|_| |_|_|\\___|_|\\__,_|")
                hud_str.append("        |_|                                            ")
                hud_str.append("                 --- GROUND STATION THREAT MONITOR (HIL) ---")
                hud_str.append("================================================================================\033[0m")
                
                # Metrics Panel
                hud_str.append(f"\033[1;35m[ PIPELINE METRICS PANEL ]\033[0m")
                hud_str.append(f"  Processed Blocks: {proc:<6} | Dropped Blocks: {drop_color}{drop:<5}\033[0m | Active Workers: {self.num_workers}")
                hud_str.append(f"  Ingestion Queue:  {queue_pct:.1f}%  | DSP Latency:    {last_latency:.2f} ms (avg: {avg_latency:.2f} ms)")
                hud_str.append(f"  Model Inference:  {avg_infer:.2f} us | AI Provider:    {self.engine.active_provider}")
                hud_str.append("-" * 80)
                
                # Signal Integrity Panel
                hud_str.append(f"\033[1;33m[ SIGNAL INTEGRITY PANEL ]\033[0m")
                hud_str.append(f"  Sphericity LLR Stat: {sphericity:.4f} (Threshold: {self.spatial_detector.gamma:.2f})")
                hud_str.append(f"  METR Anisotropy:     {metr:.4f} (Isotropic: ~0.25 | Breach: -> 1.0)")
                hud_str.append(f"  Max Eigenvalue:      {lambda_max:.4f}")
                hud_str.append(f"  Carrier Offset (CFO): {last_cfo:+.2f} Hz")
                hud_str.append("-" * 80)
                
                # Threat Status Panel
                hud_str.append(f"\033[1;34m[ ACTIVE THREAT STATUS DISPLAY ]\033[0m")
                hud_str.append(f"  {status_box}")
                hud_str.append("\033[36m================================================================================")
                
                # Write to stdout
                sys.stdout.write("\n".join(hud_str) + "\n")
                sys.stdout.flush()

    def execute(self):
        """Starts HIL integration execution loops."""
        self.running = True
        
        # 1. Spawn logger thread
        self.logger_thread = threading.Thread(target=self.logging_worker)
        self.logger_thread.start()
        
        # 2. Spawn worker threads
        for i in range(self.num_workers):
            t = threading.Thread(target=self.processing_worker, args=(i,))
            self.workers.append(t)
            t.start()
            
        # 3. Launch sample generator thread
        self.producer_thread = threading.Thread(target=self.sample_producer)
        self.producer_thread.start()
        
        # 4. Launch terminal display loop
        self.display_thread = threading.Thread(target=self.display_loop)
        self.display_thread.daemon = True
        self.display_thread.start()
        
        try:
            self.producer_thread.join()
            
            # Wait for ingestion queue to empty first, while workers and logger are still active
            self.iq_queue.join()
            
            # Now signal workers and loops to stop
            self.running = False
            
            # Wait for log queue to empty
            self.log_queue.join()
            
            for t in self.workers:
                t.join()
                
            self.logger_thread.join()
            
            print(f"\n[+] Spatial execution completed. Processed: {self.processed_blocks}, Drops: {self.dropped_blocks}")
            
        except KeyboardInterrupt:
            print("\n[!] Force exiting...")
            self.running = False
            self.producer_thread.join()
            for t in self.workers:
                t.join()
            self.logger_thread.join()
            print("[+] Terminated successfully.")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="SpaceShield: Parallelized Spatial DSP Hardware Harness.")
    parser.add_argument('-p', '--playback-file', type=str, default=None,
                        help="Path to raw binary complex signal capture (or NPY file) for HIL playback testing.")
    parser.add_argument('-t', '--sample-type', type=str, choices=['float32', 'int16'], default='float32',
                        help="Unpacking sample type for binary playback (default: float32).")
    parser.add_argument('-e', '--endianness', type=str, choices=['little', 'big', 'native'], default='little',
                        help="Unpacking byte-ordering endianness for binary playback (default: little).")
    parser.add_argument('-f', '--fs', type=float, default=2e6,
                        help="Physical sampling clock rate in Hz (e.g. 2.0e6 or 5.0e6, default: 2.0e6).")
    parser.add_argument('-d', '--duration', type=int, default=30,
                        help="Total execution/stream duration in seconds (default: 30).")
    parser.add_argument('-c', '--chunk-size', type=int, default=8192,
                        help="Temporal frame size per channel block (default: 8192).")
    parser.add_argument('-m', '--channels', type=int, default=4,
                        help="Number of antenna array receiver channels (default: 4).")
    
    args = parser.parse_args()
    
    # Run cryptographic license validation audit
    try:
        from license_validator import run_license_audit
        if not run_license_audit():
            sys.exit(1)
    except ImportError as e:
        print(f"[-] Critical Error: License validator module missing: {e}")
        sys.exit(1)
        
    harness = SpatialHardwareHarness(
        duration_sec=args.duration,
        fs=args.fs,
        chunk_size=args.chunk_size,
        num_channels=args.channels,
        playback_file=args.playback_file,
        playback_sample_type=args.sample_type,
        playback_endianness=args.endianness
    )
    harness.execute()

if __name__ == "__main__":
    main()
