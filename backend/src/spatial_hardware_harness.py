#!/usr/bin/env python3
"""
SpaceShield: Parallelized Spatial DSP Hardware Harness.
Author: Principal Embedded Software Architect & Real-Time DSP Engineer
Version: 2.1.0

Integrates:
1. Spatiotemporal Matrix Ingestion (M=4 channels, N=50 temporal window).
2. Live Hardware Ingestion via SoapyReceiverBridge (Zero-Allocation buffers).
3. Asynchronous Worker Threads (os.cpu_count()) managing thread-local state.
4. Hierarchical Spatial Fast-Path Cascade using Bartlett-corrected sphericity.
5. Lock-Free Asynchronous WORM Logger.
6. Real-time console instrumentation and JSON WebSocket backend integration.
"""

import os
import sys
import time
import queue
import threading
import numpy as np
import ctypes

def enable_ansi_support():
    if os.name == 'nt':
        try:
            kernel32 = ctypes.windll.kernel32
            h_stdout = kernel32.GetStdHandle(-11)
            mode = ctypes.c_ulong()
            if kernel32.GetConsoleMode(h_stdout, ctypes.byref(mode)):
                mode.value |= 4
                kernel32.SetConsoleMode(h_stdout, mode)
        except Exception:
            pass

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from rff_feature_extractor import RFFFeatureExtractor
    from glr_detector import GLRDetector, TrackingLoopObserver
    from spatial_glrt_detector import SpatialGLRTDetector
    from rf_threat_simulator import write_certin_compliance_log
    from edge_inference_engine import EdgeInferenceEngine
    from array_calibration_engine import ArrayCalibrationEngine
    from soapy_receiver_bridge import SoapyReceiverBridge
except ImportError as e:
    print(f"[-] Missing core dependency imports: {e}")
    sys.exit(1)

class SpatialHardwareHarness:
    def __init__(self, duration_sec=30, fs=5.0e6, chunk_size=8192, num_channels=4, playback_file=None, live_mode=False, telemetry_queue=None):
        self.duration_sec = duration_sec
        self.fs = fs
        self.chunk_size = chunk_size
        self.M = num_channels
        self.N = 50
        self.running = False
        self.live_mode = live_mode
        self.playback_file = playback_file
        self.telemetry_queue = telemetry_queue
        
        # High-priority thread-safe queues
        self.iq_queue = queue.Queue(maxsize=1000)
        self.log_queue = queue.Queue(maxsize=2000)
        
        self.spatial_detector = SpatialGLRTDetector(num_channels=self.M, window_size=self.N, p_fa=1e-7)
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
        
        # Dashboard telemetry registers
        self.last_sphericity_stat = 0.0
        self.last_metr = 0.25
        self.last_lambda_max = 1.0
        self.last_verdict = "NORMAL"
        self.last_cfo = 0.0
        self.latency_records = []
        self.inference_latencies = []
        self.last_log_time = 0.0
        
        # Runtime Config
        self.shared_gamma = 50.17
        self.antenna_attenuation_db = 0.0
        self.zero_trust_lockout = False
        
        enable_ansi_support()
        
        self.engine = EdgeInferenceEngine()
        self.engine.initialize_engine()
        
        self.calibration_engine = ArrayCalibrationEngine(num_channels=self.M, chunk_size=self.chunk_size)
        self.calibration_lock = threading.Lock()
        
        # We retain the pool for simulation fallback, but SoapyReceiverBridge handles its own pool in live mode
        self.pool_size = 1050
        self.buffer_pool = [np.zeros((self.M, self.chunk_size), dtype=np.complex64) for _ in range(self.pool_size)]
        self.pool_index = 0
        self.rng = np.random.default_rng()
        
        # 24-Thread DSP Pool scaling (or max os cores)
        self.num_workers = min(os.cpu_count() or 4, 24)
        self.workers = []
        self.producer_thread = None
        self.logger_thread = None
        self.display_thread = None
        self.soapy_bridge = None
        
        self.comp_dir = "compliance"
        self.data_dir = "data"
        os.makedirs(self.comp_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)

    def sample_producer(self):
        """Fallback simulated phase-coherent synchronized stream if not in Live SDR Mode."""
        if self.playback_file:
            from binary_file_player import BinaryFilePlayer
            player = BinaryFilePlayer(file_path=self.playback_file, fs=self.fs, num_channels=self.M, chunk_size=self.chunk_size)
            player.stream_to_harness(self)
            return

        start_time = time.time()
        t_step = self.chunk_size / self.fs
        next_time = start_time
        
        theta_auth = np.radians(15.0)
        a_auth = np.exp(1j * np.arange(self.M) * np.sin(theta_auth)).reshape(self.M, 1)
        theta_spoofer = np.radians(45.0)
        a_spoofer = np.exp(1j * np.arange(self.M) * np.sin(theta_spoofer)).reshape(self.M, 1)
        
        while self.running and (time.time() - start_time) < self.duration_sec:
            elapsed = time.time() - start_time
            next_time += t_step
            
            if elapsed < 10.0: scenario = "normal"
            elif elapsed < 20.0: scenario = "jamming"
            else: scenario = "spoofing"
                
            t = np.arange(self.chunk_size) / self.fs
            iq_signal = self.buffer_pool[self.pool_index]
            self.pool_index = (self.pool_index + 1) % self.pool_size
            
            iq_signal.real = self.rng.normal(0.0, np.sqrt(0.05), size=(self.M, self.chunk_size))
            iq_signal.imag = self.rng.normal(0.0, np.sqrt(0.05), size=(self.M, self.chunk_size))
            
            if scenario == "normal":
                iq_signal += a_auth * (0.02 * np.exp(1j * 2 * np.pi * 50.0 * t))
            elif scenario == "jamming":
                jam_w = self.rng.normal(0.0, np.sqrt(8.0), size=(1, self.chunk_size)) + 1j * self.rng.normal(0.0, np.sqrt(8.0), size=(1, self.chunk_size))
                iq_signal += a_spoofer * jam_w
            elif scenario == "spoofing":
                iq_signal += a_spoofer * (0.8 * np.exp(1j * 2 * np.pi * 15000.0 * t))
                
            try:
                self.iq_queue.put_nowait((iq_signal, scenario, time.perf_counter()))
            except queue.Full:
                with self.metrics_lock: self.dropped_blocks += 1
                
            sleep_time = next_time - time.time()
            if sleep_time > 0: time.sleep(sleep_time)

    def logging_worker(self):
        while self.running or not self.log_queue.empty():
            try:
                try:
                    scenario, sim_features, classification_result = self.log_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                write_certin_compliance_log(scenario, sim_features, classification_result, self.comp_dir, self.data_dir)
                self.log_queue.task_done()
            except Exception:
                pass

    def emergency_lockdown(self):
        with self.iq_queue.mutex:
            self.iq_queue.queue.clear()
        import json
        with open("../../compliance/certin_incident_spoofing.json", "w") as f:
            json.dump({"EMERGENCY_LOCKDOWN": True, "timestamp": time.time(), "reason": "ZERO_TRUST_LOCKOUT_TRIGGERED"}, f)
        print("\n\033[1;41;37m[!] CRITICAL: ZERO-TRUST LOCKOUT INITIATED. QUEUES PURGED.\033[0m\n")

    def processing_worker(self, worker_id):
        local_detector = SpatialGLRTDetector(num_channels=self.M, window_size=self.N, p_fa=1e-7)
        local_extractor = RFFFeatureExtractor(self.fs)
        calibrated_packet = np.zeros((self.M, self.chunk_size), dtype=np.complex64)
        
        while self.running or not self.iq_queue.empty():
            with self.metrics_lock:
                current_gamma = self.shared_gamma
                current_atten = self.antenna_attenuation_db
                lockout_active = self.zero_trust_lockout
            
            if lockout_active:
                time.sleep(0.1)
                continue
                
            local_detector.gamma = current_gamma
            
            try:
                try:
                    payload = self.iq_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                
                # Unpack either dictionary (from Soapy Bridge) or tuple (from Synthesizer)
                if isinstance(payload, dict):
                    iq_packet = payload["iq_data"]
                    scenario = "live"
                    enqueued_time = payload["timestamp"]
                else:
                    iq_packet, scenario, enqueued_time = payload
                
                if current_atten > 0.0:
                    iq_packet *= (10 ** (-current_atten / 20.0))
                
                t_start = time.perf_counter()
                try:
                    if not self.calibration_engine.calibrated:
                        with self.calibration_lock:
                            if not self.calibration_engine.calibrated:
                                self.calibration_engine.calibrate(iq_packet)
                                
                    self.calibration_engine.equalize(iq_packet, calibrated_packet)
                    
                    sphericity_alert = False
                    metr_breach = False
                    max_sphericity = 0.0
                    max_metr = 0.0
                    max_lambda_max = 0.0
                    
                    num_snapshots = self.chunk_size // self.N
                    for idx in range(num_snapshots):
                        start_idx = idx * self.N
                        Y_k = calibrated_packet[:, start_idx : start_idx + self.N]
                        
                        spatial_res = local_detector.evaluate(Y_k)
                        stat, metr, l_max = spatial_res["test_statistic_sphericity"], spatial_res["lambda_metr"], spatial_res["eigenvalues"][-1]
                        
                        max_sphericity = max(max_sphericity, stat)
                        max_metr = max(max_metr, metr)
                        max_lambda_max = max(max_lambda_max, l_max)
                            
                        if spatial_res["alert_triggered"]: sphericity_alert = True
                        if metr > 0.5: metr_breach = True
                            
                    h1_triggered = sphericity_alert or metr_breach
                    
                    if not h1_triggered:
                        with self.metrics_lock: self.fast_path_hits += 1
                        rff_cfo, rff_phase_noise, rff_iq_amp, rff_iq_phase, spectral_flatness = 5.0, 0.02, 0.05, 0.5, 0.3
                        beta = 0.99
                    else:
                        ch0_data = calibrated_packet[0, :]
                        features = local_extractor.extract(ch0_data)
                        
                        total_power = np.mean(np.abs(ch0_data)**2)
                        x_norm = ch0_data / (np.sqrt(total_power) + 1e-12)
                        beta = 1.0 - np.abs(np.mean(x_norm**2))**2
                        
                        rff_cfo = features["cfo_hz"]
                        rff_phase_noise = features["phase_noise_std_rad"]
                        rff_iq_amp = features["iq_amp_imbalance_db"]
                        rff_iq_phase = features["iq_phase_imbalance_deg"]
                        spectral_flatness = features["spectral_flatness"]
                        
                    total_ch0_power = np.mean(np.abs(calibrated_packet[0, :])**2)
                    noise_power = 0.05
                    if scenario == "jamming": in_ratio_db = 10 * np.log10(8.0 / noise_power)
                    elif scenario == "spoofing" or h1_triggered: in_ratio_db = 10 * np.log10(0.8**2 / noise_power)
                    else: in_ratio_db = -12.5
                        
                    prominence = features["spectral_peak_prominence_db"] if (h1_triggered and 'features' in locals()) else 41.5
                    rf_fingerprint_dict = {
                        "cfo_hz": rff_cfo, "phase_noise_std_rad": rff_phase_noise,
                        "iq_amp_imbalance_db": rff_iq_amp, "iq_phase_imbalance_deg": rff_iq_phase,
                        "spectral_flatness": spectral_flatness, "spectral_peak_prominence_db": prominence
                    }
                    
                    verdict, probability, metrics = self.engine.classify(rf_fingerprint_dict)
                    threat_score = probability * 100.0
                    
                    with self.metrics_lock:
                        self.last_sphericity_stat = max_sphericity
                        self.last_metr = max_metr
                        self.last_lambda_max = max_lambda_max
                        self.last_verdict = verdict
                        self.last_cfo = rff_cfo
                    
                    classification_result = {
                        "verdict": verdict, "threat_score": threat_score,
                        "itu_compliance": "VIOLATION" if verdict != "NORMAL" else "COMPLIANT",
                        "glr_alert": h1_triggered,
                        "indicators": [f"Spatial Array: Sphericity Alert={sphericity_alert}, METR={max_metr:.4f}", f"Edge-AI: Latency={metrics['inference_latency_us']:.1f} us"]
                    }
                    
                    sim_features = {
                        "rms_power_db": float(10 * np.log10(total_ch0_power + 1e-12)),
                        "glr_statistic": float(max_sphericity), "glr_threshold": float(local_detector.gamma),
                        "beta": float(beta), "rff": {"cfo": rff_cfo}
                    }
                    
                    if h1_triggered:
                        current_time = time.time()
                        should_log = False
                        with self.log_lock:
                            if current_time - self.last_log_time >= 0.5:
                                self.last_log_time = current_time
                                should_log = True
                        if should_log:
                            with self.log_lock:
                                try: self.log_queue.put_nowait((scenario, sim_features, classification_result))
                                except queue.Full: pass
                        with self.metrics_lock: self.alerts_count += 1
                            
                    t_end = time.perf_counter()
                    with self.latency_lock:
                        self.latency_records.append((t_end - t_start) * 1000.0)
                        if len(self.latency_records) > 1000: self.latency_records.pop(0)
                        self.inference_latencies.append(metrics["inference_latency_us"])
                        if len(self.inference_latencies) > 1000: self.inference_latencies.pop(0)
                            
                    with self.metrics_lock: self.processed_blocks += 1
                        
                finally:
                    self.iq_queue.task_done()
                    
            except Exception as e:
                print(f"[!] Error in processing thread {worker_id}: {e}")

    def display_loop(self):
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()
        
        while self.running:
            time.sleep(0.1)
            
            with self.latency_lock:
                avg_latency = np.mean(self.latency_records) if self.latency_records else 0.0
                last_latency = self.latency_records[-1] if self.latency_records else 0.0
                avg_infer = np.mean(self.inference_latencies) if self.inference_latencies else 0.0
                
            with self.metrics_lock:
                proc, drop, alerts, fast_hits = self.processed_blocks, self.dropped_blocks, self.alerts_count, self.fast_path_hits
                sphericity, metr, lambda_max = self.last_sphericity_stat, self.last_metr, self.last_lambda_max
                verdict, last_cfo = getattr(self, 'last_verdict', 'NORMAL'), getattr(self, 'last_cfo', 0.0)
                
            queue_pct = (self.iq_queue.qsize() / self.iq_queue.maxsize) * 100.0
            
            try:
                import json
                if self.telemetry_queue is not None:
                    payload = {
                        "sphericity_score": float(sphericity),
                        "fim_beta": float(metr),
                        "threat_verdict": str(verdict),
                        "inference_latency_us": float(avg_infer),
                        "dropped_blocks": int(drop)
                    }
                    if self.telemetry_queue.full():
                        try: self.telemetry_queue.get_nowait()
                        except queue.Empty: pass
                    self.telemetry_queue.put_nowait(payload)
            except Exception:
                pass
            
            status_box = f"\033[1;41;37m[ CRITICAL ALERT: SPOOFING DETECTED ]\033[0m" if verdict == "SPOOFING" else f"\033[1;33;41m[ CRITICAL ALERT: JAMMING DETECTED ]\033[0m" if verdict == "JAMMING" else f"\033[1;32m[ STATUS: NORMAL - ALL SIGNALS COMPLIANT ]\033[0m"
            drop_color = "\033[32m" if drop == 0 else "\033[31m"
            
            with self.print_lock:
                sys.stdout.write("\033[H")
                hud_str = [
                    "\033[36m================================================================================",
                    "   ____                     ____  _     _      _     _ ",
                    "  / ___| _ __   __ _  ___  / ___|| |__ (_) ___| | __| |",
                    "  \\___ \\| '_ \\ / _` |/ __| \\___ \\| '_ \\| |/ _ \\ |/ _` |",
                    "   ___) | |_) | (_| | (__   ___) | | | | |  __/ | (_| |",
                    "  |____/| .__/ \\__,_|\\___| |____/|_| |_|_|\\___|_|\\__,_|",
                    "        |_|                                            ",
                    f"                 --- GROUND STATION THREAT MONITOR ({'LIVE SDR' if self.live_mode else 'HIL'}) ---",
                    "================================================================================\033[0m",
                    f"\033[1;35m[ PIPELINE METRICS PANEL ]\033[0m",
                    f"  Processed Blocks: {proc:<6} | Dropped Blocks: {drop_color}{drop:<5}\033[0m | Active Workers: {self.num_workers}",
                    f"  Ingestion Queue:  {queue_pct:.1f}%  | DSP Latency:    {last_latency:.2f} ms (avg: {avg_latency:.2f} ms)",
                    f"  Model Inference:  {avg_infer:.2f} us | AI Provider:    {self.engine.active_provider}",
                    "-" * 80,
                    f"\033[1;33m[ SIGNAL INTEGRITY PANEL ]\033[0m",
                    f"  Sphericity LLR Stat: {sphericity:.4f} (Threshold: {self.spatial_detector.gamma:.2f})",
                    f"  METR Anisotropy:     {metr:.4f} (Isotropic: ~0.25 | Breach: -> 1.0)",
                    f"  Max Eigenvalue:      {lambda_max:.4f}",
                    f"  Carrier Offset (CFO): {last_cfo:+.2f} Hz",
                    "-" * 80,
                    f"\033[1;34m[ ACTIVE THREAT STATUS DISPLAY ]\033[0m",
                    f"  {status_box}",
                    "\033[36m================================================================================\033[0m"
                ]
                sys.stdout.write("\n".join(hud_str) + "\n")
                sys.stdout.flush()

    def execute(self):
        self.running = True
        self.logger_thread = threading.Thread(target=self.logging_worker, daemon=True)
        self.logger_thread.start()
        
        for i in range(self.num_workers):
            t = threading.Thread(target=self.processing_worker, args=(i,), daemon=True)
            self.workers.append(t)
            t.start()
            
        if self.live_mode:
            try:
                self.soapy_bridge = SoapyReceiverBridge(target_queue=self.iq_queue, sample_rate_msps=self.fs/1e6, chunk_size=self.chunk_size, num_channels=self.M)
                self.soapy_bridge.initialize_hardware()
                self.soapy_bridge.start()
            except Exception as e:
                print(f"[!] Failed to initialize Live SDR hardware. Fallback to simulation. Error: {e}")
                self.live_mode = False

        if not self.live_mode:
            self.producer_thread = threading.Thread(target=self.sample_producer, daemon=True)
            self.producer_thread.start()
            
        self.display_thread = threading.Thread(target=self.display_loop, daemon=True)
        self.display_thread.start()
        
        try:
            start_t = time.time()
            while self.running and (time.time() - start_t) < self.duration_sec:
                time.sleep(1.0)
            self.running = False
            self.iq_queue.join()
            self.log_queue.join()
            if self.soapy_bridge: self.soapy_bridge.stop()
        except KeyboardInterrupt:
            self.running = False
            if self.soapy_bridge: self.soapy_bridge.stop()
        finally:
            print(f"\n[+] Spatial execution completed. Processed: {self.processed_blocks}, Drops: {self.dropped_blocks}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="SpaceShield: Parallelized Spatial DSP Hardware Harness.")
    parser.add_argument('-p', '--playback-file', type=str, default=None)
    parser.add_argument('-l', '--live', action='store_true', help="Enable Live SDR Hardware Ingestion (via SoapySDR).")
    parser.add_argument('-f', '--fs', type=float, default=5.0e6)
    parser.add_argument('-d', '--duration', type=int, default=30)
    parser.add_argument('-c', '--chunk-size', type=int, default=8192)
    parser.add_argument('-m', '--channels', type=int, default=4)
    args = parser.parse_args()
    
    try:
        from license_validator import run_license_audit
        if not run_license_audit(): sys.exit(1)
    except ImportError:
        pass
        
    harness = SpatialHardwareHarness(
        duration_sec=args.duration, fs=args.fs, chunk_size=args.chunk_size,
        num_channels=args.channels, playback_file=args.playback_file, live_mode=args.live
    )
    harness.execute()

if __name__ == "__main__":
    main()
