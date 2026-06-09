#!/usr/bin/env python3
"""
SpaceShield: High-Throughput Edge Inference Engine with Dynamic Micro-Batching.
Author: Antigravity AI
Version: 2.0.0

This module implements an optimized asynchronous classification worker using ONNX Runtime.
It leverages TensorRT FP16 quantization on Jetson platforms, implements dynamic 
micro-batching to maximize throughput without adding latency, and runs non-blocking
concurrent inference over streamed RF signatures.
"""

import os
import sys
import time
import queue
import threading
import numpy as np

# Optional ONNX / ONNX Runtime imports
try:
    import onnx
    from onnx import helper, TensorProto
    import onnxruntime as ort
except ImportError:
    onnx = None
    ort = None
    print("[!] ONNX / ONNX Runtime libraries not found.")
    print("[!] To run hardware inference, execute: pip install onnx onnxruntime")

class EdgeInferenceEngine:
    def __init__(self, model_path="data/rff_classifier.onnx", max_batch_size=16, latency_timeout_sec=0.005):
        """
        Initializes the Inference Engine.
        
        Parameters:
          model_path (str): Path to the compiled ONNX model.
          max_batch_size (int): Maximum batch size for dynamic micro-batching.
          latency_timeout_sec (float): Max wait time (in seconds) to accumulate a batch.
        """
        self.model_path = model_path
        self.max_batch_size = max_batch_size
        self.latency_timeout = latency_timeout_sec
        self.session = None
        self.providers = []
        self.running = False
        
        # Thread-safe queues
        self.input_queue = queue.Queue(maxsize=1000)
        self.output_queue = queue.Queue()
        
        self.worker_thread = None
        self.input_dim = 6 # RFF feature dimensions

    def create_dummy_onnx_model(self):
        """Generates a dummy MLP ONNX classifier for self-contained testing."""
        if onnx is None:
            print("[!] Cannot generate dummy ONNX model (onnx library missing). Using mock inference.")
            return False

        print("[*] Generating dummy ONNX RFF classifier model...")
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        # Graph inputs and outputs
        input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [None, self.input_dim])
        output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [None, 3]) # [Normal, Jamming, Spoofing]

        # Initialize Weights (D x C) and Biases
        w_data = np.random.normal(0, 0.1, (self.input_dim, 3)).astype(np.float32)
        b_data = np.zeros(3, dtype=np.float32)

        w_tensor = helper.make_tensor('W1', TensorProto.FLOAT, [self.input_dim, 3], w_data.flatten())
        b_tensor = helper.make_tensor('B1', TensorProto.FLOAT, [3], b_data.flatten())

        # Mathematical Nodes
        node_matmul = helper.make_node('MatMul', ['input', 'W1'], ['matmul_out'])
        node_add = helper.make_node('Add', ['matmul_out', 'B1'], ['output'])

        graph = helper.make_graph(
            [node_matmul, node_add],
            'rff_classifier_graph',
            [input_tensor],
            [output_tensor],
            initializer=[w_tensor, b_tensor]
        )

        model = helper.make_model(graph, producer_name='SpaceShield-Compiler')
        model.opset_import[0].version = 13

        onnx.save(model, self.model_path)
        print(f"[+] Dummy ONNX model saved to: {self.model_path}")
        return True

    def initialize_engine(self):
        """Compiles and loads the model into ONNX Runtime with optimized execution providers."""
        if ort is None:
            print("[!] ONNX Runtime is not installed. Falling back to CPU Mock mode.")
            return False

        if not os.path.exists(self.model_path):
            success = self.create_dummy_onnx_model()
            if not success:
                return False

        try:
            print(f"[*] Initializing ONNX Runtime Session for: {self.model_path}")
            
            # Identify hardware acceleration execution providers
            available_providers = ort.get_available_providers()
            print(f"[*] Available hardware execution providers: {available_providers}")

            self.providers = []
            
            # 1. Jetson TensorRT Execution Provider configuration
            if 'TensorrtExecutionProvider' in available_providers:
                trt_options = {
                    'device_id': 0,
                    'trt_max_workspace_size': 1 << 30, # 1 GB
                    'trt_fp16_enable': True, # Enable FP16 execution pipeline
                    'trt_builder_optimization_level': 5
                }
                self.providers.append(('TensorrtExecutionProvider', trt_options))
                print("[+] TensorRT Hardware Acceleration selected (FP16 Enabled).")
            
            # 2. CUDA Execution Provider configuration
            if 'CUDAExecutionProvider' in available_providers:
                self.providers.append('CUDAExecutionProvider')
                print("[+] CUDA Accelerator selected.")
                
            # 3. CPU Execution Provider (Fallback)
            self.providers.append('CPUExecutionProvider')

            # Enable graph optimizations (GraphOptimizationLevel.ORT_ENABLE_ALL)
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

            self.session = ort.InferenceSession(self.model_path, sess_options, providers=self.providers)
            print("[+] Inference session compiled successfully.")
            return True
        except Exception as e:
            print(f"[-] Failed to compile execution engine: {e}")
            return False

    def run_inference_batch(self, feature_batch):
        """Runs concurrent inference over a 2D batch tensor array."""
        input_data = np.array(feature_batch, dtype=np.float32)
        
        # Run inference session
        outputs = self.session.run(None, {'input': input_data})
        batch_probs = outputs[0]
        
        results = []
        for probs in batch_probs:
            # Softmax
            exp_probs = np.exp(probs - np.max(probs))
            softmax_probs = exp_probs / np.sum(exp_probs)
            
            classes = ["NORMAL", "JAMMING", "SPOOFING"]
            verdict = classes[np.argmax(softmax_probs)]
            threat_score = float(np.max(softmax_probs) * 100.0)
            results.append((verdict, threat_score))
            
        return results

    def run_inference_mock_batch(self, feature_batch):
        """Fallback mock batch calculation when ONNX is unavailable."""
        results = []
        for feat in feature_batch:
            cfo = feat[0]
            flatness = feat[4]
            if flatness > 0.6:
                results.append(("JAMMING", 99.5))
            elif cfo > 50.0:
                results.append(("SPOOFING", 92.4))
            else:
                results.append(("NORMAL", 9.8))
        return results

    def inference_worker(self):
        """Asynchronous worker loop employing dynamic micro-batching to drain the queue."""
        print("[*] Asynchronous edge inference worker active.")
        
        while self.running:
            batch_features = []
            batch_timestamps = []
            
            try:
                # 1. Blocking wait for the first item to arrive (avoids busy waiting CPU spike)
                try:
                    feat, ts = self.input_queue.get(timeout=0.2)
                    batch_features.append(feat)
                    batch_timestamps.append(ts)
                except queue.Empty:
                    continue

                # 2. Dynamic draining: accumulate additional items up to max_batch_size
                start_accum_time = time.perf_counter()
                while len(batch_features) < self.max_batch_size:
                    # Check if timeout exceeded
                    elapsed = time.perf_counter() - start_accum_time
                    if elapsed > self.latency_timeout:
                        break
                        
                    try:
                        # Non-blocking pull for consecutive items
                        feat, ts = self.input_queue.get_nowait()
                        batch_features.append(feat)
                        batch_timestamps.append(ts)
                    except queue.Empty:
                        # Queue exhausted for this epoch
                        break

                # 3. Perform batch inference
                t_infer_start = time.perf_counter()
                if self.session:
                    batch_results = self.run_inference_batch(batch_features)
                else:
                    batch_results = self.run_inference_mock_batch(batch_features)
                
                t_infer_end = time.perf_counter()
                batch_inference_us = (t_infer_end - t_infer_start) * 1e6

                # 4. Asynchronous Verdict Dissemination
                for idx, (verdict, score) in enumerate(batch_results):
                    capture_ts = batch_timestamps[idx]
                    total_latency_us = (t_infer_end - capture_ts) * 1e6
                    
                    self.output_queue.put({
                        "verdict": verdict,
                        "threat_score": score,
                        "inference_latency_us": batch_inference_us / len(batch_results),
                        "total_latency_us": total_latency_us,
                        "batch_size": len(batch_features),
                        "timestamp": capture_ts
                    })
                    self.input_queue.task_done()
                    
            except Exception as e:
                print(f"[!] Error in inference worker loop: {e}")

    def enqueue(self, feature_vector):
        """Pushes feature vectors to the input queue without blocking the ingestion thread."""
        try:
            self.input_queue.put_nowait((feature_vector, time.perf_counter()))
            return True
        except queue.Full:
            # Evict oldest to maintain low-latency circular nature
            try:
                self.input_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.input_queue.put_nowait((feature_vector, time.perf_counter()))
                return True
            except queue.Full:
                return False

    def start(self):
        """Starts the worker daemon thread."""
        self.running = True
        self.worker_thread = threading.Thread(target=self.inference_worker)
        self.worker_thread.daemon = True
        self.worker_thread.start()

    def stop(self):
        """Gracefully stops the worker thread."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join()
        print("[*] Asynchronous edge inference worker stopped.")

    def run_benchmark(self):
        """Benchmarks execution metrics under varying micro-batch workloads."""
        print("=" * 70)
        print("         SPACESHIELD EDGE INFERENCE BENCHMARK RUN          ")
        print("=" * 70)
        
        dummy_feat = [148.5, 0.78, 4.3, 0.14, 0.11, 41.5]
        
        # Test varying batch sizes
        test_batches = [1, 4, 8, 16]
        
        for bs in test_batches:
            batch_input = [dummy_feat for _ in range(bs)]
            
            iterations = 100
            timings = []
            
            for _ in range(iterations):
                t_start = time.perf_counter()
                if self.session:
                    self.run_inference_batch(batch_input)
                else:
                    self.run_inference_mock_batch(batch_input)
                timings.append((time.perf_counter() - t_start) * 1e6)
                
            avg_us = np.mean(timings)
            per_sample_us = avg_us / bs
            
            print(f"[*] Workload: Batch Size = {bs:<2} | Avg Batch Execution: {avg_us:.1f} µs | "
                  f"Per-sample Latency: {per_sample_us:.2f} µs")
            
        print("-" * 70)
        print(f"Execution Provider: {self.session.get_providers() if self.session else 'CPU-Mock'}")
        print("=" * 70)

def main():
    engine = EdgeInferenceEngine()
    engine.initialize_engine()
    engine.run_benchmark()
    
    # Asynchronous Dynamic Batch Queue Test
    engine.start()
    
    print("[*] Streaming 20 feature vectors to test dynamic micro-batching...")
    for idx in range(20):
        # CFO, IQ Amp, IQ Phase, Phase Noise, Flatness, Prominence
        sim_feat = [148.5, 0.78, 4.3, 0.14, 0.11, 41.5]
        engine.enqueue(sim_feat)
        # Introduce a micro-sleep to simulate arrival jitter
        time.sleep(0.0005)
        
    time.sleep(0.5)
    
    # Read output verdicts
    while not engine.output_queue.empty():
        res = engine.output_queue.get()
        print(f"[Queue Output] Verdict: {res['verdict']:<8} | Score: {res['threat_score']:.1f}% | "
              f"Batch: {res['batch_size']:<2} | Infer: {res['inference_latency_us']:.1f} µs | "
              f"Total Latency: {res['total_latency_us']:.1f} µs")
        
    engine.stop()

if __name__ == "__main__":
    main()
