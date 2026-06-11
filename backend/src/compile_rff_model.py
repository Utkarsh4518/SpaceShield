#!/usr/bin/env python3
"""
SpaceShield: High-Throughput Edge Device Graph Compiler.
Author: Expert Machine Learning & Compiler Engineer
Version: 1.0.0

This utility:
1. Instantiates a concrete PyTorch classification model for RF features.
2. Synthesizes representative threat data (Normal, Jamming, Spoofing) to train the model.
3. Exports the model to a standard FP32 ONNX graph.
4. Structurally rewrites the ONNX graph weights and constants to FP16 half-precision.
5. Verifies the generated graph loads and runs successfully in the EdgeInferenceEngine.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

# Resolve path imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from edge_inference_engine import EdgeInferenceEngine

# Define the PyTorch classification model (outputs raw logits)
class RFFClassifier(nn.Module):
    def __init__(self, input_dim=6, num_classes=3):
        super(RFFClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )
        
    def forward(self, x):
        return self.network(x)

# Export wrapper that appends Softmax to output probability scores directly
class RFFClassifierExport(nn.Module):
    def __init__(self, trained_model):
        super(RFFClassifierExport, self).__init__()
        self.trained_model = trained_model
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        logits = self.trained_model(x)
        return self.softmax(logits)

def generate_synthetic_dataset(num_samples=1500):
    """Generates a high-fidelity synthetic dataset mimicking space & terrestrial RF footprints."""
    print(f"[*] Generating {num_samples} synthetic training samples...")
    X = []
    y = []
    
    # 0: NORMAL (Authentic NavIC satellite footprint)
    # Low CFO, minimal gain/phase imbalance, low phase noise, low spectral flatness
    for _ in range(num_samples // 3):
        cfo = np.random.normal(5.0, 1.5)
        iq_amp = np.random.normal(0.05, 0.01)
        iq_phase = np.random.normal(0.5, 0.1)
        phase_noise = np.random.normal(0.02, 0.005)
        flatness = np.random.uniform(0.1, 0.25)
        prominence = np.random.normal(41.5, 1.5)
        X.append([cfo, iq_amp, iq_phase, phase_noise, flatness, prominence])
        y.append(0)
        
    # 1: JAMMING (High-power broadband noise)
    # Variable CFO, high gain imbalance, high phase noise, high spectral flatness
    for _ in range(num_samples // 3):
        cfo = np.random.normal(0.0, 50.0)
        iq_amp = np.random.normal(0.1, 0.05)
        iq_phase = np.random.normal(1.0, 0.5)
        phase_noise = np.random.normal(0.05, 0.02)
        flatness = np.random.uniform(0.6, 0.95)
        prominence = np.random.normal(5.0, 3.0)
        X.append([cfo, iq_amp, iq_phase, phase_noise, flatness, prominence])
        y.append(1)
        
    # 2: SPOOFING (Terrestrial coherent spoofer generator)
    # Moderate CFO, high gain imbalance, high phase noise, low spectral flatness
    for _ in range(num_samples // 3):
        cfo = np.random.normal(150.0, 15.0)
        iq_amp = np.random.normal(0.8, 0.08)
        iq_phase = np.random.normal(4.5, 0.4)
        phase_noise = np.random.normal(0.15, 0.02)
        flatness = np.random.uniform(0.1, 0.25)
        prominence = np.random.normal(41.5, 1.5)
        X.append([cfo, iq_amp, iq_phase, phase_noise, flatness, prominence])
        y.append(2)
        
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

def train_model(model, X_train, y_train, epochs=30):
    """Trains the concrete classifier network model."""
    print("[*] Training concrete PyTorch classifier...")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0 or epoch == 1:
            # Check training accuracy
            _, preds = torch.max(outputs, 1)
            acc = (preds == y_train).sum().item() / len(y_train) * 100.0
            print(f"    - Epoch {epoch}/{epochs} | Loss: {loss.item():.4f} | Accuracy: {acc:.1f}%")
            
    print("[+] Training completed successfully.")

def quantize_fp32_to_fp16(input_model_path, output_model_path):
    """
    Structurally rewrites the ONNX graph parameters from float32 to float16.
    Natively rewrites model initializers, constant nodes, value_infos, inputs, and outputs.
    """
    print(f"[*] Compiling FP16 Quantized Model Graph: {input_model_path} -> {output_model_path}")
    model = onnx.load(input_model_path)
    graph = model.graph
    
    # 1. Convert input datatypes to FLOAT16
    for tensor_info in graph.input:
        if tensor_info.type.tensor_type.elem_type == TensorProto.FLOAT:
            tensor_info.type.tensor_type.elem_type = TensorProto.FLOAT16
            
    # 2. Convert output datatypes to FLOAT16
    for tensor_info in graph.output:
        if tensor_info.type.tensor_type.elem_type == TensorProto.FLOAT:
            tensor_info.type.tensor_type.elem_type = TensorProto.FLOAT16
            
    # 3. Convert intermediate value_info to FLOAT16
    for tensor_info in graph.value_info:
        if tensor_info.type.tensor_type.elem_type == TensorProto.FLOAT:
            tensor_info.type.tensor_type.elem_type = TensorProto.FLOAT16

    # 4. Structurally convert model weight/bias initializers to FLOAT16
    new_initializers = []
    for init in graph.initializer:
        if init.data_type == TensorProto.FLOAT:
            # Unpack float32 raw data and cast to float16
            data = numpy_helper.to_array(init)
            data_fp16 = data.astype(np.float16)
            # Create new initializer proto
            new_init = numpy_helper.from_array(data_fp16, name=init.name)
            new_initializers.append(new_init)
        else:
            new_initializers.append(init)
            
    del graph.initializer[:]
    graph.initializer.extend(new_initializers)
    
    # 5. Convert constant node attributes to FLOAT16
    for node in graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value" and attr.t.data_type == TensorProto.FLOAT:
                    data = numpy_helper.to_array(attr.t)
                    data_fp16 = data.astype(np.float16)
                    new_t = numpy_helper.from_array(data_fp16)
                    attr.t.CopyFrom(new_t)
                    
        # 6. Cast intermediate Cast nodes to cast to FLOAT16 instead of FLOAT
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == TensorProto.FLOAT:
                    attr.i = TensorProto.FLOAT16
                    
    # Save the modified, quantized graph
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    onnx.save(model, output_model_path)
    print(f"[+] Custom FP16 half-precision graph saved successfully.")

def main():
    print("=" * 70)
    # 1. Setup paths
    models_dir = "compliance/models"
    os.makedirs(models_dir, exist_ok=True)
    fp32_model_path = os.path.join(models_dir, "rff_classifier_fp32.onnx")
    fp16_model_path = os.path.join(models_dir, "rff_classifier.onnx")
    
    # 2. Dataset and training
    X_train, y_train = generate_synthetic_dataset()
    model = RFFClassifier()
    train_model(model, X_train, y_train)
    
    # 3. Export to FP32 ONNX with appended Softmax layer for dynamic probability scores
    print(f"[*] Serializing PyTorch model layout to standard ONNX: {fp32_model_path}...")
    model.eval()
    export_model = RFFClassifierExport(model)
    export_model.eval()
    dummy_input = torch.randn(1, 6, dtype=torch.float32)
    torch.onnx.export(
        export_model,
        dummy_input,
        fp32_model_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"[+] Model successfully exported to FP32 ONNX.")
    
    # 4. FP16 Quantization rewriting
    quantize_fp32_to_fp16(fp32_model_path, fp16_model_path)
    
    # Clean up temporary FP32 ONNX file
    if os.path.exists(fp32_model_path):
        try:
            os.remove(fp32_model_path)
        except Exception:
            pass

    # 5. Session Verification Layer
    print("\n" + "=" * 70)
    print("          ONNX SESSION VALIDATION LAYER          ")
    print("=" * 70)
    try:
        # Instantiate EdgeInferenceEngine target session
        print(f"[*] Initializing EdgeInferenceEngine using model: {fp16_model_path}")
        engine = EdgeInferenceEngine(model_path=fp16_model_path)
        success = engine.initialize_engine()
        
        if not success or engine.session is None:
            raise RuntimeError("Failed to load compiled ONNX model session.")
            
        print(f"[+] EdgeInferenceEngine loaded successfully.")
        print(f"    - Active Execution Provider: {engine.active_provider}")
        
        # Test verification vectors for Normal, Jamming, and Spoofing
        test_cases = {
            "NORMAL": {"cfo_hz": 4.8, "phase_noise_std_rad": 0.02, "iq_amp_imbalance_db": 0.04, "iq_phase_imbalance_deg": 0.45, "spectral_flatness": 0.18, "spectral_peak_prominence_db": 42.1},
            "JAMMING": {"cfo_hz": -1.2, "phase_noise_std_rad": 0.06, "iq_amp_imbalance_db": 0.12, "iq_phase_imbalance_deg": 1.2, "spectral_flatness": 0.85, "spectral_peak_prominence_db": 6.8},
            "SPOOFING": {"cfo_hz": 145.6, "phase_noise_std_rad": 0.16, "iq_amp_imbalance_db": 0.82, "iq_phase_imbalance_deg": 4.35, "spectral_flatness": 0.21, "spectral_peak_prominence_db": 41.8}
        }
        
        for name, feats in test_cases.items():
            verdict, prob, metrics = engine.classify(feats)
            print(f"[*] Target Scenario: {name:<8} | Verdict: {verdict:<8} | Confidence: {prob*100.0:.1f}% | Latency: {metrics['inference_latency_us']:.1f} µs")
            
        print("=" * 70)
        print("[+] End-to-end model graph compilation and verification successful.")
        
    except Exception as e:
        print(f"[!] Verification Layer Failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
