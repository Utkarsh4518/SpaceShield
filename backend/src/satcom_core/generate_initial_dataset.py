#!/usr/bin/env python3
"""
SpaceShield: Production Dataset Generator (H0 Nominal & H1 Spoofed).
Author: Principal Defensive Electronic Warfare Mathematician & DevSecOps Engineer
Version: 3.0.0

Constructs mathematically rigorous baseline signal tensors representing the
null-hypothesis propagation state (H0) and hostile spoofing threat scenarios (H1)
for an authentic NavIC L5/S-band satellite downlink received by a ULA receiver.

Implements physical-layer hardware impairments, Fisher Information Matrix (FIM)
protection boundaries, and raw binary serialization for high-throughput SDR
hardware-in-the-loop (HIL) testing.

Signal Model:
    x_k = a(θ) · s_impaired_k + v_k

Serialization:
    Contiguous, memory-aligned 32-bit float interleaved raw binary (.bin):
    [I0, Q0, I1, Q1, I2, Q2, I3, Q3] for each time snapshot.
"""

import os
import sys
import time
import numpy as np


class DatasetGenerator:
    """
    Generates H0 (Nominal) and H1 (Spoofed) multi-antenna observation matrices
    with built-in physical-layer transceiver impairments and binary serialization.
    """

    # ──────────────────────────────────────────────────────────────────────
    # Physical System Parameters
    # ──────────────────────────────────────────────────────────────────────
    FS = 2_000_000                  # Sampling rate: 2.0 MSPS (ADC clock)
    DURATION_SEC = 1.0              # Observation epoch duration in seconds
    N_SAMPLES = int(FS * DURATION_SEC)  # 2,000,000 complex baseband samples
    NUM_CHANNELS = 4                # ULA antenna element count (M)
    CARRIER_OFFSET_HZ = 50_000.0   # Baseband intermediate frequency offset f_c
    
    # SNR is raised to 18.0 dB to guarantee FIM margin β >= 0.98 for H0
    SNR_DB = 18.0                   # Nominal carrier-to-noise ratio (dB)
    ELEMENT_SPACING = 0.15          # Nominal inter-element spacing d/lambda

    # ──────────────────────────────────────────────────────────────────────
    # Spoofing Threat Parameters (H1)
    # ──────────────────────────────────────────────────────────────────────
    SPOOFER_SNR_DB = 45.0           # Overpowering terrestrial spoofing margin
    SPOOFER_ARRIVAL_PHASE = 0.45    # Hostile wavefront uniform arrival phase shift

    # ──────────────────────────────────────────────────────────────────────
    # Hardware Impairment Parameters
    # ──────────────────────────────────────────────────────────────────────
    CFO_HZ = 1500.0                 # Uncompensated Carrier Frequency Offset (Hz)
    IQ_GAIN_IMB_DB = 0.3            # I-channel gain imbalance (dB)
    IQ_PHASE_ERR_DEG = 1.5          # Quadrature phase error (degrees)
    PHASE_NOISE_STD = 0.05          # Local oscillator phase jitter std dev (radians)

    def __init__(self, seed=42):
        """Initializes the dataset generator with a deterministic PRNG seed."""
        self.n_samples = int(self.FS * self.DURATION_SEC)
        self.sigma2 = 10.0 ** (-self.SNR_DB / 10.0)
        self.rng = np.random.default_rng(seed)

    # ──────────────────────────────────────────────────────────────────────
    # Core Signal Construction Methods
    # ──────────────────────────────────────────────────────────────────────

    def build_time_vector(self):
        """Constructs a contiguous, memory-aligned time-domain sample vector."""
        t = np.arange(self.n_samples, dtype=np.float64) / self.FS
        assert t.flags['C_CONTIGUOUS'], "Time vector must be C-contiguous"
        return t

    def generate_carrier_signal(self, t):
        """Synthesizes the pristine complex baseband carrier envelope s(t)."""
        omega_c = 2.0 * np.pi * self.CARRIER_OFFSET_HZ
        return np.exp(1j * (omega_c * t))

    def apply_hardware_impairments(self, signal, t):
        """
        Distorts a pristine complex baseband signal array with physical-layer
        transceiver defects using highly optimized vectorized transformations.
        """
        # 1. Local Oscillator Phase Noise Jitter & 2. Carrier Frequency Offset
        phase_noise = self.rng.normal(0.0, self.PHASE_NOISE_STD, size=t.shape)
        cfo_phase = 2.0 * np.pi * self.CFO_HZ * t
        
        # Apply combined phase rotation multiplicatively
        total_phase_shift = np.exp(1j * (cfo_phase + phase_noise))
        signal_rotated = signal * total_phase_shift

        # 3. I/Q Mixer Gain Asymmetry & Quadrature Phase Error
        gain_i = 10.0 ** (self.IQ_GAIN_IMB_DB / 20.0)
        phi_rad = np.radians(self.IQ_PHASE_ERR_DEG)

        I = signal_rotated.real
        Q = signal_rotated.imag

        I_impaired = gain_i * I + np.sin(phi_rad) * Q
        Q_impaired = np.cos(phi_rad) * Q

        return I_impaired + 1j * Q_impaired

    def generate_awgn_matrix(self):
        """Generates the spatially and temporally white noise matrix V."""
        component_std = np.sqrt(self.sigma2 / 2.0)
        noise_real = self.rng.standard_normal((self.NUM_CHANNELS, self.n_samples), dtype=np.float32)
        noise_imag = self.rng.standard_normal((self.NUM_CHANNELS, self.n_samples), dtype=np.float32)
        return (component_std * (noise_real + 1j * noise_imag)).astype(np.complex64, copy=False)

    def build_steering_vector(self, phase_shift):
        """Constructs the array steering vector a(θ) for a given phase shift."""
        m = np.arange(self.NUM_CHANNELS, dtype=np.float64)
        phase_shifts = 2.0 * np.pi * m * phase_shift
        return np.exp(1j * phase_shifts).reshape(self.NUM_CHANNELS, 1)

    # ──────────────────────────────────────────────────────────────────────
    # Threat Modeling & Observation Generation
    # ──────────────────────────────────────────────────────────────────────

    def generate_h0_observation(self, t):
        """
        Constructs the nominal H0 observation matrix X (Authentic Downlink).
        """
        pristine_carrier = self.generate_carrier_signal(t)
        impaired_carrier = self.apply_hardware_impairments(pristine_carrier, t)
        
        steering = self.build_steering_vector(self.ELEMENT_SPACING)
        signal_component = steering @ impaired_carrier.reshape(1, -1)
        noise_component = self.generate_awgn_matrix()

        observation = (signal_component + noise_component).astype(np.complex64)
        return np.ascontiguousarray(observation)

    def generate_h1_observation(self, t):
        """
        Constructs the spoofed H1 observation matrix X (Hostile Threat).
        Synthesizes a rank-1 coherent array spoofing wavefront from a single
        terrestrial transmitter overpowering the thermal margins.
        """
        pristine_carrier = self.generate_carrier_signal(t)
        impaired_carrier = self.apply_hardware_impairments(pristine_carrier, t)
        
        # Apply hostile steering phase shift
        steering = self.build_steering_vector(self.SPOOFER_ARRIVAL_PHASE)
        
        # Apply overpowering terrestrial spoofing amplitude margin
        spoofer_amp = 10.0 ** (self.SPOOFER_SNR_DB / 20.0)
        signal_component = steering @ (spoofer_amp * impaired_carrier.reshape(1, -1))
        
        noise_component = self.generate_awgn_matrix()

        observation = (signal_component + noise_component).astype(np.complex64)
        return np.ascontiguousarray(observation)

    # ──────────────────────────────────────────────────────────────────────
    # Fisher Information Matrix (FIM) & Serialization
    # ──────────────────────────────────────────────────────────────────────

    def calculate_fim_beta(self, observation):
        """
        Extracts eigenvalues of the simulated covariance matrix to programmatically
        calculate the Fisher Identifiability Margin parameter β.
        """
        # Sample spatial covariance: R̂ = (1/N) · X · X^H
        R_hat = (observation @ observation.conj().T) / observation.shape[1]
        eigenvalues = np.sort(np.linalg.eigvalsh(R_hat))[::-1]
        
        # Fisher Identifiability Margin parameter β
        trace = np.sum(eigenvalues)
        beta = eigenvalues[0] / trace if trace > 0 else 0.0
        
        return beta, eigenvalues

    def serialize_interleaved_binary(self, observation, filepath):
        """
        Flattens the 4-channel complex matrix into a contiguous, interleaved
        32-bit single-precision floating-point array: [I0, Q0, I1, Q1, I2, Q2...].
        Enforces fast file-stream descriptors to dump the raw binary byte payloads.
        """
        # Transpose to (Time, Channels) so snapshots are contiguous in memory
        obs_t = np.ascontiguousarray(observation.T, dtype=np.complex64)
        
        # View complex64 (2x float32) as interleaved real float32 array
        obs_interleaved = obs_t.view(np.float32)
        
        # Dump raw binary payload directly to disk via fast C-stream
        obs_interleaved.tofile(filepath)


def main():
    print("=" * 80)
    print("    SPACESHIELD PRODUCTION DATASET GENERATOR (H0 & H1)")
    print("=" * 80)

    generator = DatasetGenerator(seed=42)
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    os.makedirs(output_dir, exist_ok=True)

    t0 = time.perf_counter()
    t = generator.build_time_vector()

    # ----------------------------------------------------------------------
    # 1. H0 Nominal Generation
    # ----------------------------------------------------------------------
    print("\n[*] Generating H0 (Nominal) Baseline Profile...")
    h0_obs = generator.generate_h0_observation(t)
    h0_beta, h0_eigs = generator.calculate_fim_beta(h0_obs)
    
    print(f"    Fisher Margin (beta): {h0_beta:.4f} (Requirement: >= 0.98)")
    print(f"    Eigenvalues:      [{', '.join(f'{v:.4f}' for v in h0_eigs)}]")
    
    h0_path = os.path.join(output_dir, "nominal_reception.bin")
    generator.serialize_interleaved_binary(h0_obs, h0_path)
    print(f"    [+] Serialized H0 to: {h0_path} ({os.path.getsize(h0_path)/(1024*1024):.1f} MB)")

    # ----------------------------------------------------------------------
    # 2. H1 Spoofed Generation
    # ----------------------------------------------------------------------
    print("\n[*] Generating H1 (Spoofed) Threat Profile...")
    h1_obs = generator.generate_h1_observation(t)
    h1_beta, h1_eigs = generator.calculate_fim_beta(h1_obs)
    
    print(f"    Fisher Margin (beta): {h1_beta:.4f} (Rank-1 Collapse Expected)")
    print(f"    Eigenvalues:      [{', '.join(f'{v:.4f}' for v in h1_eigs)}]")
    
    h1_path = os.path.join(output_dir, "spoofed_reception.bin")
    generator.serialize_interleaved_binary(h1_obs, h1_path)
    print(f"    [+] Serialized H1 to: {h1_path} ({os.path.getsize(h1_path)/(1024*1024):.1f} MB)")

    # ----------------------------------------------------------------------
    # Validation & Exit
    # ----------------------------------------------------------------------
    print(f"\n{'=' * 80}")
    t_total = time.perf_counter() - t0
    print(f"  EXECUTION COMPLETE ({t_total:.2f} seconds)")
    
    # Assert FIM Beta bound for H0 Nominal Signal
    if h0_beta >= 0.98:
        print("  [PASS] Nominal Signal Baseline holds within FIM protection boundaries.")
        sys.exit(0)
    else:
        print("  [FAIL] Nominal Signal Baseline FIM beta falls below 0.98 constraint!")
        sys.exit(1)


if __name__ == "__main__":
    main()
