#!/usr/bin/env python3
"""
SpaceShield: RF Fingerprinting (RFF) Feature Extractor.
Author: Antigravity AI
Version: 2.0.0

This module provides high-fidelity mathematical estimators packaged in a modular class 
to extract physical-layer transmitter hardware impairments (RFF) from raw complex I/Q streams.
These impairments act as unique hardware signatures ("fingerprints") used to identify
satellite transmitters and distinguish them from spoofers or jammers.
"""

import numpy as np
from scipy import signal

class RFFFeatureExtractor:
    def __init__(self, fs):
        """
        Initializes the feature extractor with the system sampling rate.
        
        Parameters:
          fs (float): System sampling rate in Hz.
        """
        self.fs = fs

    def estimate_cfo(self, iq_data):
        """
        Estimates the Carrier Frequency Offset (CFO) using a first-lag autocorrelation estimator.
        
        Mathematical Formulation:
          Let x[n] be the complex envelope of the received signal.
          The first-lag autocorrelation is:
            R(1) = E[x[n] * conj(x[n-1])]
          Assuming a pure carrier with frequency offset delta_f, the phase change between
          consecutive samples is:
            theta = angle(R(1)) = 2 * pi * delta_f * Ts
          Where Ts = 1 / fs. Hence:
            delta_f = (theta * fs) / (2 * pi)
        """
        if len(iq_data) < 2:
            return 0.0

        r1 = np.mean(iq_data[1:] * np.conj(iq_data[:-1]))
        cfo_rad_per_sample = np.angle(r1)
        cfo_hz = (cfo_rad_per_sample * self.fs) / (2.0 * np.pi)
        return float(cfo_hz)

    def estimate_iq_imbalance(self, iq_data):
        """
        Estimates I/Q mixer amplitude gain imbalance (g) and quadrature phase skew (phi) 
        using blind statistical moments.
        
        Mathematical Formulation:
          Let r_I = Re(x) and r_Q = Im(x) be the In-phase and Quadrature channels.
          
          1. Amplitude Gain Imbalance (g):
             g_linear = sqrt( E[r_Q^2] / E[r_I^2] )
             g_db = 20 * log10(g_linear)
             
          2. Quadrature Phase Skew (phi):
             sin(phi) = E[r_I * r_Q] / sqrt( E[r_I^2] * E[r_Q^2] )
             phi_deg = arcsin(sin(phi)) * (180 / pi)
        """
        r_i = np.real(iq_data)
        r_q = np.imag(iq_data)
        
        var_i = np.mean(r_i**2)
        var_q = np.mean(r_q**2)
        
        if var_i < 1e-12 or var_q < 1e-12:
            return 0.0, 0.0

        # 1. Gain imbalance
        g_linear = np.sqrt(var_q / var_i)
        g_db = 20.0 * np.log10(g_linear + 1e-12)
        
        # 2. Phase skew
        correlation = np.mean(r_i * r_q)
        sin_phi = correlation / (np.sqrt(var_i * var_q) + 1e-12)
        sin_phi = np.clip(sin_phi, -1.0, 1.0)
        phi_rad = np.arcsin(sin_phi)
        phi_deg = np.degrees(phi_rad)
        
        return float(g_db), float(phi_deg)

    def estimate_phase_noise(self, iq_data, cfo_hz):
        """
        Estimates local oscillator phase noise (residual phase jitter).
        
        Mathematical Formulation:
          1. Mix out the estimated CFO from the signal:
             y[n] = x[n] * exp(-j * 2 * pi * cfo_hz * n * Ts)
          2. Compute the instantaneous phase angle sequence:
             theta[n] = unwrap(angle(y[n]))
          3. Perform linear regression on theta[n] to estimate and remove residual frequency drift.
          4. Compute the standard deviation of the residual phase fluctuations (radians).
        """
        n = len(iq_data)
        if n < 10:
            return 0.0
            
        t = np.arange(n) / self.fs
        # Mix out CFO
        iq_corrected = iq_data * np.exp(-1j * 2.0 * np.pi * cfo_hz * t)
        
        # Unwrapped phase angle
        phases = np.unwrap(np.angle(iq_corrected))
        
        # Linear regression to remove slope (residual CFO)
        x = np.arange(n)
        poly = np.polyfit(x, phases, 1)
        phase_residuals = phases - np.polyval(poly, x)
        
        # Calculate standard deviation of phase residuals
        phase_noise_std = np.std(phase_residuals)
        return float(phase_noise_std)

    def calculate_spectral_metrics(self, iq_data):
        """
        Calculates Power Spectral Density (PSD) metrics.
        
        Wiener Entropy (Spectral Flatness):
          Flatness = exp( E[ln(S(f))] ) / E[S(f)]
          Where S(f) is the Welch PSD of the complex envelope. Ranges from 0 (tonal/pure carrier)
          to 1 (white noise).
          
        Spectral Peak Prominence:
          Calculated as the difference in dB between the maximum power peak and the average power floor.
        """
        nperseg = min(256, len(iq_data))
        frequencies, psd = signal.welch(iq_data, self.fs, nperseg=nperseg)
        
        # Normalize PSD to integrate to 1
        psd_norm = psd / (np.sum(psd) + 1e-12)
        
        # Wiener Entropy
        eps = 1e-12
        geometric_mean = np.exp(np.mean(np.log(psd_norm + eps)))
        arithmetic_mean = np.mean(psd_norm)
        spectral_flatness = geometric_mean / (arithmetic_mean + eps)
        
        # Peak Prominence
        psd_db = 10.0 * np.log10(psd + eps)
        max_val = np.max(psd_db)
        mean_val = np.mean(psd_db)
        peak_prominence = max_val - mean_val
        
        return float(spectral_flatness), float(peak_prominence)

    def extract(self, iq_data):
        """
        Extracts all hardware signatures and returns them as a clean dictionary.
        
        Parameters:
          iq_data (np.ndarray): Complex complex64 samples.
          
        Returns:
          dict: Map of extracted float parameters.
        """
        cfo = self.estimate_cfo(iq_data)
        amp_imb, phase_imb = self.estimate_iq_imbalance(iq_data)
        phase_noise = self.estimate_phase_noise(iq_data, cfo)
        flatness, prominence = self.calculate_spectral_metrics(iq_data)
        
        return {
            "cfo_hz": cfo,
            "iq_amp_imbalance_db": amp_imb,
            "iq_phase_imbalance_deg": phase_imb,
            "phase_noise_std_rad": phase_noise,
            "spectral_flatness": flatness,
            "spectral_peak_prominence_db": prominence
        }

if __name__ == "__main__":
    # Self-test block using simulated impairments
    fs = 1e6
    extractor = RFFFeatureExtractor(fs)
    
    t = np.arange(1000) / fs
    cfo_sim = 15000.0 # 15 kHz
    amp_imb = 10**(0.5 / 20.0) # 0.5 dB
    phase_imb = np.radians(3.0) # 3 degrees
    phase_noise_sim = np.random.normal(0, 0.1, len(t)) # 0.1 rad noise
    
    i_ch = np.cos(2 * np.pi * cfo_sim * t + phase_noise_sim)
    q_ch = amp_imb * np.sin(2 * np.pi * cfo_sim * t + phase_noise_sim + phase_imb)
    iq_sim = i_ch + 1j * q_ch
    
    features = extractor.extract(iq_sim)
    
    print("=" * 60)
    print("        RFF FEATURE EXTRACTOR CLASS DIAGNOSTIC TEST         ")
    print("=" * 60)
    print(f"Target CFO:        {cfo_sim} Hz       | Estimated: {features['cfo_hz']:.2f} Hz")
    print(f"Target Amp Imb:    0.50 dB       | Estimated: {features['iq_amp_imbalance_db']:.2f} dB")
    print(f"Target Phase Imb:  3.00 deg      | Estimated: {features['iq_phase_imbalance_deg']:.2f} deg")
    print(f"Target Phase Noise:0.10 rad      | Estimated: {features['phase_noise_std_rad']:.2f} rad")
    print("-" * 60)
    print(f"Spectral Flatness:  {features['spectral_flatness']:.4f}")
    print(f"Peak Prominence:   {features['spectral_peak_prominence_db']:.2f} dB")
    print("=" * 60)
