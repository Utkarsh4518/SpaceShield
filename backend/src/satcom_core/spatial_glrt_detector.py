#!/usr/bin/env python3
"""
SpaceShield: Multi-Antenna Spatiotemporal GLRT Array Processor.
Author: Principal Space Systems DSP Mathematician & Radar Expert
Version: 1.0.0

Implements multi-channel spatial covariance tracking and spatiotemporal GLRT
(Sphericity Test & METR) to isolate rank-1 coherent terrestrial spoofing.
"""

import numpy as np
from scipy import stats

class SpatialGLRTDetector:
    def __init__(self, num_channels=4, window_size=50, p_fa=1e-7):
        """
        Initializes the spatiotemporal array processor.
        
        Parameters:
          num_channels (int): Number of antenna channels (M).
          window_size (int): Number of temporal samples (N).
          p_fa (float): Target False Alarm Rate (P_fa).
        """
        self.M = num_channels
        self.N = window_size
        self.p_fa = p_fa
        
        # Sphericity Test Degrees of Freedom: df = 0.5 * (M - 1) * (M + 2)
        # For M=4, df = 0.5 * 3 * 6 = 9
        self.df = int(0.5 * (self.M - 1) * (self.M + 2))
        
        # Calculate Bartlett correction factor to scale chi-squared asymptotic fit
        self.rho = 1.0 - (2.0 * self.M**2 + self.M + 2.0) / (6.0 * self.M * (self.N - 1))
        
        # Dynamic Chi-squared alert threshold gamma
        self.gamma = stats.chi2.ppf(1.0 - self.p_fa, df=self.df)

    def compute_spatial_covariance(self, Y):
        """
        Computes the sample spatial covariance matrix:
          R_hat = (1 / N) * Y * Y^H
        where Y is a complex matrix of shape (M, N) and Y^H is its conjugate transpose.
        """
        Y = np.asarray(Y, dtype=np.complex64)
        if Y.shape != (self.M, self.N):
            raise ValueError(f"Input matrix Y must have shape (M, N) = ({self.M}, {self.N}), got {Y.shape}")
            
        # Conjugate transpose (Hermitian) matrix product
        R_hat = np.dot(Y, Y.conj().T) / self.N
        return R_hat

    def evaluate(self, Y):
        """
        Executes the spatiotemporal GLRT on the multi-channel array matrix Y.
        
        Mathematical Formulation:
          1. Sphericity Test (Log-Likelihood Ratio):
             Tests H0 (spherically isotropic/uncorrelated noise & multi-source signals):
               R_0 = sigma^2 * I_M
             Against H1 (structured spatial covariance due to low-rank coherent jammer/spoofer).
             
             The log-likelihood test statistic is:
               Lambda_sphericity = -N * rho * ln( det(R_hat) / (Tr(R_hat)/M)^M )
             
             Under H0: Lambda_sphericity ~ Chi-Squared(df = 0.5 * (M - 1) * (M + 2))
             
          2. Maximum Eigenvalue to Trace Ratio (METR):
             Isolates the rank-1 component (coherent directional injection footprint):
               Lambda_metr = lambda_max / Tr(R_hat)
               
        Parameters:
          Y (np.ndarray): Complex array matrix of shape (M, N).
          
        Returns:
          dict: Summary containing statistic, METR, threshold, and alert verdict.
        """
        R_hat = self.compute_spatial_covariance(Y)
        
        # Compute eigenvalues (R_hat is Hermitian, so eigenvalues are real)
        eigenvalues = np.linalg.eigvalsh(R_hat)
        eigenvalues = np.maximum(eigenvalues, 1e-12) # clip to avoid log(0) or divide-by-zero
        
        # 1. Sphericity Test LLR Calculation
        det_R = np.prod(eigenvalues)
        trace_R = np.sum(eigenvalues)
        
        # Sphericity ratio: det(R) / (Tr(R)/M)^M
        sphericity_ratio = det_R / ((trace_R / self.M) ** self.M)
        sphericity_ratio = np.clip(sphericity_ratio, 1e-15, 1.0)
        
        lambda_sphericity = -self.N * self.rho * np.log(sphericity_ratio)
        
        # 2. METR (Maximum Eigenvalue to Trace Ratio)
        lambda_max = eigenvalues[-1]
        lambda_metr = lambda_max / trace_R
        
        # Verdict decision based on Sphericity LLR (asymptotically Chi-Squared distributed)
        alert_triggered = bool(lambda_sphericity > self.gamma)
        
        return {
            "test_statistic_sphericity": float(lambda_sphericity),
            "threshold_sphericity": float(self.gamma),
            "lambda_metr": float(lambda_metr),
            "eigenvalues": [float(val) for val in eigenvalues],
            "alert_triggered": alert_triggered,
            "degrees_of_freedom": self.df
        }

if __name__ == "__main__":
    # Diagnostics Verification script
    print("=" * 70)
    print("      SPATIOTEMPORAL GLRT ARRAY DETECTOR DIAGNOSTIC CHECKS      ")
    print("=" * 70)
    
    M = 4 # Antennas
    N = 50 # Time window samples
    detector = SpatialGLRTDetector(num_channels=M, window_size=N, p_fa=1e-7)
    
    print(f"[*] Configuration:")
    print(f"    - Antennas (M):         {detector.M}")
    print(f"    - Temporal Window (N):  {detector.N}")
    print(f"    - Bartlett Factor (rho): {detector.rho:.5f}")
    print(f"    - Sphericity DoF:       {detector.df}")
    print(f"    - Chi-Squared Threshold: {detector.gamma:.4f}")
    print("-" * 70)

    # Scenario 1: H0 - Spatial White Noise (isotropic multi-path model)
    print("[*] Generating H0 Scenario: Isotropic Spatial Noise...")
    # Complex normal noise (uncorrelated channels)
    Y_h0 = (np.random.normal(0, 1, (M, N)) + 1j * np.random.normal(0, 1, (M, N))) / np.sqrt(2)
    
    res_h0 = detector.evaluate(Y_h0)
    print(f"    - Sphericity Statistic: {res_h0['test_statistic_sphericity']:.4f}")
    print(f"    - Lambda METR:          {res_h0['lambda_metr']:.4f} (Ideal isotropic: ~0.25)")
    print(f"    - Eigenvalues:          {res_h0['eigenvalues']}")
    print(f"    - Alert Triggered?      {res_h0['alert_triggered']}")
    print("-" * 70)

    # Scenario 2: H1 - Coherent Rank-1 Terrestrial Spoofing Emitter
    print("[*] Generating H1 Scenario: Coherent Terrestrial Spoofing Emitter...")
    # Legitimate background noise
    noise = (np.random.normal(0, 0.5, (M, N)) + 1j * np.random.normal(0, 0.5, (M, N))) / np.sqrt(2)
    
    # Coherent spoofer arriving from a single steer vector a_steering
    a_steering = np.array([1.0, 1j, -1.0, -1j]).reshape(M, 1) # Phase shifts across arrays
    spoofer_waveform = np.random.normal(0, 2.0, (1, N)) + 1j * np.random.normal(0, 2.0, (1, N))
    
    # Received spatial signal under H1: steer * waveform + noise
    Y_h1 = np.dot(a_steering, spoofer_waveform) + noise
    
    res_h1 = detector.evaluate(Y_h1)
    print(f"    - Sphericity Statistic: {res_h1['test_statistic_sphericity']:.4f}")
    print(f"    - Lambda METR:          {res_h1['lambda_metr']:.4f} (Ideal rank-1 dominant: -> 1.0)")
    print(f"    - Eigenvalues:          {res_h1['eigenvalues']}")
    print(f"    - Alert Triggered?      {res_h1['alert_triggered']}")
    print("=" * 70)
