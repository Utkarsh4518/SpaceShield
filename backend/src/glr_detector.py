#!/usr/bin/env python3
"""
SpaceShield: Generalized Likelihood Ratio Test (GLRT) Anomaly Detector.
Author: Antigravity AI
Version: 2.0.0

This module implements a mathematically rigorous Generalized Likelihood Ratio Test (GLRT)
to detect satellite tracking-loop anomalies (such as Doppler rate drag-off spoofing
or multi-generator spoofing) in NavIC GSO/GEO ground station operations.
"""

import numpy as np
from scipy import stats

class GLRDetector:
    def __init__(self, window_size=50, nominal_variance=0.05, p_fa=1e-7, dimension=1):
        """
        Initializes the GLR Detector.
        
        Parameters:
          window_size (int): Size of the sliding epoch window (N).
          nominal_variance (float): Expected tracking noise variance (sigma^2) under H0.
          p_fa (float): Target probability of false alarm (P_fa).
          dimension (int): Dimension of the tracking parameter vector (D). Default is 1 (scalar).
        """
        self.N = window_size
        self.sigma_sq = nominal_variance
        self.p_fa = p_fa
        self.D = dimension
        
        # Degrees of freedom for Chi-Squared distribution under H0:
        # Since we sum the squared errors of N vectors of dimension D,
        # the total degrees of freedom is N * D.
        self.df = self.N * self.D
        
        # Calculate the decision threshold gamma using the Chi-Squared PPF (inverse CDF)
        # H0: Lambda_k ~ Chi-Squared(N * D)
        # P(Lambda_k > gamma | H0) = P_fa => gamma = PPF(1 - P_fa)
        self.gamma = stats.chi2.ppf(1.0 - self.p_fa, df=self.df)

    def update_parameters(self, window_size=None, nominal_variance=None, p_fa=None, dimension=None):
        """Allows dynamic updates to the detector parameters and recalculates threshold."""
        if window_size is not None:
            self.N = window_size
        if nominal_variance is not None:
            self.sigma_sq = nominal_variance
        if p_fa is not None:
            self.p_fa = p_fa
        if dimension is not None:
            self.D = dimension
            
        self.df = self.N * self.D
        self.gamma = stats.chi2.ppf(1.0 - self.p_fa, df=self.df)
        print(f"[*] Detector calibrated: df={self.df}, threshold={self.gamma:.4f}")

    def compute_glrt(self, observations, predictions):
        """
        Computes the GLRT statistic over the sliding window.
        
        Mathematical Formulation:
          Under H0 (Nominal): Residual vector e_i = y_i - x_i follows N(0, sigma^2 * I_D)
          Under H1 (Spoofed): Residual vector e_i follows N(mu_i, sigma_1^2 * I_D)
          
          The Log-Likelihood Ratio (LLR) test statistic accumulator Lambda_k is:
            Lambda_k = (1 / sigma^2) * sum_{i=k-N+1}^{k} (y_i - x_i)^T * (y_i - x_i)
            
          Explicitly utilizing linear algebra:
            Let e_i = y_i - x_i be a column vector of dimension (D x 1).
            e_i^T * e_i represents the inner product (Euclidean norm squared).
            
            Lambda_k = (1 / sigma^2) * sum_{i=k-N+1}^{k} ||e_i||_2^2
            
          Under H0: Lambda_k follows a central Chi-Squared distribution with N*D degrees of freedom.
          If Lambda_k > gamma, we reject H0 and trigger an anomaly alert.
          
        Parameters:
          observations (np.ndarray): Array of shape (N, D) representing received measurements.
          predictions (np.ndarray): Array of shape (N, D) representing predicted nominal paths.
          
        Returns:
          dict: Output containing statistic, threshold, and alert verdict.
        """
        obs = np.asarray(observations, dtype=np.float64)
        pred = np.asarray(predictions, dtype=np.float64)
        
        # Ensure correct dimensionality
        if obs.ndim == 1:
            obs = obs.reshape(-1, 1)
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
            
        if len(obs) < self.N:
            raise ValueError(f"Observations length ({len(obs)}) is smaller than window size N={self.N}")
            
        # Select the active sliding window window
        window_obs = obs[-self.N:]
        window_pred = pred[-self.N:]
        
        # Calculate residuals: e_i = y_i - x_i
        residuals = window_obs - window_pred
        
        # Calculate statistic accumulator via linear algebra inner products
        # lambda_k = (1 / sigma^2) * sum( dot_product(e_i, e_i) )
        lambda_k = 0.0
        for i in range(self.N):
            e_i = residuals[i]
            # Vector inner product: e_i^T * e_i
            inner_prod = np.dot(e_i, e_i)
            lambda_k += inner_prod
            
        lambda_k = lambda_k / self.sigma_sq
        
        alert_triggered = bool(lambda_k > self.gamma)
        
        return {
            "test_statistic": float(lambda_k),
            "threshold": float(self.gamma),
            "alert_triggered": alert_triggered,
            "degrees_of_freedom": self.df,
            "p_value": float(1.0 - stats.chi2.cdf(lambda_k, df=self.df))
        }

class TrackingLoopObserver:
    """
    Manages the sliding window buffers of measurements and nominal estimates,
    providing continuous feed updates into the GLR detector.
    """
    def __init__(self, detector):
        self.detector = detector
        self.obs_buffer = []
        self.pred_buffer = []

    def feed(self, measured_vector, nominal_vector):
        """
        Feeds a new observation epoch and returns the GLRT results.
        """
        # Ensure inputs are array-like
        m_vec = np.atleast_1d(measured_vector)
        n_vec = np.atleast_1d(nominal_vector)
        
        self.obs_buffer.append(m_vec)
        self.pred_buffer.append(n_vec)
        
        if len(self.obs_buffer) > self.detector.N:
            self.obs_buffer.pop(0)
            self.pred_buffer.pop(0)
            
        if len(self.obs_buffer) < self.detector.N:
            return {
                "test_statistic": 0.0,
                "threshold": self.detector.gamma,
                "alert_triggered": False,
                "buffer_status": "WARMING_UP"
            }
            
        result = self.detector.compute_glrt(self.obs_buffer, self.pred_buffer)
        result["buffer_status"] = "READY"
        return result

if __name__ == "__main__":
    # Diagnostic check: scalar (D=1) and vector (D=2) tests
    print("=" * 70)
    print("          GLRT ANOMALY DETECTOR MATH ENGINE DIAGNOSTICS          ")
    print("=" * 70)
    
    # Configure detector for D=2 tracking (Doppler offset + Code phase offset)
    detector = GLRDetector(window_size=50, nominal_variance=0.05, p_fa=1e-7, dimension=2)
    observer = TrackingLoopObserver(detector)
    
    print(f"[*] Configuration:")
    print(f"    - Sliding Window Size (N): {detector.N} epochs")
    print(f"    - Dimension (D):            {detector.D} (Doppler & Code Phase)")
    print(f"    - Total Degrees of Freedom: {detector.df}")
    print(f"    - Nominal Variance (sigma): {detector.sigma_sq}")
    print(f"    - Target False Alarm Rate:  {detector.p_fa}")
    print(f"    - Decision Threshold:       {detector.gamma:.4f}")
    print("-" * 70)

    # 1. Nominal test
    print("[*] Simulating Scenario 1: Nominal Tracking (H0)...")
    # Residuals follow N(0, 0.05 * I_2)
    nominal_errors = np.random.normal(0, np.sqrt(detector.sigma_sq), (100, 2))
    
    h0_alerts = 0
    for err in nominal_errors:
        res = observer.feed(err, np.zeros(2))
        if res["alert_triggered"]:
            h0_alerts += 1
            
    print(f"    - Final Test Statistic: {res['test_statistic']:.4f}")
    print(f"    - Alert Triggered?      {res['alert_triggered']}")
    print(f"    - Total Alerts:         {h0_alerts} / 100 epochs")
    print("-" * 70)

    # 2. Dynamic drift test
    print("[*] Simulating Scenario 2: Multi-Generator Dynamic Drag-off (H1)...")
    observer = TrackingLoopObserver(detector)
    h1_alerts = 0
    breach_epoch = -1
    
    for epoch in range(100):
        if epoch < 40:
            # Nominal noise
            err = np.random.normal(0, np.sqrt(detector.sigma_sq), 2)
        else:
            # Linear drift on Doppler channel (index 0) and Code phase channel (index 1)
            drift_doppler = (epoch - 40) * 0.05
            drift_code = (epoch - 40) * 0.02
            err = np.random.normal([drift_doppler, drift_code], np.sqrt(detector.sigma_sq))
            
        res = observer.feed(err, np.zeros(2))
        if res["alert_triggered"]:
            h1_alerts += 1
            if breach_epoch == -1:
                breach_epoch = epoch
                
    print(f"    - Final Test Statistic: {res['test_statistic']:.4f}")
    print(f"    - Alert Triggered?      {res['alert_triggered']}")
    print(f"    - First Alert Epoch:    {breach_epoch} (Drift initiated at epoch 40)")
    print(f"    - Total Alerts:         {h1_alerts} / 100 epochs")
    print("=" * 70)
