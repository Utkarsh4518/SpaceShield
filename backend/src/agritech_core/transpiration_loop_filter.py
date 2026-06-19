"""
SpaceShield Sovereign Deep-Tech Edge Processing Engine
Precision Agriculture Architecture
Module: TranspirationLoopFilter
"""

import numpy as np
from numba import njit, float64

class TranspirationLoopFilter:
    def __init__(self, alpha: float = 0.5, beta: float = 0.05, gamma: float = 0.01):
        """
        Initializes the Alpha-Beta-Gamma Kalman Tracking Flywheel for Agritech sensor processing.
        Pre-allocates state matrices to guarantee zero-heap growth during execution.
        """
        self.alpha = np.float64(alpha)
        self.beta = np.float64(beta)
        self.gamma = np.float64(gamma)
        
        # State vector per sensor cluster: [Smoothed Value, Velocity, Acceleration]
        # Tracking 3 parameters: Soil Moisture, VPD, Ambient Humidity
        self.state_matrix = np.zeros((3, 3), dtype=np.float64)
        
    def process_sensor_matrix(self, soil_moisture: float, vapor_pressure_deficit: float, ambient_humidity: float) -> tuple:
        """
        Ingests real-time environmental data and predicts crop transpiration drops.
        """
        measurements = np.array([soil_moisture, vapor_pressure_deficit, ambient_humidity], dtype=np.float64)
        anomalies = np.zeros(3, dtype=np.int8)
        
        TranspirationLoopFilter._kalman_flywheel_step(
            measurements, 
            self.state_matrix, 
            self.alpha, 
            self.beta, 
            self.gamma,
            anomalies
        )
        
        return self.state_matrix[:, 0], anomalies

    @staticmethod
    @njit(fastmath=True, boundscheck=False)
    def _kalman_flywheel_step(measurements, state_matrix, alpha, beta, gamma, anomalies):
        """
        JIT-compiled loop step using native NumPy that smooths away sensor noise variances 
        and isolates real-time data drift anomalies.
        """
        dt = np.float64(1.0) # Normalized unit time step
        
        for i in range(3):
            # Extract current state
            x_est = state_matrix[i, 0]
            v_est = state_matrix[i, 1]
            a_est = state_matrix[i, 2]
            
            # 1. State Extrapolation (Prediction)
            x_pred = x_est + (v_est * dt) + (0.5 * a_est * dt * dt)
            v_pred = v_est + (a_est * dt)
            a_pred = a_est
            
            # 2. Measurement Update
            z = measurements[i]
            residual = z - x_pred
            
            # Anomaly Detection: Flag extreme localized drift spikes
            if abs(residual) > 15.0:
                anomalies[i] = 1
                # Clamp residual to prevent loop divergence during extreme volatility
                residual = 15.0 if residual > 0 else -15.0
                
            # 3. Alpha-Beta-Gamma Corrections
            x_est = x_pred + alpha * residual
            v_est = v_pred + beta * (residual / dt)
            a_est = a_pred + gamma * (residual / (0.5 * dt * dt))
            
            # Store updated state back into pre-allocated memory
            state_matrix[i, 0] = x_est
            state_matrix[i, 1] = v_est
            state_matrix[i, 2] = a_est
