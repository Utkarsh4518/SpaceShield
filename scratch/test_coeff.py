import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(".")
BACKEND_SRC = os.path.join(BASE_DIR, 'backend', 'src')
sys.path.insert(0, BACKEND_SRC)

from saturation_inverter import SaturationInverter
from tests.saturation_linearization_verifier import bandpass_filter

num_channels = 1
stride_len = 4096

inverter = SaturationInverter(channels=num_channels, stride_len=stride_len)
coefs = np.zeros((1, 5, 2), dtype=np.complex64)
coefs[0, 0, 0] = 1.0
coefs[0, 2, 0] = 0.8
coefs[0, 2, 1] = 0.2
coefs[0, 4, 0] = -0.05
inverter.coefficients = coefs

t = np.arange(stride_len)
s_target = 0.04 * np.exp(1j * 0.08 * t) + 0.03 * np.exp(1j * 0.12 * t)
s_jammer = 0.5 * np.exp(1j * 0.6 * t)
X_rf = s_target + s_jammer
X_rf = X_rf.reshape(1, -1)
s_target = s_target.reshape(1, -1)

X_lna = np.zeros_like(X_rf)
for ch in range(num_channels):
    for n in range(stride_len):
        val = X_rf[ch, n] - 0.8 * X_rf[ch, n] * (abs(X_rf[ch, n])**2)
        if n > 0:
            val -= 0.2 * X_rf[ch, n - 1] * (abs(X_rf[ch, n - 1])**2)
        val += 0.05 * X_rf[ch, n] * (abs(X_rf[ch, n])**4)
        X_lna[ch, n] = val

Y_linearized = inverter.linearize_stride(X_lna)

X_target_dist = bandpass_filter(X_lna, 0.0, 0.25)
Y_target_lin = bandpass_filter(Y_linearized, 0.0, 0.25)

dist_error = np.mean(np.abs(X_target_dist - s_target)**2)
lin_error = np.mean(np.abs(Y_target_lin - s_target)**2)

imd_suppression = 10.0 * np.log10(dist_error / (lin_error + 1e-12))
print(f"Dist Error: {dist_error}")
print(f"Lin Error: {lin_error}")
print(f"IMD Suppression: {imd_suppression} dB")
