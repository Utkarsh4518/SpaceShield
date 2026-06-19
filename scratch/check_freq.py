import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(".")
sys.path.insert(0, BASE_DIR)
from tests.saturation_linearization_verifier import bandpass_filter

N = 4096
t = np.arange(N)
T = np.exp(1j * 0.08 * t) * 0.04 + np.exp(1j * 0.12 * t) * 0.03
J = np.exp(1j * 0.6 * t) * 0.5
X_rf = T + J
X_lna = np.zeros_like(X_rf)
for n in range(N):
    val = X_rf[n] - 0.8 * X_rf[n] * (abs(X_rf[n])**2)
    if n > 0:
        val -= 0.2 * X_rf[n - 1] * (abs(X_rf[n - 1])**2)
    val += 0.05 * X_rf[n] * (abs(X_rf[n])**4)
    X_lna[n] = val

X_target_dist = bandpass_filter(X_lna.reshape(1, -1), 0.0, 0.25)[0]
print("Target dist vs T error:", np.mean(np.abs(X_target_dist - T)**2))

alpha = np.vdot(T, X_target_dist) / np.vdot(T, T)
print("Gain compression alpha:", alpha)
print("Target dist vs alpha*T error:", np.mean(np.abs(X_target_dist - alpha * T)**2))

X_imd_dist = bandpass_filter(X_lna.reshape(1, -1), 1.0, 1.2)[0]
print("IMD energy around 1.1 rad:", np.mean(np.abs(X_imd_dist)**2))
