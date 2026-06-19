import numpy as np
from scipy.optimize import minimize
import sys

np.random.seed(0)
N = 4096 * 4
t = np.arange(N)
T = np.exp(1j * 0.08 * t) * 0.04 + np.exp(1j * 0.12 * t) * 0.03
J = np.exp(1j * 0.6 * t) * 0.5
X = T + J

u30_true = X * np.abs(X)**2
u31_true = np.roll(X, 1) * np.abs(np.roll(X, 1))**2
u50_true = X * np.abs(X)**4

X_lna = X - 0.8 * u30_true - 0.2 * u31_true + 0.05 * u50_true

u30 = X_lna * np.abs(X_lna)**2
u31 = np.roll(X_lna, 1) * np.abs(np.roll(X_lna, 1))**2
u50 = X_lna * np.abs(X_lna)**4

def bandpass_filter(signal: np.ndarray, f_low: float, f_high: float) -> np.ndarray:
    freqs = np.fft.fftfreq(len(signal))
    omega = np.abs(freqs) * 2.0 * np.pi
    mask = (omega >= f_low) & (omega <= f_high)
    sig_fft = np.fft.fft(signal)
    sig_fft[~mask] = 0.0
    return np.fft.ifft(sig_fft)

X_target_dist = bandpass_filter(X_lna, 0.0, 0.25)
alpha_dist = np.vdot(T, X_target_dist) / np.vdot(T, T)
dist_error = np.mean(np.abs(X_target_dist - alpha_dist * T)**2)

def objective(c):
    c10 = c[0] + 1j*c[1]
    c30 = c[2] + 1j*c[3]
    c31 = c[4] + 1j*c[5]
    c50 = c[6] + 1j*c[7]
    Y = c10 * X_lna + c30 * u30 + c31 * u31 + c50 * u50
    Y_target_lin = bandpass_filter(Y, 0.0, 0.25)
    alpha_lin = np.vdot(T, Y_target_lin) / np.vdot(T, T)
    lin_error = np.mean(np.abs(Y_target_lin - alpha_lin * T)**2)
    return lin_error

res = minimize(objective, [1, 0, 0.8, 0, 0.2, 0, -0.05, 0], method='BFGS')
print(res.x)
lin_error = res.fun
imd_suppression = 10.0 * np.log10(dist_error / (lin_error + 1e-12))
print("Best Lin Error:", lin_error)
print("Best IMD Supp:", imd_suppression)
