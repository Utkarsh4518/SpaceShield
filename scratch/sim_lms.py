import numpy as np

np.random.seed(0)
N = 4096 * 50
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

h = np.array([ 1., -4.640679, 8.9255275, -8.9255275, 4.640679, -1.], dtype=np.float32)

t30 = np.convolve(u30, h)[:N]
t31 = np.convolve(u31, h)[:N]
t50 = np.convolve(u50, h)[:N]

elna = np.convolve(X_lna, h)[:N]

c30 = 0.0 + 0j
c31 = 0.0 + 0j
c50 = 0.0 + 0j

mu = 0.05

for i in range(5, N, 32):
    e = elna[i] + c30 * t30[i] + c31 * t31[i] + c50 * t50[i]
    norm = np.abs(t30[i])**2 + np.abs(t31[i])**2 + np.abs(t50[i])**2 + 1e-6
    factor = mu / norm
    
    c30 -= factor * e * np.conj(t30[i])
    c31 -= factor * e * np.conj(t31[i])
    c50 -= factor * e * np.conj(t50[i])

print("Converged to:", c30, c31, c50)

Y = X_lna + c30 * u30 + c31 * u31 + c50 * u50
err = np.convolve(Y, h)[:N]
print("Final OOB energy:", np.mean(np.abs(err[-4096:])**2))
print("LNA OOB energy:", np.mean(np.abs(elna[-4096:])**2))
