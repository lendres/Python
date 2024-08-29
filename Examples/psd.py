"""
Created on March 23, 2024
@author: Lance Endres
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal

np.random.seed(0)

time_step = .01
time_vec = np.arange(0, 70, time_step)

# A signal with a small frequency chirp
vals = 0.5 * np.pi * time_vec * (1 + .1 * time_vec)
print("Values Type:", type(vals))
print("Values Length:", len(vals))
print("Values Shape:", vals.shape)

sig = np.sin(vals)

print("Signal Type:", type(sig))
print("Signal Length:", len(sig))
print("Signal Shape:", sig.shape)

plt.figure(figsize=(8, 5))
plt.plot(time_vec, sig)



freqs, times, spectrogram = signal.spectrogram(sig, fs=1.0/time_step)

plt.figure(figsize=(5, 4))
plt.imshow(spectrogram, aspect='auto', cmap='hot_r', origin='lower')
plt.title('Spectrogram')
plt.ylabel('Frequency band')
plt.xlabel('Time window')
plt.tight_layout()



freqs, psd = signal.welch(sig)

plt.figure(figsize=(5, 4))
plt.semilogx(freqs, psd)
plt.title('PSD: power spectral density')
plt.xlabel('Frequency')
plt.ylabel('Power')
plt.tight_layout()