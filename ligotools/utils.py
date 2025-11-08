# utils.py

import numpy as np
from scipy.signal import butter, filtfilt
from scipy.signal import tukey
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

# -----------------------------------------------------------
# Whiten function
# -----------------------------------------------------------
def whiten(strain, interp_psd, dt):
    """
    Whiten a strain time series by dividing by the amplitude spectral density (ASD).
    """
    Nt = len(strain)
    freqs = np.fft.rfftfreq(Nt, dt)
    hf = np.fft.rfft(strain)
    white_hf = hf / (np.sqrt(interp_psd(freqs) / (dt / 2.0)))
    white_ht = np.fft.irfft(white_hf, n=Nt)
    return white_ht

# -----------------------------------------------------------
# Write WAV file function
# -----------------------------------------------------------
def write_wavfile(filename, fs, data):
    """
    Write audio data to a WAV file.
    """
    scaled = np.int16(data / np.max(np.abs(data)) * 32767)
    write(filename, fs, scaled)

# -----------------------------------------------------------
# Frequency shift function
# -----------------------------------------------------------
def reqshift(data, fshift=100, sample_rate=4096):
    """
    Shift frequency content of a signal by fshift Hz.
    """
    x = np.fft.rfft(data)
    N = len(data)
    freqs = np.fft.rfftfreq(N, 1.0 / sample_rate)
    phase = np.exp(2j * np.pi * fshift * np.arange(len(freqs)) / sample_rate)
    x_shifted = x * phase
    shifted = np.fft.irfft(x_shifted, n=N)
    return shifted

# -----------------------------------------------------------
# PSD Plot Function
# -----------------------------------------------------------
def plot_psd(strain_H1, strain_L1, fs, NFFT=4*4096, noverlap=2*4096, window='hann'):
    """
    Plot Power Spectral Density (PSD) for both detectors to minimize spectral leakage.
    """
    plt.figure(figsize=(10,6))
    Pxx_H1, freqs_H1 = mlab.psd(strain_H1, Fs=fs, NFFT=NFFT, noverlap=noverlap, window=mlab.window_hanning)
    Pxx_L1, freqs_L1 = mlab.psd(strain_L1, Fs=fs, NFFT=NFFT, noverlap=noverlap, window=mlab.window_hanning)
    
    plt.loglog(freqs_H1, np.sqrt(Pxx_H1), 'r', label='H1')
    plt.loglog(freqs_L1, np.sqrt(Pxx_L1), 'g', label='L1')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('ASD (strain/√Hz)')
    plt.legend(loc='best')
    plt.title('Power Spectral Density (PSD)')
    plt.grid(True)
    plt.show()
