import numpy as np
from scipy.signal import welch
from scipy.io.wavfile import write as wavwrite
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# --- Core Utility Functions ---

def whiten(strain, interp_psd, dt):
    """
    Whiten strain data.
    
    Parameters:
    strain (np.ndarray): Strain time series
    interp_psd (scipy.interpolate.interp1d): Interpolated PSD
    dt (float): Time step
    
    Returns:
    np.ndarray: Whitened strain data
    """
    Nt = len(strain)
    freqs = np.fft.rfftfreq(Nt, dt)
    
    # whitening: transform to frequency domain, divide by ASD, then transform back, 
    # taking care to get normalization right.
    hf = np.fft.rfft(strain)
    # real(interp_psd) is needed to protect against tiny imaginary parts 
    # introduced by interpolation.
    norm = 1./np.sqrt(1./(dt*2))
    white_hf = hf / (np.sqrt(np.real(interp_psd(freqs))) * norm)
    white_ht = np.fft.irfft(white_hf, n=Nt)
    return white_ht

def write_wavfile(filename,fs,data):
    """
    Write strain data to a WAV file.
    
    Parameters:
    filename (str): Name of the output WAV file
    fs (int): Sample rate
    data (np.ndarray): Strain time series
    """
    d = np.int16(data/np.max(np.abs(data)) * 32767 * 0.9)
    wavwrite(filename,int(fs),d)

def reqshift(data,fshift=100,sample_rate=4096):
    """
    Frequency shift data by fshift Hz.
    
    Parameters:
    data (np.ndarray): Strain time series
    fshift (float): Frequency shift in Hz
    sample_rate (int): Sample rate
    
    Returns:
    np.ndarray: Frequency-shifted data
    """
    x = np.fft.rfft(data)
    T = len(data)/sample_rate
    df = 1.0/T
    nbins = int(fshift/df)
    # print T,df,nbins,len(x)
    # fshift < 0 => move signals to higher frequencies
    if fshift > 0:
        x[nbins:] = x[:-nbins]
        x[:nbins] = 0.
    # fshift > 0 => move signals to lower frequencies
    else:
        x[:-nbins] = x[nbins:]
        x[-nbins:] = 0.
    return np.fft.irfft(x)

# --- Plotting Utility Functions ---

def plot_asd(freqs, Pxx_H1, Pxx_L1, eventname, plottype):
    """
    Plot the Amplitude Spectral Density (ASD) for H1 and L1.
    Saves the figure to the 'figures/' directory.
    
    Parameters:
    freqs (np.ndarray): Frequency array
    Pxx_H1 (np.ndarray): Power Spectral Density for H1
    Pxx_L1 (np.ndarray): Power Spectral Density for L1
    eventname (str): Name of the event (e.g., 'GW150914')
    plottype (str): File extension for the plot (e.g., 'png')
    """
    plt.figure(figsize=(10,8))
    plt.suptitle('Power Spectral Density', fontsize=16)
    
    plt.subplot(2,1,1)
    plt.loglog(freqs, np.sqrt(Pxx_H1), 'b', label='H1 strain')
    plt.loglog(freqs, np.sqrt(Pxx_L1), 'r', label='L1 strain')
    plt.axis([20, 2048, 1e-24, 1e-21])
    plt.ylabel('ASD (strain/rtHz)', fontsize=12)
    plt.grid(True)
    plt.legend(loc='upper center')
    
    plt.subplot(2,1,2)
    plt.loglog(freqs, np.sqrt(Pxx_H1), 'b', label='H1 strain')
    plt.loglog(freqs, np.sqrt(Pxx_L1), 'r', label='L1 strain')
    plt.axis([20, 2048, 1e-24, 1e-21])
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('ASD (strain/rtHz)', fontsize=12)
    plt.grid(True)
    plt.legend(loc='upper center')
    
    save_filename = f'figures/{eventname}_ASDs.{plottype}'
    plt.savefig(save_filename)
    print(f"Saved ASD plot to {save_filename}")
    plt.close() 

def plot_strain_asd(freqs, Pxx_L1, Pxx_H1, Pxx, eventname, plottype):
    """
    Plot the ASDs of L1, H1, and a smooth model.
    Saves the figure to 'figures/'.
    
    Parameters:
    freqs (np.ndarray): Frequency array
    Pxx_L1 (np.ndarray): Power Spectral Density for L1
    Pxx_H1 (np.ndarray): Power Spectral Density for H1
    Pxx (np.ndarray): Power Spectral Density for the smooth model
    eventname (str): Name of the event (e.g., 'GW150914')
    plottype (str): File extension for the plot (e.g., 'png')
    """
    f_min = 20.
    f_max = 2000.
    
    plt.figure(figsize=(10,8))
    plt.loglog(freqs, np.sqrt(Pxx_L1),'g',label='L1 strain')
    plt.loglog(freqs, np.sqrt(Pxx_H1),'r',label='H1 strain')
    plt.loglog(freqs, np.sqrt(Pxx),'k',label='H1 strain, O1 smooth model')
    plt.axis([f_min, f_max, 1e-24, 1e-19])
    plt.grid('on')
    plt.ylabel('ASD (strain/rtHz)')
    plt.xlabel('Freq (Hz)')
    plt.legend(loc='upper center')
    plt.title('Advanced LIGO strain data near '+eventname)
    
    save_filename = f'figures/{eventname}_strain.{plottype}'
    plt.savefig(save_filename)
    print(f"Saved strain ASD plot to {save_filename}")
    plt.close()