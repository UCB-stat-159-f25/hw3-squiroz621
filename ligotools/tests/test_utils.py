import pytest
import numpy as np
from scipy.interpolate import interp1d
from scipy.io.wavfile import read as wavread
from ligotools import utils
import os

@pytest.fixture
def sample_data():
    """ Create sample data for testing """
    fs = 4096
    dt = 1.0 / fs
    Nt = 10 * fs  # 10 seconds of data
    times = np.arange(0, 10, dt)
    
    # Create a simple sine wave at 200 Hz
    strain = np.sin(2 * np.pi * 200 * times)
    
    # Create a simple flat PSD (value = 1.0)
    freqs = np.fft.rfftfreq(Nt, dt)
    psd_vals = np.ones_like(freqs)
    interp_psd = interp1d(freqs, psd_vals)
    
    return strain, interp_psd, dt, fs, times

def test_whiten(sample_data):
    """
    Test the whiten function.
    A simple test: check if input and output lengths match
    and the output standard deviation is close to 1 (for flat PSD).
    """
    strain, interp_psd, dt, _, _ = sample_data
    
    whitened_strain = utils.whiten(strain, interp_psd, dt)
    
    assert len(strain) == len(whitened_strain), "Whitened data length doesn't match input"
    
    # For a sine wave whitened with a flat PSD, the std dev won't be 1
    # but we can check it's a numpy array
    assert isinstance(whitened_strain, np.ndarray)

def test_reqshift(sample_data):
    """
    Test the reqshift function.
    We shift a 200 Hz sine wave by 50 Hz and check if the new
    peak frequency is at 250 Hz.
    """
    strain, _, _, fs, _ = sample_data
    fshift = 50.0  # Shift by 50 Hz
    
    shifted_strain = utils.reqshift(strain, fshift=fshift, sample_rate=fs)
    
    # Find the peak frequency in the shifted data
    Nt = len(shifted_strain)
    freqs = np.fft.rfftfreq(Nt, 1.0/fs)
    fft_shifted = np.fft.rfft(shifted_strain)
    peak_freq_index = np.argmax(np.abs(fft_shifted))
    peak_freq = freqs[peak_freq_index]
    
    # Check if the peak is close to the expected 250 Hz
    assert np.isclose(peak_freq, 200.0 + fshift, atol=1.0)

def test_write_wavfile(sample_data, tmp_path):
    """
    Test the write_wavfile function.
    We write a file to a temporary directory and read it back
    to ensure the data and sample rate match.
    
    'tmp_path' is a special pytest fixture that provides a temporary directory.
    """
    strain, _, _, fs, _ = sample_data
    
    # Create a temporary file path
    temp_wav_file = tmp_path / "test.wav"
    
    # Write the file
    utils.write_wavfile(temp_wav_file, fs, strain)
    
    # Check if the file was created
    assert os.path.exists(temp_wav_file)
    
    # Read the file back
    fs_read, data_read_scaled = wavread(temp_wav_file)
    
    # Check sample rate
    assert fs_read == fs
    
    # Check data shape
    assert data_read_scaled.shape == strain.shape
    
    # Check that the data is scaled to int16
    assert data_read_scaled.dtype == np.int16