import numpy as np
from scipy.signal import butter, filtfilt
import pywt

def highpass_filter(data, cutoff=0.1, fs=50.0, order=5):
    """Apply a high-pass filter to the seismic data."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data)

def wavelet_denoising(data, wavelet='db4', level=1):
    """Apply wavelet denoising to the seismic data."""
    coeffs = pywt.wavedec(data, wavelet, level=level)
    coeffs[1:] = [np.zeros_like(c) for c in coeffs[1:]]
    return pywt.waverec(coeffs, wavelet)
