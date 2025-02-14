import pandas as pd
import numpy as np
import scipy.signal as signal

def seismic_noise_tomography(file_path):
    """Perform seismic noise tomography."""
    data = pd.read_excel(file_path)

    print("Columns in the data:", data.columns)

    data['time_abs'] = pd.to_datetime(data['time_abs'])

    time = data['time_abs']
    velocity = data['velocity']

    time_seconds = (time - time.min()).dt.total_seconds()

    fs = 1 / np.mean(np.diff(time_seconds))
    print(f"Calculated sampling frequency (fs): {fs} Hz")

    low_freq = 0.1
    high_freq = fs / 2 * 0.9
    print(f"Low cutoff frequency: {low_freq} Hz, High cutoff frequency: {high_freq} Hz")

    b, a = signal.butter(2, [low_freq, high_freq], 'bandpass', fs=fs)
    filtered_velocity = signal.filtfilt(b, a, velocity)

    normalized_velocity = (filtered_velocity - np.mean(filtered_velocity)) / np.std(filtered_velocity)

    print("Filtered and normalized velocity calculated successfully.")
