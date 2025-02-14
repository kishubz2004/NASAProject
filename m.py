import pandas as pd
import numpy as np
from obspy import read

# Path to miniseed file (replace with your actual path)
mseed_file = r"D:\Nasa space apps\space_apps_2024_seismic_detection\space_apps_2024_seismic_detection\data\mars\training\data\*.mseed"

# Read the seismic data using obspy
st = read(mseed_file)
tr = st[0]  # Assume the first trace

# Extract time and velocity (ground motion) data
time = tr.times('utcdatetime')  # Get absolute times (UTC)
velocity = tr.data  # Get velocity or seismic data

# Create a DataFrame
data = pd.DataFrame({
    'time_abs': time,
    'velocity': velocity
})

# Save to Excel file
output_file = r"D:\Nasa space apps\surce code\mars_velocity_data.xlsx"
data.to_excel(output_file, index=False)

print(f"Data saved to {output_file}")
