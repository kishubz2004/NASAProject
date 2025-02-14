import glob
import obspy
import pandas as pd
import os

def load_mseed_data(mseed_file_path):
    """Load seismic data from a MiniSEED file."""
    st = obspy.read(mseed_file_path)
    tr = st[0]  # Taking the first trace
    return tr.data, tr.stats

def load_csv_data(csv_file_path):
    """Load seismic data from a CSV file."""
    data = pd.read_csv(csv_file_path)
    time = data['time'].values
    amplitude = data['amplitude'].values
    return time, amplitude

def load_catalog(catalog_file_path):
    """Load quake catalog from CSV file."""
    catalog = pd.read_csv(catalog_file_path)
    print(f"Columns in catalog: {catalog.columns}")
    return catalog

def list_files_in_directory(directory_path, extension=".mseed"):
    search_pattern = os.path.join(directory_path, f"**/*{extension}")
    files = glob.glob(search_pattern, recursive=True)
    print(f"Files found in {directory_path}: {files}")
    return files

