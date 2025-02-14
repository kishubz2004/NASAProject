from data_loading import load_mseed_data, load_catalog, list_files_in_directory
from preprocessing import highpass_filter, wavelet_denoising
from model import build_model
from test_model import test_model_on_data
from seismic_noise_tomography import seismic_noise_tomography
import numpy as np
import os
import joblib  # For saving the model
from sklearn.preprocessing import LabelEncoder


def mars_workflow(training_data_path, catalog_path, test_data_path, output_path):
    """Main workflow for handling Mars seismic data."""

    # Load and preprocess Mars data for training
    mseed_files = list_files_in_directory(training_data_path)
    if not mseed_files:
        print(f"No training data found in {training_data_path}. Skipping Mars workflow.")
        return  # Skip the rest of the function if no files are found

    print(f"Files found in {training_data_path}: {mseed_files}")

    # Load the quake catalog
    catalog = load_catalog(catalog_path)

    # Prepare training data
    all_denoised_data = []
    for mseed_file in mseed_files:
        seismic_data, _ = load_mseed_data(mseed_file)
        filtered_data = highpass_filter(seismic_data, cutoff=0.1, fs=50)
        denoised_data = wavelet_denoising(filtered_data)
        all_denoised_data.append(denoised_data)

    # Convert to NumPy array and reshape for training
    X_train = np.array(all_denoised_data).reshape(-1, 72000, 1)

    # Encode labels to numeric format using LabelEncoder
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(catalog['evid'].values)

    if X_train.size == 0:
        print("No training data to process. Skipping Mars workflow.")
        return

    # Build and train the neural network model
    model = build_model(input_shape=(X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.2)

    # Test the model on Mars test set
    test_model_on_data(test_data_path, model, os.path.join(output_path, "mars_output.csv"))

    # Run Seismic Noise Tomography (if applicable)
    seismic_noise_tomography(r"D:\Nasa space apps\surce code\mars_velocity_data.xlsx")
