from keras.src.utils import pad_sequences
from sklearn.preprocessing import LabelEncoder
from data_loading import load_mseed_data, load_catalog, list_files_in_directory
from preprocessing import highpass_filter, wavelet_denoising
from model import build_model
from test_model import test_model_on_data
from seismic_noise_tomography import seismic_noise_tomography
import numpy as np
import os
import joblib  # For saving the model


def lunar_workflow(training_data_path, catalog_path, test_data_paths, output_path):
    """Main workflow for handling lunar seismic data."""

    print("Loading and preprocessing lunar training data...")
    mseed_files = list_files_in_directory(training_data_path)
    catalog = load_catalog(catalog_path)

    all_denoised_data = []
    for mseed_file in mseed_files:
        print(f"Processing {os.path.basename(mseed_file)}")
        seismic_data, _ = load_mseed_data(mseed_file)
        print(f"Loaded seismic data from {mseed_file}, data size: {seismic_data.size}")
        filtered_data = highpass_filter(seismic_data, cutoff=0.1, fs=50)
        denoised_data = wavelet_denoising(filtered_data)
        all_denoised_data.append(denoised_data)

    if len(all_denoised_data) == 0:
        print("No data was processed. Check your input files.")
        return

    X_train = pad_sequences(all_denoised_data, maxlen=72000)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

    print(f"Number of training samples: {X_train.shape[0]}")
    if X_train.shape[0] == 0:
        raise ValueError("No training samples found.")

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(catalog['mq_type'].values)

    print("Building and training the lunar model...")
    model = build_model(input_shape=(72000, 1))

    model.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.05)  # Reduced validation split

    # Save the model and label encoder
    model.save(os.path.join(output_path, 'lunar_model.h5'))
    joblib.dump(label_encoder, os.path.join(output_path, 'label_encoder.pkl'))

    print("Lunar model saved.")

    # Test the model on lunar test sets
    for test_path in test_data_paths:
        test_model_on_data(test_path, model, os.path.join(output_path, "lunar_output.csv"))

    print("Lunar Workflow Completed.")
