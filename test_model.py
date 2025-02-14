import os
import pandas as pd
from data_loading import load_mseed_data, list_files_in_directory
from preprocessing import highpass_filter, wavelet_denoising
from keras.src.utils import pad_sequences


def test_model_on_data(test_data_path, model, output_file):
    """Test the model on new data and output predictions, save them to a catalog file."""

    # Load the test data
    mseed_files = list_files_in_directory(test_data_path)

    # If no files are found, skip the directory
    if len(mseed_files) == 0:
        print(f"No test data found in {test_data_path}. Skipping testing.")
        return

    all_test_data = []
    catalog_data = []

    for mseed_file in mseed_files:
        seismic_data, stats = load_mseed_data(mseed_file)  # Load MiniSEED file
        filtered_data = highpass_filter(seismic_data, cutoff=0.1, fs=50)
        denoised_data = wavelet_denoising(filtered_data)
        all_test_data.append(denoised_data)

        # Extract time information for the catalog
        start_time = stats.starttime  # Assuming stats contain start time (from obspy)
        time_abs = start_time.strftime('%Y-%m-%dT%H:%M:%S.%f')

        # Append catalog entry
        catalog_data.append({'filename': os.path.basename(mseed_file), 'time_abs': time_abs})

    # Pad or truncate sequences to match the training data shape
    X_test = pad_sequences(all_test_data, maxlen=72000)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)  # Add channel dimension

    # Check if there is data to predict on
    if X_test.shape[0] == 0:
        print(f"No valid data to predict in {test_data_path}. Skipping predictions.")
        return

    # Make predictions
    test_predictions = (model.predict(X_test) > 0.5).astype(int)

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save catalog data with predictions
    catalog_df = pd.DataFrame(catalog_data)
    catalog_df['predictions'] = test_predictions.flatten()  # Add predictions to the catalog
    catalog_df.to_csv(output_file, index=False)

    print(f"Catalog file with predictions saved at: {output_file}")
