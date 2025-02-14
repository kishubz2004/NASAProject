# Seismic-ANN-Detection

This project uses deep learning techniques to detect seismic events from both lunar and Mars datasets. It processes raw seismic data, removes noise, trains models, and evaluates the performance on test data. The workflow is separated for lunar and Mars data, with each having its own preprocessing steps and model.

## Table of Contents
1. [Project Structure](#project-structure)
2. [Requirements](#requirements)
3. [Setup Instructions](#setup-instructions)
4. [Lunar Workflow](#lunar-workflow)
5. [Mars Workflow](#mars-workflow)
6. [Model Training](#model-training)
7. [Testing](#testing)
8. [Resources](#resources)
9. [Troubleshooting](#troubleshooting)
10. [Acknowledgements](#acknowledgements)

## Project Structure

```
├── data                          # Data files for lunar and Mars seismic events
│   ├── lunar
│   │   ├── training
│   │   ├── test
│   └── mars
│       ├── training
│       ├── test
│
├── results                       # Output directory for results
│   ├── lunar
│   └── mars
│
├── src                           # Source code files
│   ├── main.py                   # Main entry point to run both workflows
│   ├── lunar_workflow.py          # Lunar seismic data workflow
│   ├── mars_workflow.py           # Mars seismic data workflow
│   ├── model.py                  # Contains the build_model function
│   ├── preprocessing.py           # Data preprocessing methods (filtering, wavelet denoising)
│   ├── data_loading.py            # Loading MiniSEED and CSV data
│   ├── seismic_noise_tomography.py# Seismic noise tomography functionality
│   ├── test_model.py              # Function for testing the trained models
│   └── utils.py                  # Utility functions for managing datasets
│
└── README.md                     # This file (documentation)
```

## Requirements

Make sure you have the following dependencies installed:

- Python 3.9+
- TensorFlow 2.x
- ObsPy
- Pandas
- NumPy
- Scikit-learn
- SciPy
- PyWavelets

You can install the necessary packages using:

```bash
pip install -r requirements.txt
```

*Note: `requirements.txt` should contain the package list mentioned above.*

## Setup Instructions

1. **Clone the repository:**

    ```bash
    git clone https://github.com/username/Seismic-ANN-Detection.git
    cd Seismic-ANN-Detection
    ```

2. **Ensure data files are in the correct location:**
    - Place lunar data in `data/lunar/training` and `data/lunar/test`.
    - Place Mars data in `data/mars/training` and `data/mars/test`.

3. **Run the workflow for both lunar and Mars data:**

    ```bash
    python src/main.py
    ```

This will:
- Load, preprocess, and clean the data.
- Train and save the model for both lunar and Mars seismic events.
- Evaluate the model using test data.

## Lunar Workflow

The lunar workflow consists of loading lunar seismic data, applying preprocessing (highpass filtering and wavelet denoising), training a neural network, and testing the model.

### Lunar Training

- The training data is loaded from the path specified in `main.py`.
- Data preprocessing:
    - **Highpass Filtering**: Removes low-frequency noise from the data.
    - **Wavelet Denoising**: Denoises the seismic signals using wavelets.
- After preprocessing, a 1D Convolutional Neural Network (CNN) is trained to detect seismic events.

### Lunar Testing

- After training, the model is tested on lunar test datasets.
- Predictions are saved in `results/lunar/lunar_output.csv`.

## Mars Workflow

The Mars workflow is similar to the lunar workflow, with data sourced from Mars missions.

### Mars Training

- Training data is loaded from the `mars` directory.
- Similar preprocessing steps (highpass filtering, wavelet denoising) are applied to Mars data.
- A neural network model is trained using these preprocessed signals.

### Mars Testing

- Once trained, the Mars model is tested with Mars test data, and predictions are saved in `results/mars/mars_output.csv`.

## Model Training

The `build_model()` function constructs and trains a neural network for seismic event detection. Below is a breakdown of the layers in the model:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Flatten, Conv1D, MaxPooling1D

def build_model(input_shape):
    """Build and return a neural network model."""
    model = Sequential()
    
    # 1D Convolutional layer
    model.add(Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))

    # Flatten for Dense layers
    model.add(Flatten())

    # Fully connected Dense layers
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model
```

### Parameters:
- **Input Shape**: The shape of the preprocessed data (length of the signal).
- **Conv1D Layer**: Extracts spatial features from the seismic signals.
- **Dense Layers**: Fully connected layers to interpret the extracted features.
- **Dropout Layers**: Prevent overfitting during training.

### Training:
```python
model.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.2)
```

- **Epochs**: The number of times the model will be trained on the dataset.
- **Batch size**: How many samples are used in one iteration of the optimization algorithm.
- **Validation Split**: 20% of the training data is used for validation.

## Testing

Once trained, the model is tested on test data. The test data undergoes similar preprocessing and is passed to the trained model for predictions. Results are saved in CSV files.

```python
test_model_on_data(test_data_path, model, output_file)
```

- **Input**: Test data directory.
- **Output**: CSV file with the model's predictions.

## Resources
- **Seismic Noise Tomography**: [GitHub Repository](https://github.com/bgoutorbe/seismic-noise-tomography)
- **NASA Space Apps Challenge**: [Website](https://www.spaceappschallenge.org/)
- **Obspy Documentation**: [Obspy](https://docs.obspy.org/)

## Troubleshooting

### 1. **Zero or One Predictions**
- Ensure the dataset is well-balanced between the classes. If the model always predicts the same class, use class weighting to counteract imbalances:
  ```python
  class_weights = {0: 1.0, 1: 2.0}  # Example for class imbalance
  ```

### 2. **Warnings about AVX2 FMA**
- These warnings are informational and can be ignored unless you're optimizing for performance.

### 3. **Model Performance Issues**
- If the model isn't performing well:
  - Try adjusting the **threshold** for predictions:
    ```python
    test_predictions = (model.predict(X_test) > 0.6).astype(int)
    ```
  - Ensure data preprocessing steps are correctly applied to both training and test data.

### 4. **FileNotFoundError**
- Double-check the data directory structure to ensure files are placed correctly as per the project structure.

## Acknowledgements

- This project was developed for NASA's Space Apps Challenge 2024.
- Special thanks to Vishwas Sai, Supreeth, Kishore, Gagan Sai Deep.

