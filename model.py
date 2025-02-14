from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Flatten, Input


def build_model(input_shape):
    """Build a neural network model using updated Keras best practices."""
    model = Sequential()

    # Define Input layer to avoid deprecation warnings
    model.add(Input(shape=input_shape))

    # Flatten the input
    model.add(Flatten())

    # First dense layer with ReLU activation
    model.add(Dense(100, activation='relu'))

    # Add dropout to prevent overfitting
    model.add(Dropout(0.5))

    # Output layer for binary classification
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model using Adam optimizer and binary cross-entropy loss
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model
