
#model.py
# %%

# Import necessary libraries
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from itertools import product
from sklearn import metrics
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Set the path to the data directory
PATH = os.path.join('data')

# Create an array of actions (signs) labels by listing the contents of the data directory
actions = np.array(os.listdir(PATH))

# Define the number of sequences and frames
sequences = 30
frames = 10

# Create a label map to map each action label to a numeric value
label_map = {label:num for num, label in enumerate(actions)}

# Initialize empty lists to store landmarks and labels
landmarks, labels = [], []

# Iterate over actions and sequences to load landmarks and corresponding labels
for action, sequence in product(actions, range(sequences)):
    temp = []
    for frame in range(frames):
        npy = np.load(os.path.join(PATH, action, str(sequence), str(frame) + '.npy'))
        temp.append(npy)
    landmarks.append(temp)
    labels.append(label_map[action])

# Convert landmarks and labels to numpy arrays
X, Y = np.array(landmarks), to_categorical(labels).astype(int)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=34, stratify=Y)

# Define the model architecture
# model = Sequential()
# model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(10,126)))  # Increased from 32
# model.add(LSTM(128, return_sequences=True, activation='relu'))  # Increased from 64
# model.add(LSTM(64, return_sequences=False, activation='relu'))  # Increased from 32
# model.add(Dense(64, activation='relu'))  # Increased from 32
# model.add(Dense(actions.shape[0], activation='softmax'))
# Define a more effective model architecture
model = Sequential()

# Input shape is now (10, 266) - additional orientation vectors for each hand
model.add(LSTM(96, return_sequences=True, activation='relu', input_shape=(10, 266)))
model.add(BatchNormalization())
model.add(Dropout(0.25))

# Add bidirectional LSTM layers for better temporal pattern recognition
model.add(Bidirectional(LSTM(128, return_sequences=True, activation='relu')))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Bidirectional(LSTM(64, return_sequences=False, activation='relu')))
model.add(BatchNormalization())
model.add(Dropout(0.25))

# Deeper classification layers
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(64, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

# Compile with a lower learning rate for better convergence
optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Add more sophisticated callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.00001)

# Train with proper validation
model.fit(
    X_train, Y_train, 
    epochs=250,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping, model_checkpoint, reduce_lr]
)


# Save the trained model
model.save('my_model.keras')

# Make predictions on the test set
predictions = np.argmax(model.predict(X_test), axis=1)
# Get the true labels from the test set
test_labels = np.argmax(Y_test, axis=1)

# Calculate the accuracy of the predictions
accuracy = metrics.accuracy_score(test_labels, predictions)


