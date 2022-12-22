import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

# Set the path to the dataset
DATASET_PATH = "dataset/dogs_vs_cats"

# Set the image size and number of classes
IMG_SIZE = 150
NUM_CLASSES = 2

# Set the batch size and epochs
BATCH_SIZE = 32
EPOCHS = 10

# Function to preprocess the images
def preprocess_image(image):
  image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
  image = image / 255.0
  return image

# Function to load and preprocess the data
def load_data():
  # Initialize the list of images and labels
  images = []
  labels = []

  # Loop through the dataset directory
  for label in os.listdir(DATASET_PATH):
    # Loop through the images in each subdirectory
    for image in os.listdir(os.path.join(DATASET_PATH, label)):
      # Load the image and preprocess it
      img = cv2.imread(os.path.join(DATASET_PATH, label, image))
      img = preprocess_image(img)

      # Add the image and label to the lists
      images.append(img)
      labels.append(int(label == "cats"))

  # Convert the lists to NumPy arrays
  images = np.array(images)
  labels = np.array(labels)

  # Shuffle the data
  indices = np.arange(len(images))
  np.random.shuffle(indices)
  images = images[indices]
  labels = labels[indices]

  return images, labels

# Load and preprocess the data
images, labels = load_data()

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2)

# Create the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(128, (3, 3), activation="relu"))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(NUM_CLASSES, activation="sigmoid"))

# Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("Test loss:", loss)
print("Test accuracy:", accuracy)

# Plot the training and validation accuracy
plt.plot(history.history["accuracy"], label="accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.legend()
plt.show()

# Plot the training and validation loss
plt.plot(history.history["loss"], label="loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.legend()
plt.show()

