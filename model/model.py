import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from keras import layers
from keras.models import Sequential

import os
import pathlib

# Path to the local image directory
data_dir = os.path.join(os.getcwd(), "data", "images")
# image_count = len(list(data_dir.glob('*\\*.png')))
# print(image_count)

'''
The batch size is the number of samples
that are processed before the model is updated.
'''
batch_size = 64 
img_width = 128
img_height = 128

# Create a dataset and reserve a fraction of it for model evaluation
# - reference: https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image_dataset_from_directory
training_set, validation_set = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels = "inferred",
    class_names = [
        "circles",
        "rectangles",
        "triangles"
    ],
    batch_size=batch_size,
    image_size=(img_height, img_width),
    shuffle=False,
    validation_split = 0.65,
    subset = "both"
)

# Create a simple sequential mode
model = Sequential([
    keras.Input(shape=(img_height, img_width, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(3, activation='softmax')  # Assuming 3 shape classes (circle, triangle, square)
])

#Compile the model (specify the training configuration)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#Print the model summary
model.summary()

#Train the model
history = model.fit(
    training_set,
    batch_size=batch_size,
    epochs = 6,
    #Pass in the validation set to monitor validation loss after each epoch
    validation_data=validation_set
)
print(history.history)

model_path = os.path.join(os.getcwd(), "shape_classifier_model.keras")
model.save(model_path)

#Print the model summary
model.summary()
