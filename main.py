import os
import dataset
from phraselabel import DeterministicLabeler

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow.keras as keras
from tensorflow.keras.layers import *
from tensorflow.keras import losses, metrics, optimizers as optim
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

num_classes = len(dataset.train.class_names)

labeler = DeterministicLabeler()
model = keras.Sequential([
    Conv2D(32, (3, 3), activation='relu'),
    Conv2D(64, (3, 3), activation='relu'),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])
model._name = labeler(model)
