import os
import shutil
import dataset
from phraselabel import DeterministicLabeler

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import *
from tensorflow.keras import losses, metrics, optimizers as optim
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

num_classes = len(dataset.class_names)

labeler = DeterministicLabeler()

preprocessing = keras.Sequential([
    Rescaling(scale=1./255),
    RandomFlip('horizontal'),
    RandomRotation(0.2),
    RandomTranslation(0.15, 0.15),
    # RandomZoom(0.1)
])

model = keras.Sequential([
    # preprocessing,

    Conv2D(64, (5, 5), activation='relu', padding='same'),
    Conv2D(128, (4, 4), activation='relu', padding='same'),
    Conv2D(256, (4, 4), activation='relu'),
    BatchNormalization(),

    Conv2D(512, (3, 3), activation='relu'),
    Conv2D(512, (3, 3), activation='relu'),
    Conv2D(512, (3, 3), activation='relu'),
    Flatten(),

    Dense(256, activation='relu'),
    Dense(num_classes, activation='softmax')
])
model._name = labeler(model)
print(model.name)

shutil.rmtree(f'tensorboard/{model.name}', ignore_errors=True)
shutil.rmtree(f'checkpoints/{model.name}', ignore_errors=True)

callbacks = [
    TensorBoard(log_dir=f'tensorboard/{model.name}'),
    ModelCheckpoint(filepath=f'checkpoints/{model.name}')
]

metric = [metrics.Accuracy(), metrics.Precision(), metrics.Recall()]

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=metric
)
model.fit(
    dataset.train(128),
    batch_size=128,
    epochs=50,
    validation_data=dataset.val(128),
    callbacks=callbacks
)
