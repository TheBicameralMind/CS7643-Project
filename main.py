import os
import shutil

import dataset
from phraselabel import DeterministicLabeler

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import losses, metrics
from tensorflow.keras import optimizers as optim
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)
from tensorflow.keras.layers import *

num_classes = len(dataset.class_names)

labeler = DeterministicLabeler()

model = keras.Sequential(
    [
        Conv2D(
            16,
            (5, 5),
            activation="relu",
            padding="same",
            input_shape=(*dataset.image_size, 3),
        ),
        Conv2D(32, (4, 4), activation="relu", padding="same"),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(32, (3, 3), activation="relu", padding="same"),
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation="relu"),
        Conv2D(32, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(64, activation="relu"),
        Dropout(0.25),
        Dense(num_classes, activation="softmax"),
    ]
)
model._name = labeler(model)
print(model.name)

shutil.rmtree(f"tensorboard/{model.name}", ignore_errors=True)
shutil.rmtree(f"checkpoints/{model.name}", ignore_errors=True)

callbacks = [
    TensorBoard(log_dir=f"tensorboard/{model.name}"),
    ModelCheckpoint(filepath=f"checkpoints/{model.name}", save_best_only=False),
    EarlyStopping(patience=8, restore_best_weights=True),
    ReduceLROnPlateau(patience=4, factor=0.1),
]

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics="accuracy")

batch = 64
train = dataset.train(batch)
val = dataset.val(batch)
test = dataset.test(batch)

model.fit(
    train,
    batch_size=batch,
    epochs=38,
    validation_data=val,
    callbacks=callbacks,
    class_weight=dataset.get_class_weights(),
)

model.evaluate(test, batch_size=batch)

model.save(f"models/{model.name}")
