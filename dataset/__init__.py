import numpy as np
from skimage.exposure import equalize_adapthist
from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_size = (45, 45)
batch_size = 16


def _clahe(img: np.ndarray) -> np.ndarray:
    return equalize_adapthist(img, clip_limit=0.1)


_train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)
_test_datagen = ImageDataGenerator(rescale=1. / 255)

train = _train_datagen.flow_from_directory(
    "dataset/Train",
    subset="training",
    seed=666,
    batch_size=batch_size,
    target_size=image_size,
    class_mode="categorical",
)

val = _train_datagen.flow_from_directory(
    "dataset/Train",
    subset="validation",
    seed=666,
    batch_size=batch_size,
    target_size=image_size,
    class_mode="categorical",
)

test = _test_datagen.flow_from_directory(
    "dataset/Test",
    seed=666,
    batch_size=batch_size,
    target_size=image_size,
    class_mode="categorical",
)
