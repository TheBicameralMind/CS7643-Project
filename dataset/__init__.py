import os

import numpy as np
from pathlib import Path
from skimage.exposure import equalize_adapthist
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

_data_dir = Path('dataset')
_train = _data_dir / Path('Train')
_test = _data_dir / Path('Test')

image_size = (45, 45)
class_names = np.array(sorted([item.name for item in _test.glob('*')], key=lambda x: int(x)))


# def _get_label(file_path) -> np.ndarray:
#     # Convert the path to a list of path components
#     parts = tf.strings.split(file_path, os.path.sep)
#     # The second to last is the class-directory
#     return parts[-2] == class_names
#
#
# def _decode_img(img):
#     # Convert the compressed string to a 3D uint8 tensor
#     img = tf.io.decode_jpeg(img, channels=3)
#     # Resize the image to the desired size
#     return tf.image.resize(img, image_size)
#
#
# def _process_path(file_path):
#     label = _get_label(file_path)
#     # Load the raw data from the file as a string
#     img = tf.io.read_file(file_path)
#     img = _decode_img(img)
#     return img, label
#
#
# def _configure_for_performance(ds, batch_size):
#     ds = ds.cache()
#     ds = ds.shuffle(buffer_size=1000)
#     ds = ds.batch(batch_size)
#     ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
#     return ds
#
# def _dataset(from_path, batch_size) -> tf.data.Dataset:
#     ds = tf.data.Dataset.list_files(str(from_path / '*/*'), shuffle=False)
#     ds = ds.map(_process_path, num_parallel_calls=tf.data.AUTOTUNE)
#     ds = _configure_for_performance(ds, batch_size)
#
#     return ds
#
# def train(batch_size=32) -> tf.data.Dataset:
#     return _dataset(_train, batch_size)
#
# def test(batch_size=32) -> tf.data.Dataset:
#     return _dataset(_test, batch_size)

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

def train(batch_size)  -> tf.data.Dataset:
    return _train_datagen.flow_from_directory(
    "dataset/Train",
    subset="training",
    seed=666,
    batch_size=batch_size,
    target_size=image_size,
    class_mode="categorical",
)

def val(batch_size) -> tf.data.Dataset:
    return _train_datagen.flow_from_directory(
    "dataset/Train",
    subset="validation",
    seed=666,
    batch_size=batch_size,
    target_size=image_size,
    class_mode="categorical",
)

def test(batch_size) -> tf.data.Dataset:
    return _test_datagen.flow_from_directory(
    "dataset/Test",
    seed=666,
    batch_size=batch_size,
    target_size=image_size,
    class_mode="categorical",
)
