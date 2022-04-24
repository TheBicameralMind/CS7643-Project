import os

from typing import Dict
import numpy as np
from pathlib import Path
from tf_clahe import clahe
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

_data_dir = Path('dataset')
_train = _data_dir / Path('Train')
_test = _data_dir / Path('Test')
_class_weights = None

image_size = (32, 32)
class_names = np.array(sorted([item.name for item in _test.glob('*')], key=lambda x: int(x)))

def _clahe(img: np.ndarray, label) -> np.ndarray:
    return clahe(img, clip_limit=0.1), label

def _get_label(file_path) -> np.ndarray:
    # Convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    return tf.cast(class_names == parts[-2], tf.float32)


def _decode_img(img):
    # Convert the compressed string to a 3D uint8 tensor
    img = tf.io.decode_jpeg(img, channels=3)
    # Resize the image to the desired size
    return tf.image.resize(img, image_size)


def _process_path(file_path):
    label = _get_label(file_path)
    # Load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = _decode_img(img)
    return img, label


def _configure_for_performance(ds, batch_size):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds

def _dataset(from_path, batch_size) -> tf.data.Dataset:
    ds = tf.data.Dataset.list_files(str(from_path / '*/*'), shuffle=False)
    ds = ds.map(_process_path, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(_clahe, num_parallel_calls=tf.data.AUTOTUNE)
    ds = _configure_for_performance(ds, batch_size)

    return ds

def train(batch_size=32) -> tf.data.Dataset:
    return _dataset(_train, batch_size)

def test(batch_size=32) -> tf.data.Dataset:
    return _dataset(_test, batch_size)

def get_class_weights() -> Dict: 
    global _class_weights
    
    if _class_weights is None: 
        weights = {int(cn): len(list((Path('dataset/Train') / cn).rglob('*'))) for cn in class_names}
        maxweight = max(weights.values())
        for i in range(len(class_names)): 
            weights[i] = maxweight / weights[i]
        _class_weights = weights
    
    return _class_weights

#
#
# _train_datagen = ImageDataGenerator(
#     rescale=1. / 255,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     validation_split=0.2
# )
# _test_datagen = ImageDataGenerator(rescale=1. / 255)
#
# train = _train_datagen.flow_from_directory(
#     "dataset/Train",
#     subset="training",
#     seed=666,
#     batch_size=batch_size,
#     target_size=image_size,
#     class_mode="categorical",
# )
#
# val = _train_datagen.flow_from_directory(
#     "dataset/Train",
#     subset="validation",
#     seed=666,
#     batch_size=batch_size,
#     target_size=image_size,
#     class_mode="categorical",
# )
#
# test = _test_datagen.flow_from_directory(
#     "dataset/Test",
#     seed=666,
#     batch_size=batch_size,
#     target_size=image_size,
#     class_mode="categorical",
# )
