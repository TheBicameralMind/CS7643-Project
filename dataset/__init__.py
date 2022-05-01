from typing import Dict
import numpy as np
from pathlib import Path
from skimage.exposure import equalize_adapthist as clahe
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

_data_dir = Path('dataset')
_train = _data_dir / Path('Train')
_test = _data_dir / Path('Test')
_class_weights = None

image_size = (32, 32)
class_names = np.array(sorted([item.name for item in _test.glob('*')], key=lambda x: int(x)))

def _clahe(img: np.ndarray) -> np.ndarray:
    return clahe(img.astype(np.uint8), clip_limit=0.1)


def get_class_weights() -> Dict: 
    global _class_weights
    
    if _class_weights is None: 
        weights = {int(cn): len(list((Path('dataset/Train') / cn).rglob('*'))) for cn in class_names}
        maxweight = max(weights.values())
        for i in range(len(class_names)): 
            weights[i] = maxweight / weights[i]
        _class_weights = weights
    
    return _class_weights


_train_datagen = ImageDataGenerator(
	rotation_range=10,
	zoom_range=0.15,
	width_shift_range=0.1,
	height_shift_range=0.1,
	shear_range=0.15,
	horizontal_flip=False,
	vertical_flip=False,
	fill_mode="nearest",
    validation_split=0.2,
    preprocessing_function=_clahe
)
_test_datagen = ImageDataGenerator(preprocessing_function=_clahe)

def train(batch_size=32) -> tf.data.Dataset: 
    return _train_datagen.flow_from_directory(
        "dataset/Train",
        subset="training",
        seed=666,
        batch_size=batch_size,
        target_size=image_size,
        class_mode="categorical",
    ) 

def val(batch_size=32) -> tf.data.Dataset: 
    return _train_datagen.flow_from_directory(
        "dataset/Train",
        subset="validation",
        seed=666,
        batch_size=batch_size,
        target_size=image_size,
        class_mode="categorical",
    )

def test(batch_size=32) -> tf.data.Dataset: 
    return _test_datagen.flow_from_directory(
        "dataset/Test",
        seed=666,
        batch_size=batch_size,
        target_size=image_size,
        class_mode="categorical",
    )
