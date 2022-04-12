import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow import keras

image_size = (45, 45)

train = keras.utils.image_dataset_from_directory(
    'dataset/Train',
    validation_split=0.2,
    subset='training',
    seed=666,
    image_size=image_size,
    batch_size=16,
    label_mode='categorical'
)

val = keras.utils.image_dataset_from_directory(
    'dataset/Train',
    validation_split=0.2,
    subset='validation',
    seed=666,
    batch_size=16,
    image_size=image_size,
    label_mode='categorical'
)

test = keras.utils.image_dataset_from_directory(
    'dataset/Test',
    seed=666,
    batch_size=16,
    image_size=image_size,
    label_mode='categorical'
)
