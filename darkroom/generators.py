import random
import warnings

import numpy as np
import PIL
from PIL import Image
Image.warnings.simplefilter("error")
warnings.simplefilter("default", DeprecationWarning)
import tensorflow.contrib.keras as keras

full_size = 333
target_size = 299

def _load_image(image_filename, augmented):
    global full_size, target_size

    image = keras.preprocessing.image.load_img(image_filename)
    if not augmented:
        image = image.resize((target_size, target_size),
                             resample=PIL.Image.LANCZOS)
        return image

    image = image.resize((full_size, full_size),
                         resample=PIL.Image.LANCZOS)
    random_size = random.randint(target_size, full_size)
    random_offset_h = random.randint(0, full_size - random_size)
    random_offset_v = random.randint(0, full_size - random_size)

    image = image.crop((random_offset_h, random_offset_v,
        random_size + random_offset_h, random_size + random_offset_v))
    if random_size != target_size:
        image = image.resize((target_size, target_size),
                             resample=PIL.Image.LANCZOS)
    if np.random.random() < 0.5:
        image = image.transpose(PIL.Image.FLIP_LEFT_RIGHT)

    return image

def _process_image(image):
    array = keras.preprocessing.image.img_to_array(image)
    processed = keras.applications.inception_v3.preprocess_input(array)

    return processed


def _generate_weighted_sequence(sequence, batch_size, augmented):
    while True:
        samples = random.sample(sequence, batch_size)

        batch_x = []
        batch_y = []
        batch_w = []
        for (image_filename, score, weight) in samples:
            image = _load_image(image_filename, augmented)
            processed = _process_image(image)

            batch_x.append(processed)
            batch_y.append(np.array([score]))
            batch_w.append(weight)

        yield (np.stack(batch_x), np.stack(batch_y), np.stack(batch_w))

def _generate_sequence(sequence, batch_size, augmented):
    while True:
        samples = random.sample(sequence, batch_size)

        batch_x = []
        batch_y = []
        for (image_filename, score) in samples:
            image = _load_image(image_filename, augmented)
            processed = _process_image(image)

            batch_x.append(processed)
            batch_y.append(np.array([score]))

        yield (np.stack(batch_x), np.stack(batch_y))


def generate_sequence(sequence, batch_size, weighted=True, augmented=True):
    if weighted:
        return _generate_weighted_sequence(sequence, batch_size, augmented)
    else:
        return _generate_sequence(sequence, batch_size, augmented)
