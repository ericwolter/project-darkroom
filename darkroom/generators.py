import random

import numpy as np
import PIL
import tensorflow.contrib.keras as keras

import settings as s


def _load_image(image_filename, augmented):
    image = keras.preprocessing.image.load_img(image_filename)
    if not augmented:
        image = image.resize((s.IMAGE_TARGET_SIZE, s.IMAGE_TARGET_SIZE),
                             resample=PIL.Image.LANCZOS)
        return image

    image = image.resize((s.IMAGE_FULL_SIZE, s.IMAGE_FULL_SIZE),
                         resample=PIL.Image.LANCZOS)
    random_size = random.randint(s.IMAGE_TARGET_SIZE, s.IMAGE_FULL_SIZE)
    random_offset_h = random.randint(0, s.IMAGE_FULL_SIZE - random_size)
    random_offset_v = random.randint(0, s.IMAGE_FULL_SIZE - random_size)

    window = (random_offset_h, random_offset_v,
              random_size + random_offset_h, random_size + random_offset_v)
    image = image.crop(window)

    if random_size != s.IMAGE_TARGET_SIZE:
        image = image.resize((s.IMAGE_TARGET_SIZE, s.IMAGE_TARGET_SIZE),
                             resample=PIL.Image.LANCZOS)
    if np.random.random() < 0.5:
        image = image.transpose(PIL.Image.FLIP_LEFT_RIGHT)

    return image


def _process_image(image, preprocess):
    x = keras.preprocessing.image.img_to_array(image)
    x = np.expand_dims(x, axis=0)
    x = preprocess(x)
    x = np.squeeze(x, axis=0)

    return x


def _generate_weighted_sequence(sequence, batch_size,
                                category, augmented,
                                preprocess):
    while True:
        samples = random.sample(sequence, batch_size)

        batch_x = []
        batch_y = []
        batch_w = []
        for (image_filename, score, weight) in samples:
            image = _load_image(image_filename, augmented)
            processed = _process_image(image, preprocess)

            batch_x.append(processed)
            if category:
                if score:
                    batch_y.append(np.array([1, 0]))
                else:
                    batch_y.append(np.array([0, 1]))
            else:
                batch_y.append(np.array([score]))
            batch_w.append(weight)

        yield (np.stack(batch_x), np.stack(batch_y), np.stack(batch_w))


def _generate_sequence(sequence, batch_size, category, augmented, preprocess):
    while True:
        samples = random.sample(sequence, batch_size)

        batch_x = []
        batch_y = []
        for (image_filename, score) in samples:
            image = _load_image(image_filename, augmented)
            processed = _process_image(image, preprocess)

            batch_x.append(processed)
            if category:
                if score:
                    batch_y.append(np.array([1, 0]))
                else:
                    batch_y.append(np.array([0, 1]))
            else:
                batch_y.append(np.array([score]))

        yield (np.stack(batch_x), np.stack(batch_y))


def generate_sequence(sequence, batch_size,
                      category=False, weighted=True, augmented=True,
                      preprocess=keras.applications.inception_v3.preprocess_input):
    if weighted:
        return _generate_weighted_sequence(sequence, batch_size,
                                           category, augmented,
                                           preprocess)
    else:
        return _generate_sequence(sequence, batch_size,
                                  category, augmented,
                                  preprocess)
