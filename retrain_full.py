#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import sys
import random

import numpy as np
import tensorflow as tf

from tensorflow.python.platform import gfile


class FLAGS:
    summaries_dir = '/tmp/retrain_logs'


def getCUHKPQSequence(image_dir):
    sequence = []
    for pathname, subdirectories, files in gfile.Walk(image_dir):
        image_filenames = [f for f in files
                           if os.path.splitext(f)[1] == '.jpg']
        score = 1 if "HighQuality" in pathname else 0
        for image_filename in image_filenames:
            sequence.append((os.path.join(pathname, image_filename), score))

    return sequence


def generateCUHKPQSequence(sequence, batch_size):
    while True:
        samples = random.sample(sequence, batch_size)

        batch_x = []
        batch_y = []
        for (image_filename, score) in samples:
            image = tf.contrib.keras.preprocessing.image.load_img(
                image_filename, target_size=(299, 299))
            array = tf.contrib.keras.preprocessing.image.img_to_array(image)
            processed = tf.contrib.keras.applications.inception_v3.preprocess_input(array)
            batch_x.append(processed)
            batch_y.append(np.array([score]))

        yield (np.stack(batch_x), np.stack(batch_y))


def prepare_file_system():
    """Setup the directory we'll write summaries to for TensorBoard"""
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)


def main(_):

    prepare_file_system()

    print('Creating InceptionV3 model...')
    cnn = tf.contrib.keras.applications.InceptionV3(weights='imagenet',
                                                    include_top=False,
                                                    pooling='avg')
    cnn.trainable = False
    print('Replacing final layer...')
    output = tf.contrib.keras.layers.Dense(1)(cnn.output)

    print('Compiling the model...')
    model = tf.contrib.keras.models.Model(cnn.input, output)
    model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
                  loss=tf.losses.mean_squared_error,
                  metrics=[tf.contrib.keras.metrics.binary_accuracy])

    print('Getting samples...')
    sequence = getCUHKPQSequence('dataset/')
    print('Splitting data into training, validation and test sets')
    num_samples = len(sequence)
    training_set = random.sample(sequence, int(0.8 * num_samples))
    validation_set = random.sample(sequence, int(0.1 * num_samples))
    test_set = random.sample(sequence, int(0.1 * num_samples))
    print(len(training_set), 'training samples')
    print(len(validation_set), 'validation samples')
    print(len(test_set), 'test samples')

    callbacks = [
        tf.contrib.keras.callbacks.TensorBoard(
            log_dir=FLAGS.summaries_dir,
            histogram_freq=0,
            write_graph=True)
    ]

    batch_size = 16
    print('Training the top layer...')
    model.fit_generator(
        generateCUHKPQSequence(training_set, batch_size),
        len(training_set) // batch_size,
        epochs=64,
        callbacks=callbacks,
        validation_data=generateCUHKPQSequence(validation_set, batch_size),
        validation_steps=len(validation_set) // batch_size)


if __name__ == '__main__':
    tf.app.run(main=main, argv=[sys.argv[0]])
