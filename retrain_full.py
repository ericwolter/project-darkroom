#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import sys
import re
import random

import numpy as np
import tensorflow as tf
import tensorflow.contrib.keras as keras

from tensorflow.python.platform import gfile

class FLAGS:
    summaries_dir = '/tmp/retrain_logs'
    model_dir = '/tmp/prestige'


def get_CUHKPQ_sequence(image_dir):
    sequence = []
    for pathname, subdirectories, files in gfile.Walk(image_dir):
        image_filenames = [f for f in files
                           if os.path.splitext(f)[1] == '.jpg']
        score = 1 if "HighQuality" in pathname else 0
        for image_filename in image_filenames:
            sequence.append((os.path.join(pathname, image_filename), score))

    return sequence


def generate_CUHKPQ_sequence(sequence, batch_size):
    while True:
        samples = random.sample(sequence, batch_size)

        batch_x = []
        batch_y = []
        for (image_filename, score) in samples:
            image = keras.preprocessing.image.load_img(
                image_filename, target_size=(299, 299))
            array = keras.preprocessing.image.img_to_array(image)
            processed = keras.applications.inception_v3.preprocess_input(array)
            batch_x.append(processed)
            batch_y.append(np.array([score]))

        yield (np.stack(batch_x), np.stack(batch_y))


def prepare_file_system():
    """Setup the directory we'll write summaries to for TensorBoard"""
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.model_dir)


def build_model(fresh_start=True, train_top_only=True):

    model = None
    if not fresh_start and tf.gfile.Exists(FLAGS.model_dir):
        files = sorted(tf.gfile.ListDirectory(FLAGS.model_dir),
            key=lambda filename:
                re.search(r"-([0-9]*\.?[0-9]*)\.hdf5", filename).group(1))

        if files and files[0]:
            print('Loading previously trained model...')
            best_filename = files[0]
            modelpath = os.path.join(FLAGS.model_dir, best_filename)
            model = keras.models.load_model(modelpath)

            if train_top_only:
                for layer in model.layers:
                    layer.trainable = False
                model.layers[-1].trainable = True

    if not model:
        print('Creating InceptionV3 model...')
        cnn = keras.applications.InceptionV3(weights='imagenet',
                                                        include_top=False,
                                                        pooling='avg')
        cnn.trainable = not train_top_only
        print('Replacing final layer...')
        output = keras.layers.Dense(1)(cnn.output)

        print('Compiling the model...')
        model = keras.models.Model(cnn.input, output)

    return model

def main(_):

    prepare_file_system()
    model = build_model(fresh_start=False, train_top_only=True)

    print('Compiling model...')
    model.compile(optimizer=keras.optimizers.Nadam(),
                  loss=tf.losses.mean_squared_error,
                  metrics=[keras.metrics.binary_accuracy])

    print('Getting samples...')
    sequence = get_CUHKPQ_sequence('dataset/')
    print('Splitting data into training, validation and test sets')
    num_samples = len(sequence)
    training_set = random.sample(sequence, int(0.8 * num_samples))
    validation_set = random.sample(sequence, int(0.1 * num_samples))
    test_set = random.sample(sequence, int(0.1 * num_samples))
    print(len(training_set), 'training samples')
    print(len(validation_set), 'validation samples')
    print(len(test_set), 'test samples')

    model_filename = "weights.{epoch:02d}-{val_loss:.2f}.hdf5"
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(FLAGS.model_dir, model_filename),
            save_best_only=True,
            period=8,
        ),
        keras.callbacks.TensorBoard(
            log_dir=FLAGS.summaries_dir,
            histogram_freq=0,
            write_graph=True)
    ]

    batch_size = 16
    print('Training the top layer...')
    model.fit_generator(
        generate_CUHKPQ_sequence(training_set, batch_size),
        len(training_set) // batch_size,
        epochs=64,
        callbacks=callbacks,
        validation_data=generate_CUHKPQ_sequence(validation_set, batch_size),
        validation_steps=len(validation_set) // batch_size)


if __name__ == '__main__':
    tf.app.run(main=main, argv=[sys.argv[0]])
