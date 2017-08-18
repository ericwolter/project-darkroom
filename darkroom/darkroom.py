#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import sys

import tensorflow as tf
import tensorflow.contrib.keras as keras

import settings as s
import datasets.utils
from datasets import cuhkpq
from generators import generate_sequence
from models.prestige import PrestigeClass

from tensorflow.contrib.keras.python.keras.applications.imagenet_utils import preprocess_input

# Config to turn on JIT compilation
# config = tf.ConfigProto(intra_op_parallelism_threads=8)
# config.graph_options.optimizer_options.global_jit_level = \
#     tf.OptimizerOptions.ON_1

sess = tf.Session(config=config)
keras.backend.set_session(sess)


def prepare_file_system():
    """Setup the directory we'll write summaries to for TensorBoard"""
    if tf.gfile.Exists(s.SUMMARIES_DIRECTORY):
        tf.gfile.DeleteRecursively(s.SUMMARIES_DIRECTORY)
    tf.gfile.MakeDirs(s.SUMMARIES_DIRECTORY)
    tf.gfile.MakeDirs(s.MODEL_DIRECTORY)
    tf.gfile.MakeDirs(s.BOTTLENECK_DIRECTORY)


def main(_):

    print('Getting samples...')
    training_set, validation_set, test_set = datasets.utils.split_sequence(
        cuhkpq.get_sequence())

    print('Loading model...')
    prepare_file_system()
    prestige = PrestigeClass()
    prestige.create_model()

    prestige.model.compile(optimizer=keras.optimizers.Nadam(),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    batch_size = 4
    iterations = len(training_set) // batch_size
    epochs = 1024

    model_filename = "weights.{epoch:02d}-{val_loss:.2f}.hdf5"
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(s.MODEL_DIRECTORY, model_filename),
            save_best_only=True,
            period=1,
        ),
        keras.callbacks.TensorBoard(
            log_dir=s.SUMMARIES_DIRECTORY,
            histogram_freq=0,
            write_graph=True)
    ]

    print('Training...')
    training_generator = generate_sequence(
        training_set,
        batch_size,
        category=True,
        weighted=False,
        preprocess=preprocess_input)
    validation_generator = generate_sequence(
        validation_set,
        batch_size,
        category=True,
        weighted=False,
        preprocess=preprocess_input)

    prestige.model.fit_generator(
        training_generator,
        iterations,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=validation_generator,
        validation_steps=len(validation_set) // batch_size)


if __name__ == '__main__':
    tf.app.run(main=main, argv=[sys.argv[0]])
