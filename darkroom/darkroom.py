#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import sys
import re


import numpy as np
import h5py
import tensorflow as tf
import tensorflow.contrib.keras as keras
from tensorflow.python.platform import gfile
import random

import datasets.utils
from datasets import ava
import generators
from models import Prestige

# Config to turn on JIT compilation
config = tf.ConfigProto()
config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

sess = tf.Session(config=config)
keras.backend.set_session(sess)

class FLAGS:
    summaries_dir = '/tmp/retrain_logs'
    model_dir = 'D:\deeplearning\model\prestige'
    bottleneck_dir = 'D:\\deeplearning\\bottleneck\\prestige'


def prepare_file_system():
    """Setup the directory we'll write summaries to for TensorBoard"""
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.model_dir)
    tf.gfile.MakeDirs(FLAGS.bottleneck_dir)


def save_bottleneck(model, generator, input_set, batch_size, prefix, weighted):
    bottleneck_iterations = len(input_set) // batch_size * 2

    progbar = keras.utils.Progbar(bottleneck_iterations)
    for batch_index in range(bottleneck_iterations):
        samples = next(generator)
        inputs = samples[0]
        scores = samples[1]
        if weighted:
            weights = samples[2]
        bottlenecks = model.predict(inputs)

        for (input_index, bottleneck) in enumerate(bottlenecks):
            score = scores[input_index]
            if weighted:
                weight = weights[input_index]
            filename = prefix + str(batch_index * batch_size + input_index) + '.npz'
            with open(os.path.join(FLAGS.bottleneck_dir, filename), 'wb') as f:
                if weighted:
                    np.savez_compressed(f, bottleneck=bottleneck, score=score, weight=weight)
                else:
                    np.savez_compressed(f, bottleneck=bottleneck, score=score)

        progbar.update(batch_index + 1)
    progbar.update(bottleneck_iterations, force=True)


def _bottleneck_generate_sequence(sequence, batch_size, weighted):
    while True:
        samples = random.sample(sequence, batch_size)

        batch_x = []
        batch_y = []
        batch_w = []
        for filename in samples:
            loaded = np.load(os.path.join(FLAGS.bottleneck_dir, filename))
            batch_x.append(loaded['bottleneck'])
            batch_y.append(loaded['score'])
            if weighted:
                batch_w.append(loaded['weight'])

        if weighted:
            yield (np.stack(batch_x), np.stack(batch_y), np.stack(batch_w))
        else:
            yield (np.stack(batch_x), np.stack(batch_y))

def main(_):

    print('Loading model...')
    # keras.backend.set_learning_phase(False)
    prepare_file_system()
    prestige = Prestige()
    prestige.create_model()
    # prestige.load_top_weights('D:\\deeplearning\model\\prestige_final\\top_weights.hdf5')
    # prestige.convert_keras_to_tensorflow()

    # print('Getting samples...')
    # sequence = ava.get_sequence('D:\deeplearning\dataset\AVA', normalized=True)
    # training_set, validation_set, test_set = datasets.utils.split_sequence(sequence)
    # training_set = datasets.utils.weight_sequence(training_set, 90, 0, 1)

    # save_bottleneck(
    #     base_model,
    #     generators.generate_sequence(training_set, 32, weighted=True, augmented=True),
    #     training_set, 32,
    #     prefix='bottleneck_training_', weighted=True)
    # save_bottleneck(
    #     base_model,
    #     generators.generate_sequence(validation_set, 32, weighted=False, augmented=False),
    #     validation_set, 32,
    #     prefix='bottleneck_validation_', weighted=False)

    training_set = []
    validation_set = []
    for pathname, subdirectories, files in gfile.Walk(FLAGS.bottleneck_dir):
        for f in files:
            file_components = os.path.splitext(f)
            if file_components[1] == '.npz':
                if 'training' in file_components[0]:
                    training_set.append(f)
                elif 'validation' in file_components[0]:
                    validation_set.append(f)

    prestige.top_model.compile(optimizer=keras.optimizers.Nadam(),
                      loss=keras.losses.mean_squared_error,
                      metrics=[keras.metrics.mean_squared_error])
    batch_size = 8
    iterations = len(training_set) // batch_size
    epochs = 1024

    model_filename = "weights.{epoch:02d}-{val_loss:.2f}.hdf5"
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(FLAGS.model_dir, model_filename),
            save_best_only=True,
            period=1,
        ),
        keras.callbacks.TensorBoard(
            log_dir=FLAGS.summaries_dir,
            histogram_freq=0,
            write_graph=True)
    ]

    print('Training...')
    prestige.top_model.fit_generator(
        _bottleneck_generate_sequence(training_set, batch_size, weighted=True),
        iterations,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=_bottleneck_generate_sequence(validation_set, batch_size, weighted=False),
        validation_steps=len(validation_set) // batch_size)
    # model.fit_generator(
    #     generators.generate_sequence(training_set, batch_size, weighted=True, augmented=True),
    #     iterations,
    #     epochs=epochs,
    #     callbacks=callbacks,
    #     validation_data=generators.generate_sequence(validation_set, batch_size, weighted=False, augmented=False),
    #     validation_steps=len(validation_set) // batch_size)


if __name__ == '__main__':
    tf.app.run(main=main, argv=[sys.argv[0]])
