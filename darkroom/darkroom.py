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
from datasets import flickr
import generators
from models import PrestigeClass

# Config to turn on JIT compilation
config = tf.ConfigProto()
config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

sess = tf.Session(config=config)
keras.backend.set_session(sess)

class FLAGS:
    summaries_dir = '/tmp/retrain_logs'
    model_dir = 'D:\\deeplearning\\model\\prestige'
    bottleneck_dir = 'D:\\deeplearning\\bottleneck\\prestige_flickr'


def prepare_file_system():
    """Setup the directory we'll write summaries to for TensorBoard"""
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.model_dir)
    tf.gfile.MakeDirs(FLAGS.bottleneck_dir)


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def save_bottleneck(model, input_set, batch_size, prefix):
    bottleneck_iterations = len(input_set) // batch_size

    progbar = keras.utils.Progbar(bottleneck_iterations)
    batch_index = 0
    for samples in batch(input_set, batch_size):
        batch_x = []
        batch_y = []
        for (image_filename, score) in samples:
            image = generators._load_image(image_filename, False)
            processed = generators._process_image(image)

            batch_x.append(processed)
            batch_y.append(np.array([score]))

        bottlenecks = model.predict(np.stack(batch_x))
        scores = np.stack(batch_y)

        for (input_index, bottleneck) in enumerate(bottlenecks):
            score = scores[input_index]
            filename = prefix + str(batch_index * batch_size + input_index) + '.npz'
            with open(os.path.join(FLAGS.bottleneck_dir, filename), 'wb') as f:
                np.savez_compressed(f, bottleneck=bottleneck, score=score)

        batch_index = batch_index + 1
        progbar.update(batch_index)
    progbar.update(bottleneck_iterations, force=True)

    # for batch_index in range(bottleneck_iterations):
    #     samples = next(generator)
    #     inputs = samples[0]
    #     scores = samples[1]
    #     if weighted:
    #         weights = samples[2]
    #     bottlenecks = model.predict(inputs)
    #
    #     for (input_index, bottleneck) in enumerate(bottlenecks):
    #         score = scores[input_index]
    #         if weighted:
    #             weight = weights[input_index]
    #         filename = prefix + str(batch_index * batch_size + input_index) + '.npz'
    #         with open(os.path.join(FLAGS.bottleneck_dir, filename), 'wb') as f:
    #             if weighted:
    #                 np.savez_compressed(f, bottleneck=bottleneck, score=score, weight=weight)
    #             else:
    #                 np.savez_compressed(f, bottleneck=bottleneck, score=score)

# def save_bottleneck(model, generator, input_set, batch_size, prefix, weighted):
#     bottleneck_iterations = len(input_set) // batch_size
#
#     progbar = keras.utils.Progbar(bottleneck_iterations)
#     for batch_index in range(bottleneck_iterations):
#         samples = next(generator)
#         inputs = samples[0]
#         scores = samples[1]
#         if weighted:
#             weights = samples[2]
#         bottlenecks = model.predict(inputs)
#
#         for (input_index, bottleneck) in enumerate(bottlenecks):
#             score = scores[input_index]
#             if weighted:
#                 weight = weights[input_index]
#             filename = prefix + str(batch_index * batch_size + input_index) + '.npz'
#             with open(os.path.join(FLAGS.bottleneck_dir, filename), 'wb') as f:
#                 if weighted:
#                     np.savez_compressed(f, bottleneck=bottleneck, score=score, weight=weight)
#                 else:
#                     np.savez_compressed(f, bottleneck=bottleneck, score=score)
#
#         progbar.update(batch_index + 1)
#     progbar.update(bottleneck_iterations, force=True)


def _bottleneck_generate_sequence(sequence, batch_size, weighted):
    while True:
        samples = random.sample(sequence, batch_size)

        batch_x = []
        batch_y = []
        batch_w = []
        for filename in samples:
            loaded = np.load(os.path.join(FLAGS.bottleneck_dir, filename))
            batch_x.append(loaded['bottleneck'])
            if loaded['score']:
                batch_y.append([1,0])
            else:
                batch_y.append([0,1])

            if weighted:
                batch_w.append(loaded['weight'])

        if weighted:
            yield (np.stack(batch_x), np.stack(batch_y), np.stack(batch_w))
        else:
            yield (np.stack(batch_x), np.stack(batch_y))

def main(_):

    # print('Getting samples...')
    # training_set, validation_set, test_set = flickr.get_sequences('D:\\deeplearning\\dataset\\FLICKR')

    print('Loading model...')
    # keras.backend.set_learning_phase(False)
    prepare_file_system()
    prestige = PrestigeClass()
    prestige.create_model()
    prestige.load_top_weights('D:\\deeplearning\model\\prestige\\weights.314-0.60.hdf5')
    # prestige.convert_keras_to_tensorflow()

    # save_bottleneck(
    #     prestige.base_model,
    #     training_set, 197,
    #     prefix='bottleneck_training_')
    # save_bottleneck(
    #     prestige.base_model,
    #     validation_set, 179,
    #     prefix='bottleneck_validation_')
    #
    # return

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
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
    batch_size = 32
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
        _bottleneck_generate_sequence(training_set, batch_size, weighted=False),
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
