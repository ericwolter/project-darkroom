#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import sys
import random

import tensorflow as tf
import numpy as np
from tensorflow.python.platform import gfile
import tensorflow.contrib.keras as keras

INPUT_TENSOR_NAME = 'input_1_1:0'
OUTPUT_TENSOR_NAME = 'output_node0:0'
IMAGE_PATH = 'images/305163.jpg'
PREDICTION = 4.84523809523809


def create_prestige_graph():
    """"Creates a graph from saved GraphDef file and returns a Graph object.
    Returns:
        Graph holding the trained Prestige network, and various tensors we'll
        be manipulating.
    """
    with tf.Graph().as_default() as graph:
        model_filename = os.path.join(
            'model/', 'prestige_trained.pb')
        with gfile.FastGFile(model_filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            input_tensor, output_tensor = (
              tf.import_graph_def(graph_def, name='', return_elements=[
                INPUT_TENSOR_NAME, OUTPUT_TENSOR_NAME]))
    return graph, input_tensor, output_tensor


def load_image(image_path):
    image = keras.preprocessing.image.load_img(
        image_path, target_size=(299, 299))
    array = keras.preprocessing.image.img_to_array(image)
    processed = keras.applications.inception_v3.preprocess_input(array)

    return processed


def getScore(sess, image_path, input_tensor, output_tensor):
    processed = load_image(image_path)
    score = sess.run(
        output_tensor, {
            input_tensor: [processed]})
    return score


def score_directory(sess, directory_path, input_tensor, output_tensor):
    for pathname, subdirectories, files in gfile.Walk(directory_path):
        image_paths = [f for f in files
                       if os.path.splitext(f)[1] == '.JPG']
        image_paths = [os.path.join(directory_path, f) for f in image_paths]
        test_set = image_paths

        results = []
        prog = keras.utils.Progbar(len(test_set))
        for idx, image_path in enumerate(test_set):
            score = getScore(sess, image_path, input_tensor, output_tensor)[0]
            results.append((image_path, score))
            prog.update(idx)
        prog.update(len(test_set), force=True)
        print()

        results = sorted(results, key=lambda x: x[1], reverse=True)
        best_images = results[:3]
        worst_images = results[-3:]

        with open('/tmp/results.txt', 'w') as results_file:
            results_file.write(str(results))

        print(best_images)
        print(worst_images)


def score_CUHKPQ(sess, input_tensor, output_tensor):
    with open('CUHKPQ.txt', 'r') as dataset_file:
        image_paths = dataset_file.readlines()
        image_paths = [x.strip() for x in image_paths]
        test_set = image_paths
        # test_set = random.sample(image_paths, 20)

        results = []
        prog = keras.utils.Progbar(len(test_set))
        for idx, image_path in enumerate(test_set):
            score = getScore(sess, image_path, input_tensor, output_tensor)[0]
            results.append((image_path, score))
            prog.update(idx)
        prog.update(len(test_set), force=True)
        print()

        results = sorted(results, key=lambda x: x[1], reverse=True)
        best_images = results[:3]
        worst_images = results[-3:]

        with open('/tmp/results.txt', 'w') as results_file:
            results_file.write(str(results))

        print(best_images)
        print(worst_images)


def main(_):
    # Set up the pre-trained graph.
    graph, input_tensor, output_tensor = create_prestige_graph()

    with tf.Session(graph=graph) as sess:
        # score_CUHKPQ(sess, input_tensor, output_tensor)
        score_directory(sess, '/Users/eric/Downloads/Nikon',
                        input_tensor, output_tensor)
        # print(getScore(sess, IMAGE_PATH, input_tensor, output_tensor))


if __name__ == '__main__':
    tf.app.run(main=main, argv=[sys.argv[0]])
