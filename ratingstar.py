#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import sys

import tensorflow as tf
from tensorflow.python.platform import gfile
import tensorflow.contrib.keras as keras

INPUT_TENSOR_NAME = 'input_1'
OUTPUT_TENSOR_NAME = 'output_node0'
IMAGE_PATH = 'images/305163.jpg'
PREDICTION = 4.84523809523809

def create_prestige_graph():
    """"Creates a graph from saved GraphDef file and returns a Graph object.
    Returns:
        Graph holding the trained Prestige network, and various tensors we'll be
        manipulating.
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


def main(_):
    # Set up the pre-trained graph.
    graph, input_tensor, output_tensor = create_prestige_graph()
    keras.backend.set_learning_phase(0)

    config = tf.ConfigProto(device_count = {'GPU': 0})

    image = keras.preprocessing.image.load_img(
        IMAGE_PATH, target_size=(299, 299))
    array = keras.preprocessing.image.img_to_array(image)
    processed = keras.applications.inception_v3.preprocess_input(array)

    with tf.Session(graph=graph, config=config) as sess:
        with tf.device('/cpu:0'):
            pred = sess.run(output_tensor, {'input_1:0': [processed], 'batch_normalization_1/keras_learning_phase:0': False})
            print(pred)


if __name__ == '__main__':
    tf.app.run(main=main, argv=[sys.argv[0]])
