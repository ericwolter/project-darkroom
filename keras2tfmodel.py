#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import tensorflow as tf
import tensorflow.contrib.keras as keras
keras.backend.set_learning_phase(False)

json_filepath = 'model/prestige.json'
weight_filepath = 'model/weights.23-0.27.hdf5'
num_output = 1
prefix_output_node_names_of_final_network = 'output_node'
output_directory = 'model/'
output_graph_name = 'prestige_trained.pb'

print('Creating InceptionV3 model...')
cnn = keras.applications.InceptionV3(weights='imagenet',
                                     include_top=False,
                                     pooling='avg')
cnn.trainable = False
print('Replacing final layer...')
output = keras.layers.Dense(1)(cnn.output)
output.trainable = False

print('Compiling the model...')
model = keras.models.Model(cnn.input, output)
print('Disabling learning...')
for layer in model.layers:
    layer.trainable = False

print('Saving model to json...')
with open(json_filepath, 'w') as json_file:
    json_file.write(model.to_json())

print('Loading model from json...')
# load json and create model
with open(json_filepath, 'r') as json_file:
    model_json = json_file.read()

model = keras.models.model_from_json(model_json)

print('Loading weights...')
model.load_weights(weight_filepath)

print('Adding output tensor...')
pred = [None]*num_output
pred_node_names = [None]*num_output
for i in range(num_output):
    pred_node_names[i] = prefix_output_node_names_of_final_network+str(i)
    pred[i] = tf.identity(model.output[i], name=pred_node_names[i])
print('output nodes names are: ', pred_node_names)

print('Saving model to tf format...')
sess = keras.backend.get_session()
constant_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)
tf.train.write_graph(constant_graph, output_directory, output_graph_name, as_text=False)
print('saved the constant graph (ready for inference) at: ', os.path.join(output_directory, output_graph_name))
