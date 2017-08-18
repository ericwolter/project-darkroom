from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import tensorflow as tf
import tensorflow.contrib.keras as keras

output_directory = 'model/'
output_graph_name = 'prestige_trained.pb'

class Prestige:
    # def load_model(model_path):
    #     # workaround:
    #     # https://github.com/fchollet/keras/issues/4044
    #     with h5py.File(model_path, 'a') as f:
    #         if 'optimizer_weights' in f.keys():
    #             del f['optimizer_weights']
    #
    #     self.model = keras.models.load_model(model_path)
    def create_model(self):
        print('Creating InceptionV3 model...')
        self.base_model = keras.applications.InceptionV3(weights='imagenet',
                                                    include_top=False,
                                                    pooling='avg')
        self.base_model.trainable = False

        print('Adding top model...')
        self.top_model = keras.models.Sequential()
        self.top_model.add(keras.layers.Dense(1, input_shape=self.base_model.output_shape[1:]))

        print('Combining base and top model...')
        self.model = keras.models.Model(self.base_model.input, self.top_model(self.base_model.output))

    def load_weights(self, model_weights_path):
        self.model.load_weights(model_weights_path)

    def load_base_weights(self, base_weights_path):
        self.base_model.load_weights(base_weights_path)

    def load_top_weights(self, top_weights_path):
        self.top_model.load_weights(top_weights_path)

    def set_full_training(self):
        self.model.trainable = True

    def set_top_training(self):
        self.base_model.trainable = False
        self.top_model.trainable = True

    def convert_keras_to_tensorflow(self):
        print('Adding output tensor...')
        num_output = self.model.output.shape[1]
        pred = [None]*num_output
        pred_node_names = [None]*num_output
        for i in range(num_output):
            pred_node_names[i] = 'output_node' + str(i)
            pred[i] = tf.identity(self.model.output[i], name=pred_node_names[i])
        print('output nodes names are: ', pred_node_names)

        print('Saving model to tf format...')
        sess = keras.backend.get_session()
        constant_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)
        tf.train.write_graph(constant_graph, output_directory, output_graph_name, as_text=False)
        print('saved the constant graph (ready for inference) at: ', os.path.join(output_directory, output_graph_name))

        self.model = None
        self.base_model = None
        self.top_model = None

class PrestigeClass(Prestige):
    def create_model(self):
        print('Creating InceptionV3 model...')
        self.base_model = keras.applications.InceptionV3(weights='imagenet',
                                                    include_top=False,
                                                    pooling='avg')
        self.base_model.trainable = False

        print('Adding top model...')
        self.top_model = keras.models.Sequential()
        self.top_model.add(keras.layers.Dense(2, activation='softmax', name='predictions', input_shape=self.base_model.output_shape[1:]))

        print('Combining base and top model...')
        self.model = keras.models.Model(self.base_model.input, self.top_model(self.base_model.output))
