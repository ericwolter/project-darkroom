#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import hashlib
import os.path
import re
import sys

import numpy as np
import tensorflow as tf

from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

FLAGS = None

BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
BOTTLENECK_TENSOR_SIZE = 2048
MODEL_INPUT_WIDTH = 299
MODEL_INPUT_HEIGHT = 299
MODEL_INPUT_DEPTH = 3
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'
MAX_NUM_IMAGES = 2 ** 27 - 1  # ~134M

def create_image_lists(image_dir, testing_percentage, validation_percentage):
    """Builds a list of training images from the file system.
    Analyzes the sub folders in the image directory, splits them into stable
    training, testing, and validation sets, and returns a data structure
    describing the lists of images for each label and their paths.
    Args:
        image_dir: String path to a folder containing subfolders of images.
        testing_percentage: Integer percentage of the images to reserve for tests.
        validation_percentage: Integer percentage of images reserved for validation.
    Returns:
        A list containing an entry each image with rating, split
        into training, testing, and validation sets within each label.
    """
    if not gfile.Exists(image_dir):
        print("Image directory '" + image_dir + "' not found.")
        return None
    training_images = []
    testing_images = []
    validation_images = []

    extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']

    dataset_dirs = [os.path.join(image_dir, x) for x in gfile.ListDirectory(image_dir)]
    dataset_dirs = [x for x in dataset_dirs if gfile.IsDirectory(x)]

    for dataset_dir in dataset_dirs:
        dataset = os.path.basename(dataset_dir)
        if dataset == "CUHKPQ":
            print("Looking for quality classes in '" + dataset_dir + "'")
            quality_dirs = [os.path.join(dataset_dir, x) for x in gfile.ListDirectory(dataset_dir)]
            quality_dirs = [x for x in quality_dirs if gfile.IsDirectory(x)]
            for quality_dir in quality_dirs:
                quality = os.path.basename(quality_dir)
                if quality == "HighQuality":
                    quality = 1
                elif quality == "LowQuality":
                    quality = 0

                sub_dirs = [x[0] for x in gfile.Walk(quality_dir)]
                for sub_dir in sub_dirs:
                    print("Looking for images in '" + sub_dir + "'")
                    for extension in extensions:
                        file_glob = os.path.join(sub_dir, '*.' + extension)
                        for file_name in gfile.Glob(file_glob):
                            hash_name = re.sub(r'_nohash_.*$', '', file_name)
                            hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
                            percentage_hash = ((int(hash_name_hashed, 16) %
                                              (MAX_NUM_IMAGES + 1)) *
                                             (100.0 / MAX_NUM_IMAGES))
                            image = (file_name, quality)
                            if percentage_hash < validation_percentage:
                                validation_images.append(image)
                            elif percentage_hash < (testing_percentage + validation_percentage):
                                testing_images.append(image)
                            else:
                                training_images.append(image)
    return {
        'training': training_images,
        'validation': validation_images,
        'testing': testing_images
    }


def create_inception_graph():
    """"Creates a graph from saved GraphDef file and returns a Graph object.
    Returns:
        Graph holding the trained Inception network, and various tensors we'll be
        manipulating.
    """
    with tf.Graph().as_default() as graph:
        model_filename = os.path.join(
            FLAGS.model_dir, 'classify_image_graph_def.pb')
        with gfile.FastGFile(model_filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            bottleneck_tensor, jpeg_data_tensor, resized_input_tensor = (
              tf.import_graph_def(graph_def, name='', return_elements=[
                  BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME,
                  RESIZED_INPUT_TENSOR_NAME]))
    return graph, bottleneck_tensor, jpeg_data_tensor, resized_input_tensor

def ensure_dir_exists(dir_name):
    """Makes sure the folder exists on disk.
    Args:
        dir_name: Path string to the folder we want to create.
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def get_bottleneck_path(file_name, bottleneck_dir, category):
    """"Returns a path to a bottleneck file for a label at the given index.
    Args:
        file_name: Full name to image.
        bottleneck_dir: Folder string holding cached files of bottleneck values.
        category: Name string of set to pull images from - training, testing, or
        validation.
    Returns:
        File system path string to an image that meets the requested parameters.
    """
    return os.path.join(bottleneck_dir, category, file_name) + '.txt'

def run_bottleneck_on_image(sess, image_data, image_data_tensor,
                            bottleneck_tensor):
    """Runs inference on an image to extract the 'bottleneck' summary layer.
    Args:
    sess: Current active TensorFlow Session.
    image_data: String of raw JPEG data.
    image_data_tensor: Input data layer in the graph.
    bottleneck_tensor: Layer before the final softmax.
    Returns:
    Numpy array of bottleneck values.
    """
    bottleneck_values = sess.run(
        bottleneck_tensor,
        {image_data_tensor: image_data})
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values

def create_bottleneck_file(bottleneck_path, image_path, category, sess,
                           jpeg_data_tensor, bottleneck_tensor):
    """Create a single bottleneck file."""
    print('Creating bottleneck at ' + bottleneck_path)
    ensure_dir_exists(os.path.dirname(bottleneck_path))

    if not gfile.Exists(image_path):
        tf.logging.fatal('File does not exist %s', image_path)
    image_data = gfile.FastGFile(image_path, 'rb').read()
    try:
        bottleneck_values = run_bottleneck_on_image(
                sess, image_data, jpeg_data_tensor, bottleneck_tensor)
    except:
        raise RuntimeError('Error during processing file %s' % image_path)

    bottleneck_string = ','.join(str(x) for x in bottleneck_values)
    with open(bottleneck_path, 'w') as bottleneck_file:
        bottleneck_file.write(bottleneck_string)

def get_or_create_bottleneck(sess, file_name, category, bottleneck_dir,
                             jpeg_data_tensor, bottleneck_tensor):
    """Retrieves or calculates bottleneck values for an image.
    If a cached version of the bottleneck data exists on-disk, return that,
    otherwise calculate the data and save it to disk for future use.
    Args:
        sess: The current active TensorFlow Session.
        file_name: Full name to image.
        category: Name string of which  set to pull images from - training, testing,
        or validation.
        bottleneck_dir: Folder string holding cached files of bottleneck values.
        jpeg_data_tensor: The tensor to feed loaded jpeg data into.
        bottleneck_tensor: The output tensor for the bottleneck values.
    Returns:
        Numpy array of values produced by the bottleneck layer for the image.
    """
    bottleneck_path = get_bottleneck_path(file_name, bottleneck_dir, category)
    if not os.path.exists(bottleneck_path):
        create_bottleneck_file(bottleneck_path, file_name, category, sess,
                               jpeg_data_tensor, bottleneck_tensor)

    with open(bottleneck_path, 'r') as bottleneck_file:
        bottleneck_string = bottleneck_file.read()

    did_hit_error = False
    try:
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    except ValueError:
        print('Invalid float found, recreating bottleneck')
        did_hit_error = True
    if did_hit_error:
        create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                               image_dir, category, sess, jpeg_data_tensor,
                               bottleneck_tensor)
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        # Allow exceptions to propagate here, since they shouldn't happen after a
        # fresh creation
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    return bottleneck_values

def cache_bottlenecks(sess, image_lists, bottleneck_dir, jpeg_data_tensor,
                      bottleneck_tensor):
    """Ensures all the training, testing, and validation bottlenecks are cached.
    Because we're likely to read the same image multiple times (if there are no
    distortions applied during training) it can speed things up a lot if we
    calculate the bottleneck layer values once for each image during
    preprocessing, and then just read those cached values repeatedly during
    training. Here we go through all the images we've found, calculate those
    values, and save them off.
    Args:
        sess: The current active TensorFlow Session.
        image_lists: Dictionary of training images for each label.
        bottleneck_dir: Folder string holding cached files of bottleneck values.
        jpeg_data_tensor: Input tensor for jpeg data from file.
        bottleneck_tensor: The penultimate output layer of the graph.
    Returns:
        Nothing.
    """
    how_many_bottlenecks = 0
    ensure_dir_exists(bottleneck_dir)

    for category in ['training', 'testing', 'validation']:
        category_list = image_lists[category]

        for (file_name, _) in category_list:
            get_or_create_bottleneck(sess, file_name, category, bottleneck_dir,
                                     jpeg_data_tensor, bottleneck_tensor)

            how_many_bottlenecks += 1
            if how_many_bottlenecks % 100 == 0:
                print(str(how_many_bottlenecks) + ' bottleneck files created.')

def prepare_file_system():
  """Setup the directory we'll write summaries to for TensorBoard"""
  if tf.gfile.Exists(FLAGS.summaries_dir):
    tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
  tf.gfile.MakeDirs(FLAGS.summaries_dir)
  if FLAGS.intermediate_store_frequency > 0:
    ensure_dir_exists(FLAGS.intermediate_output_graphs_dir)
  return

def main(_):
    # Prepare necessary directories  that can be used during training
    prepare_file_system()

    # Set up the pre-trained graph.
    graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor = (
        create_inception_graph())

    # Look at the folder structure, and create lists of all the images.
    image_lists = create_image_lists(FLAGS.image_dir, FLAGS.testing_percentage,
                                    FLAGS.validation_percentage)

    with tf.Session(graph=graph) as sess:
        cache_bottlenecks(sess, image_lists, FLAGS.bottleneck_dir,
                          jpeg_data_tensor, bottleneck_tensor)
        # file_writer = tf.summary.FileWriter('logs/', sess.graph)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--image_dir',
        type=str,
        default='dataset/',
        help='Path to folders of labeled images.'
    )
    parser.add_argument(
        '--output_graph',
        type=str,
        default='/tmp/output_graph.pb',
        help='Where to save the trained graph.'
    )
    parser.add_argument(
        '--intermediate_output_graphs_dir',
        type=str,
        default='/tmp/intermediate_graph/',
        help='Where to save the intermediate graphs.'
    )
    parser.add_argument(
        '--intermediate_store_frequency',
        type=int,
        default=0,
        help='How many steps to store intermediate graph. If "0" then will not store.'
    )
    parser.add_argument(
        '--summaries_dir',
        type=str,
        default='/tmp/retrain_logs',
        help='Where to save summary logs for TensorBoard.'
    )
    parser.add_argument(
        '--testing_percentage',
        type=int,
        default=10,
        help='What percentage of images to use as a test set.'
    )
    parser.add_argument(
        '--validation_percentage',
        type=int,
        default=10,
        help='What percentage of images to use as a validation set.'
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        default='model/',
        help='Path to classify_image_graph_def.pb.'
    )
    parser.add_argument(
        '--bottleneck_dir',
        type=str,
        default='/tmp/bottleneck',
        help='Path to cache bottleneck layer values as files.'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
