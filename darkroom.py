#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import hashlib
import os.path
import random
import re
import sys

from tensorflow.python import debug as tf_debug

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
        category: Name string of which set to pull images from - training, testing,
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

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def add_final_training_ops(final_tensor_name, bottleneck_tensor):
    """Adds a new fully-connected layer for training.
    We need to retrain the top layer to assign the aesthetics score, so this function
    adds the right operations to the graph, along with some variables to hold the
    weights, and then sets up all the gradients for the backward pass.
    Args:
        final_tensor_name: Name string for the new final node that produces results.
        bottleneck_tensor: The output of the main CNN graph.
    Returns:
        The tensors for the training and cross entropy results, and tensors for the
        bottleneck input and ground truth input.
    """
    with tf.name_scope('input'):
        bottleneck_input = tf.placeholder_with_default(
            bottleneck_tensor, shape=[None, BOTTLENECK_TENSOR_SIZE],
            name='BottleneckInputPlaceholder')

        ground_truth_input = tf.placeholder(tf.float32, [None, 1], name='GroundTruthInput')

    # Organizing the following ops as `final_training_ops` so they're easier
    # to see in TensorBoard
    layer_name = 'final_training_ops'
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            initial_value = tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, 1],
                                                 stddev=0.001)

            layer_weights = tf.Variable(initial_value, name='final_weights')
            variable_summaries(layer_weights)
        with tf.name_scope('biases'):
            layer_biases = tf.Variable(tf.zeros([1]), name='final_biases')
            variable_summaries(layer_biases)

        with tf.name_scope('Wx_plus_b'):
            logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases
            tf.summary.histogram('pre_activations', logits)

    final_tensor = tf.identity(logits, name=final_tensor_name)
    tf.summary.histogram('activations', final_tensor)

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.squared_difference(logits, ground_truth_input))
    tf.summary.scalar('loss', loss)

    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
        train_step = optimizer.minimize(loss)

    return (train_step, loss, bottleneck_input, ground_truth_input, final_tensor)

def add_evaluation_step(result_tensor, ground_truth_tensor):
    """Inserts the operations we need to evaluate the accuracy of our results.
    Args:
        result_tensor: The new final node that produces results.
        ground_truth_tensor: The node we feed ground truth data
        into.
    Returns:
        evaluation step
    """
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            difference = tf.abs(result_tensor - ground_truth_tensor)
            correct_prediction = tf.less_equal(difference, 0.1)
        with tf.name_scope('accuracy'):
            evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', evaluation_step)
    return evaluation_step

def get_random_cached_bottlenecks(sess, image_lists, how_many, category,
                                  bottleneck_dir, jpeg_data_tensor,
                                  bottleneck_tensor):
    """Retrieves bottleneck values for cached images.
    If no distortions are being applied, this function can retrieve the cached
    bottleneck values directly from disk for images. It picks a random set of
    images from the specified category.
    Args:
        sess: Current TensorFlow Session.
        image_lists: Dictionary of training images for each label.
        how_many: If positive, a random sample of this size will be chosen.
        If negative, all bottlenecks will be retrieved.
        category: Name string of which set to pull from - training, testing, or
        validation.
        bottleneck_dir: Folder string holding cached files of bottleneck values.
        jpeg_data_tensor: The layer to feed jpeg image data into.
        bottleneck_tensor: The bottleneck output layer of the CNN graph.
    Returns:
        List of bottleneck arrays, their corresponding ground truths, and the
        relevant filenames.
    """
    bottlenecks = []
    ground_truths = []
    filenames = []

    category_list = image_lists[category]

    if how_many >= 0:
        for unused_i in range(how_many):
            (file_name, quality) = random.choice(category_list)
            bottleneck = get_or_create_bottleneck(sess, file_name,
                                     category, bottleneck_dir,
                                     jpeg_data_tensor, bottleneck_tensor)
            ground_truth = np.zeros(1, dtype=np.float32)
            ground_truth[0] = quality

            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)
            filenames.append(file_name)
    else:
        # Retrieve all bottlenecks.
        for (file_name, quality) in category_list:
            bottleneck = get_or_create_bottleneck(sess, file_name,
                                     category, bottleneck_dir,
                                     jpeg_data_tensor, bottleneck_tensor)
            ground_truth = np.zeros(1, dtype=np.float32)
            ground_truth[0] = quality

            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)
            filenames.append(file_name)

    return bottlenecks, ground_truths, filenames

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
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        cache_bottlenecks(sess, image_lists, FLAGS.bottleneck_dir,
                          jpeg_data_tensor, bottleneck_tensor)
        # file_writer = tf.summary.FileWriter('logs/', sess.graph)
        (train_step, loss, bottleneck_input, ground_truth_input,
         final_tensor) = add_final_training_ops(FLAGS.final_tensor_name,
                                                bottleneck_tensor)

        # Create the operations we need to evaluate the accuracy of our new layer.
        evaluation_step = add_evaluation_step(
            final_tensor, ground_truth_input)

        # Merge all the summaries and write them out to the summaries_dir
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
                                             sess.graph)

        validation_writer = tf.summary.FileWriter(
            FLAGS.summaries_dir + '/validation')

        # Set up all our weights to their initial default values.
        init = tf.global_variables_initializer()
        sess.run(init)

        # Run the training for as many cycles as requested on the command line.
        for i in range(FLAGS.how_many_training_steps):
            (train_bottlenecks,
             train_ground_truth, _) = get_random_cached_bottlenecks(
                sess, image_lists, FLAGS.train_batch_size, 'training',
                FLAGS.bottleneck_dir, jpeg_data_tensor, bottleneck_tensor)

            train_summary, _ = sess.run(
                [merged, train_step],
                feed_dict={bottleneck_input: train_bottlenecks,
                           ground_truth_input: train_ground_truth})
            train_writer.add_summary(train_summary, i)

            # Every so often, print out how well the graph is training.
            is_last_step = (i + 1 == FLAGS.how_many_training_steps)
            if (i % FLAGS.eval_step_interval) == 0 or is_last_step:
                train_accuracy, loss_value = sess.run(
                    [evaluation_step, loss],
                    feed_dict={bottleneck_input: train_bottlenecks,
                               ground_truth_input: train_ground_truth})
                print('%s: Step %d: Train accuracy = %.1f%%' % (datetime.now(), i,
                                                train_accuracy * 100))
                print('%s: Step %d: Loss = %f' % (datetime.now(), i,
                                           loss_value))

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
        '--how_many_training_steps',
        type=int,
        default=4000,
        help='How many training steps to run before ending.'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='How large a learning rate to use when training.'
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
        '--eval_step_interval',
        type=int,
        default=10,
        help='How often to evaluate the training results.'
    )
    parser.add_argument(
        '--train_batch_size',
        type=int,
        default=1000,
        help='How many images to train on at a time.'
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
        default='bottleneck/',
        help='Path to cache bottleneck layer values as files.'
    )
    parser.add_argument(
        '--final_tensor_name',
        type=str,
        default='final_result',
        help="""\
        The name of the output classification layer in the retrained graph.\
        """
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
