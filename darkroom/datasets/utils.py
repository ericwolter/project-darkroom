import os.path
import hashlib

import numpy as np
from tensorflow.python.util import compat


def split_sequence(sequence):
    print('Splitting data into training, validation and test sets')
    num_samples = len(sequence)

    training_set = []
    validation_set = []
    test_set = []
    for (image_filename, score) in sequence:
        hash_name = os.path.basename(image_filename)
        hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
        percentage_hash = ((int(hash_name_hashed, 16) %
                           (num_samples + 1)) *
                           (100.0 / num_samples))

        if percentage_hash < 10:
            test_set.append((image_filename, score))
        elif percentage_hash < (10 + 10):
            validation_set.append((image_filename, score))
        else:
            training_set.append((image_filename, score))

    print(len(training_set), 'training samples')
    print(len(validation_set), 'validation samples')
    print(len(test_set), 'test samples')

    return training_set, validation_set, test_set


def weight_sequence(sequence, num_bins, min_edge, max_edge):
    hist, bin_edges = np.histogram(
        [t[1] for t in sequence], num_bins, (min_edge, max_edge))
    sample_count = np.sum(hist)
    sample_bin_ratio = np.divide(hist, sample_count)
    with np.errstate(divide='ignore'):
        sample_weights = np.divide(1, sample_bin_ratio)
        sample_weights[~np.isfinite(sample_weights)] = 0
    sample_weights = np.divide(sample_weights, np.sum(sample_weights))

    weighted_sequence = []
    for (image_filename, score) in sequence:
        indx = np.digitize(score, bin_edges)
        weight = sample_weights[indx]
        weighted_sequence.append((image_filename, score, weight))

    return weighted_sequence
