import os.path

from tensorflow.python.platform import gfile

import settings as s


def get_sequence(image_dir=s.DATASET_CUHKPQ_DIRECTORY, normalized=False):

    print('Using CUHKPQ dataset...')
    good_sequence = []
    bad_sequence = []
    for pathname, subdirectories, files in gfile.Walk(image_dir):
        image_filenames = [f for f in files
                           if os.path.splitext(f)[1] == '.jpg']
        if "HighQuality" in pathname:
            score = 1
            score_sequence = good_sequence
        else:
            score = 0
            score_sequence = bad_sequence

        for image_filename in image_filenames:
            image_path = os.path.join(pathname, image_filename)
            score_sequence.append((image_path, score))

    equal_size = min(len(good_sequence), len(bad_sequence))
    sequence = good_sequence[:equal_size] + bad_sequence[:equal_size]

    return sequence
