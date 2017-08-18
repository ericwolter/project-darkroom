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
            for image_filename in image_filenames:
                good_sequence.append((os.path.join(pathname, image_filename),
                                score))
        else:
            score = 0
            for image_filename in image_filenames:
                bad_sequence.append((os.path.join(pathname, image_filename),
                                score))
        score = 1 if "HighQuality" in pathname else 0

    equal_size = min(len(good_sequence), len(bad_sequence))
    sequence = good_sequence[:equal_size] + bad_sequence[:equal_size]

    return sequence
