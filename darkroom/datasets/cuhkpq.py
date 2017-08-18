import os.path

from tensorflow.python.platform import gfile

import settings as s


def get_sequence(image_dir=s.DATASET_CUHKPQ_DIRECTORY, normalized=False):

    print('Using CUHKPQ dataset...')
    sequence = []
    for pathname, subdirectories, files in gfile.Walk(image_dir):
        image_filenames = [f for f in files
                           if os.path.splitext(f)[1] == '.jpg']
        score = 1 if "HighQuality" in pathname else 0
        for image_filename in image_filenames:
            sequence.append((os.path.join(pathname, image_filename),
                            score))

    return sequence
