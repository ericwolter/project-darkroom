import os.path

import settings as s


def _load_sequence(set_dir):
    sequence = []
    list_path = os.path.join(set_dir, 'list.txt')

    with open(list_path) as f:
        for idx, line in enumerate(f.readlines()):
            data = line.strip().split(',')
            imageID = data[0]
            imageClass = int(data[1])
            imagePath = os.path.join(set_dir, imageID + '.jpg')

            sequence.append((imagePath, imageClass))

    return sequence


def get_sequences(image_dir=s.DATASET_FLICKR_DIRECTORY):
    print('Using FLICKR dataset...')

    training_set = _load_sequence(os.path.join(image_dir, 'train'))
    validation_set = _load_sequence(os.path.join(image_dir, 'test'))
    test_set = []

    print(len(training_set), 'training samples')
    print(len(validation_set), 'validation samples')
    print(len(test_set), 'test samples')

    return training_set, validation_set, test_set
