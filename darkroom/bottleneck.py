import os.path

from tensorflow.python.platform import gfile

import settings as s


def load_bottleneck_set(dataset):
    training_set = []
    validation_set = []

    dataset_bottleneck_directory = os.path.join(
        s.BOTTLENECK_DIRECTORY, dataset)
    for pathname, subdirectories, files in gfile.Walk(dataset_bottleneck_directory):
        for f in files:
            file_components = os.path.splitext(f)
            if file_components[1] == '.npz':
                if 'training' in file_components[0]:
                    training_set.append(f)
                elif 'validation' in file_components[0]:
                    validation_set.append(f)

    return training_set, validation_set


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def save_bottleneck(model, input_set, batch_size, prefix):
    bottleneck_iterations = len(input_set) // batch_size

    progbar = keras.utils.Progbar(bottleneck_iterations)
    batch_index = 0
    for samples in batch(input_set, batch_size):
        batch_x = []
        batch_y = []
        for (image_filename, score) in samples:
            image = generators._load_image(image_filename, False)
            processed = generators._process_image(image)

            batch_x.append(processed)
            batch_y.append(np.array([score]))

        bottlenecks = model.predict(np.stack(batch_x))
        scores = np.stack(batch_y)

        for (input_index, bottleneck) in enumerate(bottlenecks):
            score = scores[input_index]
            filename = prefix + str(batch_index * batch_size + input_index) + '.npz'
            with open(os.path.join(s.BOTTLENECK_DIRECTORY, filename), 'wb') as f:
                np.savez_compressed(f, bottleneck=bottleneck, score=score)

        batch_index = batch_index + 1
        progbar.update(batch_index)
    progbar.update(bottleneck_iterations, force=True)

    # for batch_index in range(bottleneck_iterations):
    #     samples = next(generator)
    #     inputs = samples[0]
    #     scores = samples[1]
    #     if weighted:
    #         weights = samples[2]
    #     bottlenecks = model.predict(inputs)
    #
    #     for (input_index, bottleneck) in enumerate(bottlenecks):
    #         score = scores[input_index]
    #         if weighted:
    #             weight = weights[input_index]
    #         filename = prefix + str(batch_index * batch_size + input_index) + '.npz'
    #         with open(os.path.join(FLAGS.bottleneck_dir, filename), 'wb') as f:
    #             if weighted:
    #                 np.savez_compressed(f, bottleneck=bottleneck, score=score, weight=weight)
    #             else:
    #                 np.savez_compressed(f, bottleneck=bottleneck, score=score)

# def save_bottleneck(model, generator, input_set, batch_size, prefix, weighted):
#     bottleneck_iterations = len(input_set) // batch_size
#
#     progbar = keras.utils.Progbar(bottleneck_iterations)
#     for batch_index in range(bottleneck_iterations):
#         samples = next(generator)
#         inputs = samples[0]
#         scores = samples[1]
#         if weighted:
#             weights = samples[2]
#         bottlenecks = model.predict(inputs)
#
#         for (input_index, bottleneck) in enumerate(bottlenecks):
#             score = scores[input_index]
#             if weighted:
#                 weight = weights[input_index]
#             filename = prefix + str(batch_index * batch_size + input_index) + '.npz'
#             with open(os.path.join(FLAGS.bottleneck_dir, filename), 'wb') as f:
#                 if weighted:
#                     np.savez_compressed(f, bottleneck=bottleneck, score=score, weight=weight)
#                 else:
#                     np.savez_compressed(f, bottleneck=bottleneck, score=score)
#
#         progbar.update(batch_index + 1)
#     progbar.update(bottleneck_iterations, force=True)


def _bottleneck_generate_sequence(sequence, batch_size, weighted):
    while True:
        samples = random.sample(sequence, batch_size)

        batch_x = []
        batch_y = []
        batch_w = []
        for filename in samples:
            loaded = np.load(os.path.join(s.BOTTLENECK_DIRECTORY, filename))
            batch_x.append(loaded['bottleneck'])
            if loaded['score']:
                batch_y.append([1, 0])
            else:
                batch_y.append([0, 1])

            if weighted:
                batch_w.append(loaded['weight'])

        if weighted:
            yield (np.stack(batch_x), np.stack(batch_y), np.stack(batch_w))
        else:
            yield (np.stack(batch_x), np.stack(batch_y))
