import os.path

def get_sequence(image_dir):
    print('Using CUHKPQ dataset...')
    sequence = []
    for pathname, subdirectories, files in gfile.Walk(image_dir):
        image_filenames = [f for f in files
                           if os.path.splitext(f)[1] == '.jpg']
        score = 7 if "HighQuality" in pathname else 3
        normScore = (score - 1.0) / (10.0 - 1.0)
        if normalized:
            for image_filename in image_filenames:
                sequence.append((os.path.join(pathname, image_filename),
                                normScore))
        else:
            for image_filename in image_filenames:
                sequence.append((os.path.join(pathname, image_filename),
                                score))

    return sequence
