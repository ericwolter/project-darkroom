import os.path

class AVAImage:
    def __init__(self, index, imageID, ratings, semantics, challengeID, imagePath):
        self.index = index
        self.imageID = imageID
        self.ratings = [int(i) for i in ratings]
        self.semantics = semantics
        self.challengeID = challengeID
        self.imagePath = imagePath

        self.numRatings = sum(self.ratings)
        self.score = 0
        for scale, count in enumerate(self.ratings):
            self.score += (scale + 1) * count
        self.score /= self.numRatings
        # normalized score
        self.normScore = (self.score - 1.0) / (10.0 - 1.0)

    def __repr__(self):
        return " ".join([
            self.imageID,
            str(self.score)])


def get_sequence(image_dir, normalized=True):
    sequence = []

    with open(os.path.join(image_dir, 'AVA.txt')) as f:
        for idx, line in enumerate(f.readlines()):
            data = line.strip().split(" ")
            imageID = data[1]
            imagePath = os.path.join(image_dir, 'images', imageID + '.jpg')
            image = AVAImage(data[0], imageID, data[2:12], data[12:14], data[14],
                imagePath)

            if normalized:
                sequence.append((image.imagePath, image.normScore))
            else:
                sequence.append((image.imagePath, image.score))

    return sequence
