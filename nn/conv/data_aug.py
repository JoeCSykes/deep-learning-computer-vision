from keras import Sequential
from keras_cv.layers import RandomFlip, RandomRotation, RandomTranslation, RandomZoom, RandomShear


class DataAugmentor:

    @staticmethod
    def build(model):
        data_augmentation = Sequential([
            RandomRotation(0.1),
            RandomTranslation(height_factor=0.1, width_factor=0.1),
            RandomShear(0.2),
            RandomZoom(0.2),
            RandomFlip("horizontal"),
        ])

        model = Sequential([data_augmentation, model])

        return model
