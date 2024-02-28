import numpy as np
import h5py
from keras.src.utils import to_categorical


class HDF5DatasetGenerator:
    def __init__(self, db_path: str,
                 batch_size: int,
                 preprocessors: list | None = None,
                 aug=None,
                 binarize=True,
                 classes=2):
        self.batch_size = batch_size
        self.preprocessors = preprocessors
        self.aug = aug
        self.binarize = binarize
        self.classes = classes

        self.db = h5py.File(db_path)
        self.num_images = self.db["labels"].shape[0]

    def generator(self, passes=np.inf):
        epochs = 0

        while epochs < passes:
            for i in np.arange(0, self.num_images, self.batch_size):
                images = self.db["images"][i: i + self.batch_size]
                labels = self.db["labels"][i: i + self.batch_size]

                if self.binarize:
                    labels = to_categorical(labels, self.classes)

                if self.preprocessors is not None:
                    proc_images = []

                    for image in images:
                        for p in self.preprocessors:
                            image = p.preprocess(image)

                        proc_images.append(image)

                    images = np.array(proc_images)

                yield images, labels

            epochs += 1

    def close(self):
        self.db.close()
