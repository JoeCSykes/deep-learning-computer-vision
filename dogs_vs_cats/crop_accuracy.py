from dogs_vs_cats.config import dogs_vs_cats_config as config
from preprocessors.image_to_array_preprocessor import ImageToArrayPreprocessor
from preprocessors.simple_preprocessor import SimplePreprocessor
from preprocessors.mean_preprocessor import MeanPreprocessor
from preprocessors.crop_preprocessor import CropPreprocessor
from import_outport.hdf5_dataset_generator import HDF5DatasetGenerator
from helper_funcs.ranked import rank5_accuracy
from keras_core.models import load_model
import keras_cv
import numpy as np
import progressbar
import json

with open(config.DATASET_MEAN) as json_file:
    means = json.load(json_file)

sp = SimplePreprocessor(227, 227)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
cp = CropPreprocessor(227, 227)
iap = ImageToArrayPreprocessor()

print("[INFO] loading model...")
model = load_model(config.MODEL_PATH)


print("[INFO] predicting on test data (no crops)...")
test_gen = HDF5DatasetGenerator(config.TEST_HDF5, 64, preprocessors=[sp, mp, iap], classes=2)
predictions = model.predict_generator(test_gen.generator(), steps=test_gen.num_images // 64, max_queue_size=64 * 2)
(rank1, _) = rank5_accuracy(predictions, test_gen.db["labels"])
print(f"[INFO] rank-1: {rank1 * 100 : .2f}%")
test_gen.close()

test_gen = HDF5DatasetGenerator(config.TEST_HDF5, 64, preprocessors=[mp], classes=2)
predictions = []

widgets = ["Evaluating: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(max_value=test_gen.num_images // 64, widgets=widgets).start()

for (idx, (images, labels)) in enumerate(test_gen.generator(passes=1)):
    for image in images:
        crops = cp.preprocess(image)
        crops = np.array([iap.preprocess(c) for c in crops], dtype="float32")

        pred = model.predict(crops)
        predictions.append(pred.mean(axis=0))

    pbar.update(idx)

pbar.finish()
print("[INFO] predicting on test data (with crops)...")
(rank1, _) = rank5_accuracy(predictions, test_gen.db["labels"])
print(f"[INFO] rank-1: {rank1 * 100 : .2f}%")
test_gen.close()
