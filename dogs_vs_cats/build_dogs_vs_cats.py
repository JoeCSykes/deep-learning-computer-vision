
from dogs_vs_cats.config import dogs_vs_cats_config as config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from preprocessors.aspect_aware_preprocessor import AspectAwarePreprocessor
from import_outport.hdf5_dataset_writer import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar
import json
import cv2

train_paths = list(paths.list_images(config.IMAGES_PATH))
train_labels = [p.split("/")[-1].split(".")[0] for p in train_paths]

le = LabelEncoder()
train_labels = le.fit_transform(train_labels)

# generate test data
split = train_test_split(train_paths,
                         train_labels,
                         test_size=config.NUM_TEST_IMAGES,
                         stratify=train_labels,
                         random_state=42
                         )
(train_paths, test_paths, train_labels, test_labels) = split

# generate val data
split = train_test_split(train_paths,
                         train_labels,
                         test_size=config.NUM_VAL_IMAGES,
                         stratify=train_labels,
                         random_state=42
                         )
(train_paths, val_paths, train_labels, val_labels) = split

datasets = [("train", train_paths, train_labels, config.TRAIN_HDF5),
            ("val", val_paths, val_labels, config.VAL_HDF5),
            ("test", test_paths, test_labels, config.TEST_HDF5)]

aap = AspectAwarePreprocessor(256, 256)
# stores avg pixel intensity per channel
(R, G, B) = ([], [], [])

for (dtype, paths, labels, output_path) in datasets:
    print(f"[INFO] building {output_path}...")
    writer = HDF5DatasetWriter((len(paths), 256, 256, 3), output_path)

    widgets = ["buildingDataset: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(max_value=len(paths), widgets=widgets).start()

    for (idx, (path, label)) in enumerate(zip(paths, labels)):
        image = cv2.imread(path)
        image = aap.preprocess(image)

        if dtype == "train":
            (b, g, r) = cv2.mean(image)[:3]
            R.append(r)
            G.append(g)
            B.append(b)

        writer.add([image], [label])
        pbar.update(idx)

    pbar.finish()
    writer.close()

print("[INFO] serializing means...")
D = {"R": np.mean(R), "G": np.mean(G), "B": np.mean(B)}
with open(config.DATASET_MEAN, "w") as f:
    f.write(json.dumps(D))
