from config import tiny_imagenet_config as config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from import_outport.hdf5_dataset_writer import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar
import json
import cv2
import os


train_paths = list(paths.list_images(config.TRAIN_IMAGES))
train_labels = [p.split(os.path.sep)[-3] for p in train_paths]
print(train_paths)
# le = LabelEncoder()
# train_labels = le.fit_transform(train_labels)

# split = train_test_split(train_paths, train_labels, test_size=config.NUM_TEST_IMAGES, stratify=train_labels,
#                          random_state=42)
# (train_paths, test_paths, train_labels, test_labels) = split
#
# with open(config.VAL_MAPPINGS) as f:
#     M = f.read().strip().split("\n")
# M = [r.split("\t")[:2] for r in M]
# val_paths = [os.path.sep.join([config.VAL_IMAGES, m[0]]) for m in M]
# val_labels = le.transform([m[1] for m in M])
#
# for i in range(10):
#     print(f"val_path = {val_paths[i]}, val_label = {val_labels[i]}")