import os

import tensorflow
from tensorflow import keras
import matplotlib
import cv2
from keras.preprocessing.image import img_to_array, load_img
from keras_cv.layers import RandomFlip, RandomRotation, RandomTranslation, RandomZoom, RandomShear
from preprocessors.data_augmentor import DataAugmentor
import numpy as np
import argparse
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-o", "--output", required=True, help="path to output directory to store augmentation examples")
args = vars(ap.parse_args())

print("[INFO] loading example image...")
image = load_img(args["image"])
image_arr = img_to_array(image)
image = np.expand_dims(image, axis=0)

image_label = [[1]]

print("[INFO] augmenting data...")
# see here for demo https://www.tensorflow.org/tutorials/images/data_augmentation
data_augmentation = keras.Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.1),
    RandomZoom(0.2),
    RandomTranslation(height_factor=0.1, width_factor=0.1),
    RandomShear(0.2)
])

da = DataAugmentor(data_augmentation)
aug_data, aug_data_labels = da.augment(image_list=image, num_aug_images=4, label_list=image_label)

print(np.append(image_label, aug_data_labels, axis=0))
print("[INFO] displaying images...")
fig = plt.figure()
for idx, img in enumerate(aug_data):
    cv2.imwrite(os.path.join(args["output"], f"augmented_image_{idx}.png"), img)
    fig.add_subplot(2, 2, idx+1)
    plt.imshow(aug_data[idx].astype("uint8"))
plt.show()



