from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.utils.image_utils import img_to_array
from keras.utils import np_utils
from nn.conv.lenet import LeNet
from nn.conv.minivggnet import MiniVGGNet
from imutils import paths
from helper_funcs.plot_funcs import plot_training_loss_and_accuracy_keras
import numpy as np
import argparse
import imutils
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to dataset of faces")
ap.add_argument("-m", "--model", required=True, help="path to output model")
ap.add_argument("-g", "--graph", help="path to output graph")
args = vars(ap.parse_args())

data = []
labels = []

for image_path in sorted(list(paths.list_images(args["dataset"]))):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = imutils.resize(image, width=28)
    image = img_to_array(image)
    data.append(image)

    label = image_path.split(os.path.sep)[-2]
    label = "smiling" if label == "positives" else "not_smiling"
    labels.append(label)

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

le = LabelEncoder().fit(labels)
labels = np_utils.to_categorical(le.transform(labels), 2)

# account for skew in labeled data
class_totals = labels.sum(axis=0)  # array showing the total number of examples per class
class_weight = (class_totals.max() / class_totals)  # calculates the weight of each class (larges class normalised)
class_weight = {i: class_weight[i]
                for i in range(len(class_totals))}

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

print("[INFO] compiling model...")
# model = LeNet.build(width=28, height=28, depth=1, classes=2)
model = MiniVGGNet.build(width=28, height=28, depth=1, classes=2)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

checkpoint = ModelCheckpoint(args["model"], monitor="val_loss", mode="min", save_best_only=True, verbose=1)
callbacks = [checkpoint]

print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY), class_weight=class_weight,
              batch_size=64, epochs=15, callbacks=callbacks, verbose=1)

print("[INFO} evaluating network...")
predictions = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=le.classes_))

if args["graph"]:
    plot_training_loss_and_accuracy_keras(
        epoch_num=15,
        H=H,
        savefig_path=args["graph"]
    )
else:
    plot_training_loss_and_accuracy_keras(
        epoch_num=15,
        H=H
    )
