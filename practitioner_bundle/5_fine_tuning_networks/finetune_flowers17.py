from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from preprocessors.image_to_array_preprocessor import ImageToArrayPreprocessor
from preprocessors.aspect_aware_preprocessor import AspectAwarePreprocessor
from helper_funcs.simple_dataset_loader import SimpleDatasetLoader
from nn.conv.fcheadnet import FCHeadNet
from nn.conv.data_aug import DataAugmentor
from keras.optimizers import RMSprop, SGD
from keras.applications import VGG16
from keras.layers import Input
from keras.models import Model
from imutils import paths
import numpy as np
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-m", "--model", required=True, help="path to output model")
args = vars(ap.parse_args())

print("[INFO] loading images...")
image_paths = list(paths.list_images(args["dataset"]))
class_names = set([pt.split(os.path.sep)[-2] for pt in image_paths])

aap = AspectAwarePreprocessor(224, 224)
iap = ImageToArrayPreprocessor()

sdl = SimpleDatasetLoader(preprocessors=[aap, iap])
(data, labels) = sdl.load(image_paths, verbose=500)
data = data.astype("float") / 255.0

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

base_model = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
head_model = FCHeadNet.build(base_model, len(class_names), 256)
model = Model(inputs=base_model.input, outputs=head_model)

# this freezes the weights in the base model
for layer in base_model.layers:
    layer.trainable = False

# add data augmentation
model = DataAugmentor.build(model)

print("[INFO] compiling the model...")
# small lr to warm up FC head
# always use lr that is multiple orders of magnitude smaller than orig learning rate
opt = RMSprop(learning_rate=0.001)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics="accuracy")

# train the head of the network for a few epochs (all other
# layers are frozen) -- this will allow the new FC layers to
# start to become initialized with actual "learned" values
# versus pure random
print("[INFO] training head...")
# warm-up phase ~ 10-30 epochs depending on dataset size
model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=25, verbose=1)

print("[INFO] evaluating after initialization...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=class_names))

# now head FC layers have been trained/initialized, lets
# unfreeze the final set of CONV layers and make them trainable
for layer in base_model.layers[15:]:
    layer.trainable = True

# for the changes to the model to take effect we need to recompile
# the model, this time using SGD with a VERY small learning rate
print("[INFO] re-compiling model...")
opt = SGD(learning_rate=0.001)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] fine-tuning model...")
model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=100, verbose=1)

print("[INFO] evaluating after fine-tuning...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=class_names))

print("[INFO] serializing model...")
model.save(args["model"])

