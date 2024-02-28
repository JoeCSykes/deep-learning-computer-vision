from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from preprocessors.aspect_aware_preprocessor import AspectAwarePreprocessor
from preprocessors.image_to_array_preprocessor import ImageToArrayPreprocessor
from helper_funcs.simple_dataset_loader import SimpleDatasetLoader
from nn.conv.minivggnet import MiniVGGNet
from keras.optimizers import SGD
from imutils import paths
from helper_funcs.plot_funcs import plot_training_loss_and_accuracy_keras
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
args = vars(ap.parse_args())

print("[INFO] loading images...")
image_paths = list(paths.list_images(args["dataset"]))
class_names = list(set(pt.split(os.path.sep)[-2] for pt in image_paths))

aap = AspectAwarePreprocessor(64, 64)
iap = ImageToArrayPreprocessor()

sdl = SimpleDatasetLoader(preprocessors=[aap, iap])
(data, labels) = sdl.load(image_paths, verbose=500)
data = data.astype("float") / 255.0

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

print("[INFO] compling model...")
opt = SGD(learning_rate=0.05)
model = MiniVGGNet.build(width=64, height=64, depth=3, classes=len(class_names))
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=100, verbose=1)


print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=class_names))

plot_training_loss_and_accuracy_keras(
    epoch_num=100,
    H=H,
    savefig_path="practitioner_bundle/chp_2_practitioner_bundle/output/MVGGN_without_dataaug_analysis"
)

# see jupiter notebook version for graph and results
