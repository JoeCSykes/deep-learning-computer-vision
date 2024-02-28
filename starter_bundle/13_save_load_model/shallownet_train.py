from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from preprocessors.simple_preprocessor import SimplePreprocessor
from preprocessors import ImageToArrayPreprocessor
from helper_funcs.simple_dataset_loader import SimpleDatasetLoader
from helper_funcs.plot_funcs import plot_training_loss_and_accuracy_keras
from nn.conv.shallownet import ShallowNet
from keras.optimizers import SGD
from imutils import paths
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-m", "--model", required=True, help="path to output model")
args = vars(ap.parse_args())

print("[INFO] loading images...")
image_paths = list(paths.list_images(args["dataset"]))

sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()

sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(image_paths, verbose=500)
data = data.astype("float") / 255.0

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

print("[INFO] compiling model...")
opt = SGD(lr=0.005)
model = ShallowNet.build(width=32, height=32, depth=3, classes=3)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] training network...")
num_epochs = 100
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=num_epochs, verbose=1)

print("[INFO] serializing model...")
model.save(args["model"])

print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=["cat", "dog", "panda"]
                            ))

plot_training_loss_and_accuracy_keras(epoch_num=num_epochs,
                                      H=H,
                                      )
