from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from preprocessors.simple_preprocessor import SimplePreprocessor
from preprocessors.image_to_array_preprocessor import ImageToArrayPreprocessor
from helper_funcs.simple_dataset_loader import SimpleDatasetLoader
from helper_funcs.plot_funcs import plot_training_loss_and_accuracy_keras
from nn.conv.shallownet import ShallowNet
from keras.optimizers import SGD
from imutils import paths
import argparse

# commandline args
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
args = vars(ap.parse_args())

# get images
print("[INFO] load images..")
image_paths = list(paths.list_images(args["dataset"]))

# initialize img processors
sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()

# load dataset and scale raw pixel intensities to range [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(image_paths, verbose=500)
data = data.astype("float") / 255.0

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

# convert labels from integers to vectors
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

print("[INFO] compiling model...")
opt = SGD(lr=0.005)
model = ShallowNet.build(width=32, height=32, depth=3, classes=3)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# train network
print("[INFO] training network...")
num_epochs = 150
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=num_epochs, verbose=1)

# eval network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=["cat", "dog", "panda"],
                            ))

plot_training_loss_and_accuracy_keras(epoch_num=num_epochs,
                                      H=H,
                                      # savefig_path="chp_12_starter_bundle/output/shallownet_animals.png"
                                      )
