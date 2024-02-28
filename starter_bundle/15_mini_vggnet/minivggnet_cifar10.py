import matplotlib
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from nn.conv.minivggnet import MiniVGGNet
from keras.optimizers import SGD
from keras.datasets import cifar10
from helper_funcs.plot_funcs import plot_training_loss_and_accuracy_keras
import argparse

# set the matplotlib backend so figures can be saved in the background
matplotlib.use("Agg")

# model params
epoch_num = 40
batch_size = 64
img_w = 32
img_h = 32
img_d = 3
classes = 10
learning_rate = 0.01
momentum = 0.9

# commandline args
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

print("[INFO] loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

# convert labels from int to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

label_names = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

print("[INFO] compiling model...")
opt = SGD(lr=learning_rate, decay=learning_rate / epoch_num, momentum=momentum, nesterov=True)
model = MiniVGGNet.build(width=img_w, height=img_h, depth=img_d, classes=classes)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] training network...")
print(f"Train on {len(trainY)} samples, validate on {len(testY)}")
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=batch_size, epochs=epoch_num, verbose=1)

print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=batch_size)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1),
                            labels=label_names
                            ))

plot_training_loss_and_accuracy_keras(epoch_num=epoch_num,
                                      H=H,
                                      savefig_path=args["output"])
