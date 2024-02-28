from sklearn.preprocessing import LabelBinarizer
from nn.conv.minigooglenet import MiniGoogleNet
from callbacks.training_monitor import TrainingMonitor
from nn.conv.data_aug import DataAugmentor
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from keras.datasets import cifar10
import numpy as np
import argparse
import os

NUM_EPOCHS = 70
INIT_lR = 1e-3

def poly_decay(epoch):
    max_epochs = NUM_EPOCHS
    base_lr = INIT_lR
    # power of 1 gives linear decay, increasing power increases initial and lessens final rate of decay
    power = 1.0

    alpha = base_lr * (1 - (epoch / float(max_epochs))) ** power

    return alpha

ap =  argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="path to output model")
ap.add_argument("-o", "--output", required=True, help="path to output directory (logs, plots, etc.)")
args = vars(ap.parse_args())

print("[INFO loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float")
testX = testX.astype("float")

# apply mean subtraction
mean = np.mean(trainX, axis=0)
trainX -= mean
testX -= mean

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# construct callbacks
fig_path = os.path.sep.join([args["output"], "{}.png".format(os.getpid())])
json_path = os.path.sep.join([args["output"], "{}.json".format(os.getpid())])
callbacks = [TrainingMonitor(fig_path, json_path=json_path), LearningRateScheduler(poly_decay)]

print("[INFO] compiling model...")
opt = SGD(lr=INIT_lR, momentum=0.9)
model = MiniGoogleNet.build(width=32, height=32, depth=3, classes=10)
model = DataAugmentor.build(model)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] training network...")
model.fit(trainX, trainY, batch_size=64, validation_data=(testX, testY),
          epochs=NUM_EPOCHS, callbacks=callbacks, verbose=1)
print("[INFO] serializing network...")
model.save(args["model"])
