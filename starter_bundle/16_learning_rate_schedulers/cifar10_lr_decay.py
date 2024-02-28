import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from nn.conv.minivggnet import MiniVGGNet
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from keras.datasets import cifar10
from helper_funcs.plot_funcs import plot_training_loss_and_accuracy_keras
import argparse


def step_decay(epoch):
    initAlpha = 0.01
    factor = 0.25
    drop_every = 5

    # compute learning rate for current epoch
    alpha = initAlpha * (factor ** np.floor((1 + epoch) / drop_every))

    return float(alpha)


# parameters
learning_rate = 0.01  # redundant as learning rate set by step decay
momentum = 0.9
img_w = 32
img_h = 32
img_d = 3
class_num = 10
batch_size = 64
epoch_num = 64

# command line args
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="path to the output loss/accuracy plot")
args = vars(ap.parse_args())

# load data
print("[INFO] loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

# convert labels
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

label_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# init optimizer and model
callbacks = [LearningRateScheduler(step_decay)]
opt = SGD(learning_rate=learning_rate, momentum=momentum, nesterov=True)
model = MiniVGGNet.build(width=img_w, height=img_h, depth=img_d, classes=class_num)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# train network
print("[INFO] training network...")
print(f"training set size = {len(trainY)}, validation set size = {len(testY)}")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
              batch_size=batch_size, epochs=epoch_num, callbacks=callbacks, verbose=1)

print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=batch_size)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=label_names))

plot_training_loss_and_accuracy_keras(epoch_num=epoch_num, H=H)
