from keras import Sequential
from keras_cv.layers import RandomRotation, RandomTranslation, RandomFlip
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from nn.conv.minivggnet import MiniVGGNet
from keras.optimizers import SGD
from keras.datasets import cifar10
from helper_funcs.plot_funcs import plot_training_loss_and_accuracy_keras
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-r", "--report", required=True, help="path to output reports directory")
ap.add_argument("-m", "--model", required=True, help="path to output models directory")
ap.add_argument("-n", "--num-models", type=int, default=5, help="# of models to train")
args = vars(ap.parse_args())

# normally train 5-10 CNNs in an ensemble due to computing expense

((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

label_names = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

data_augmentation = Sequential([
    RandomRotation(0.1),
    RandomTranslation(height_factor=0.1, width_factor=0.1),
    RandomFlip("horizontal"),
])

LEARNING_RATE = 0.01
NUM_EPOCHS = 40
BATCH_SIZE = 64

for i in range(0, args["num_models"]):
    print(f"[INFO] training model {i + 1}/{args['num_models']}")
    opt = SGD(learning_rate=LEARNING_RATE, weight_decay=LEARNING_RATE / NUM_EPOCHS, momentum=0.9, nesterov=True)
    model = Sequential([data_augmentation,
                        MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
                        ])
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=64, epochs=40, verbose=1)
    model_path = [args["model"], f"model_{i}.model"]
    model.save(os.path.sep.join(model_path))

    predictions = model.predict(testX, batch_size=BATCH_SIZE)
    report = classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=label_names)

    report_path = [args["report"], f"model_{i}_report.txt"]
    with open(os.path.sep.join(report_path), "w") as f:
        f.write(report)

    graph_path = [args["report"], f"model_{i}_graph.txt"]
    plot_training_loss_and_accuracy_keras(epoch_num=NUM_EPOCHS,
                                          H=H,
                                          savefig_path=os.path.sep.join(graph_path)
                                          )
