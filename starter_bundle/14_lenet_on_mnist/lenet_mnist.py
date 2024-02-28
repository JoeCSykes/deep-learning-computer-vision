from nn.conv.lenet import LeNet
from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
from keras import backend as K
from helper_funcs.plot_funcs import plot_training_loss_and_accuracy_keras

print("[INFO] accessing MNIST...")
dataset = datasets.fetch_openml("mnist_784", parser='auto')
data = dataset.data

# params
img_h = 28
img_w = 28
img_d = 1
num_classes = 10
learning_rate = 0.01
batch_size = 128
epochs = 20

print(data.shape, type(data))
if K.image_data_format == "channels_first":
    data = data.values.reshape(data.shape[0], img_d, img_w, img_h)
else:
    data = data.values.reshape(data.shape[0], img_w, img_h, img_d)

# split data
(trainX, testX, trainY, testY) = train_test_split(data / 255.0,
                                                  dataset.target.astype("int"),
                                                  test_size=0.25,
                                                  random_state=42,
                                                  )
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

# initialize optimizer & model
print("[INFO] compiling model...")
opt = SGD(lr=learning_rate)
model = LeNet.build(width=img_w, height=img_h, depth=img_d, classes=num_classes)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# train network
print("[INFO] training model...")
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=batch_size, epochs=epochs, verbose=1)

# evaluating the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=batch_size)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=[str(x) for x in lb.classes_]
                            ))

plot_training_loss_and_accuracy_keras(epoch_num=epochs,
                                      H=H,
                                      )
