import sys
sys.path.insert(
    0,
    '/'
)
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from preprocessors.simple_preprocessor import SimplePreprocessor
from helper_funcs.simple_dataset_loader import SimpleDatasetLoader
from imutils import paths
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
args = vars(ap.parse_args())

# grab the list of image paths
print("[INFO] loading images...")
image_paths = list(paths.list_images(args["dataset"]))

sp = SimplePreprocessor(32, 32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(image_paths, verbose=500)
data = data.reshape((data.shape[0], 3072))

# encode labels as integers
le = LabelEncoder()
labels = le.fit_transform(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=5)

# loop over our set of regularizers
for r in (None, "l1", "l2"):
    # trainSGD classifier using Softmax and specified regularization for 10 epochs
    print(f"[INFO] training model with {r} penalty")
    model = SGDClassifier(loss="log_loss", penalty=r, max_iter=10, learning_rate="constant", eta0=0.01, random_state=None)
    model.fit(trainX, trainY)

    # evaluate classifier
    acc = model.score(testX, testY)
    print(F"[INFO] '{r}' penalty accuracy: {acc * 100: .2f}")
