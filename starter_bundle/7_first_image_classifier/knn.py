import sys
sys.path.insert(
    0,
    '/'
)
# from simple_dataset_loader import JoeSimpleDatasetLoader
# from simple_preprocessor import JoeSimplePreprocessor
from helper_funcs.simple_dataset_loader import SimpleDatasetLoader
from preprocessors.simple_preprocessor import SimplePreprocessor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import argparse

# commandline arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")
ap.add_argument("-k", "--neighbors", type=int, default=1,
                help="# of nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type=int, default=-1,
                help="# of jobs for k-NN distance (-1 uses all available cores")
args = vars(ap.parse_args())

# step 1: load data
print("[INFO] loading images...")
image_paths = list(paths.list_images(args["dataset"]))

sp = SimplePreprocessor(32, 32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(image_paths, verbose=500)
data = data.reshape((data.shape[0], 32*32*3))

print(f"[INFO] features matrix: {data.nbytes / (1024 * 1000.0): .1f}MB")

# step 2: build train and test sets

# encode labels as integers
le = LabelEncoder()
labels = le.fit_transform(labels)

print(f"DT of data = {data.shape}, DT of labels = {labels.shape}")
# split data into 75% train and 25% test
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

# step 3: train and evaluate

print("[INFO] evaluating k-NN classifier...")
model = KNeighborsClassifier(n_neighbors=args["neighbors"], n_jobs=args["jobs"])
model.fit(trainX, trainY)
print(classification_report(testY, model.predict(testX), target_names=le.classes_))
