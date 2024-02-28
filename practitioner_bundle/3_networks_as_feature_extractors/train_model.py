from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import argparse
import pickle
import h5py

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--db", required=True, help="path to HDF5 database")
ap.add_argument("-m", "--model", required=True, help="path to output model")
ap.add_argument("-j", "--jobs", type=int, default=-1,
                help="# of jobs to run when tuning hyper parameters, -1 means use all processors")
args = vars(ap.parse_args())

# previously randomly shuffled data before saving to hdf5 file
with h5py.File(args["db"], "r") as db:
    idx = int(db["labels"].shape[0] * 0.75)

    print("[INFO] Tuning hyperparameters...")
    params = {"C": [1.0, 10.0, 100.0, 1000.0, 10000.0]}
    model = GridSearchCV(LogisticRegression(max_iter=150), params, cv=3, n_jobs=args["jobs"], verbose=10)
    model.fit(db["features"][:idx], db["labels"][:idx])
    print(f"[INFO] best hyperparameters: {model.best_params_}")

    print("[INFO] evaluating...")
    preds = model.predict(db["features"][idx:])
    label_names = [label.decode() for label in db["label_names"]]
    print(classification_report(db["labels"][idx:], preds, target_names=label_names))

    print("[INFO] saving model...")
    with open(args["model"], "wb") as file:
        file.write(pickle.dumps(model.best_estimator_))
