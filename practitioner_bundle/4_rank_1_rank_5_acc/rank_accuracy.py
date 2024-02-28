from helper_funcs.ranked import rank5_accuracy
import argparse
import pickle
import h5py

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--db", required=True, help="path to HDF% database")
ap.add_argument("-m", "--model", required=True, help="path to pre-trained model")
args = vars(ap.parse_args())

print("[INFO] loading pre-trained model...")
model = pickle.loads(open(args["model"], "rb").read())

with h5py.File(args["db"], "r") as db:
    idx = int(db["labels"].shape[0] * 0.75)

    print("[INFO] predicting...")
    preds = model.predict_proba(db["features"][idx:])
    (rank_1, rank_5) = rank5_accuracy(preds, db["labels"][idx:])

    print(f"[INFO] rank-1: {(rank_1 * 100)}")
    print(f"[INFO] rank-5: {(rank_5 * 100)}")
