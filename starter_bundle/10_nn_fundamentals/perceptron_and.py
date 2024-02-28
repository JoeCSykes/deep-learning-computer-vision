from nn.perceptron import Perceptron
import numpy as np

# construct or dataset

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [0], [0], [1]])

print("[INFO] training perceptron...")
p = Perceptron(N=X.shape[1], alpha=0.1)
p.fit(X, y, epochs=20)

print("[INFO] testing perceptron..")
for (x, target) in zip(X, y):
    pred = p.predict(x)
    print(f"[INFO] data={x}, ground-truth={target[0]}, pred={pred}")
