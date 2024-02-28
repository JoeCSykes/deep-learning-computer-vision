import numpy as np
import cv2

labels = ["dog", "cat", "panda"]
np.random.seed(1)

# initialize weight and bias vector
W = np.random.randn(3, 3072)
b = np.random.randn(3)

# get and resize image
orig = cv2.imread("./datasets/beagle.png")
image = cv2.resize(orig, (32, 32)).flatten()

# compute output scores
scores = W.dot(image) + b

for (label, score) in zip(labels, scores):
    print(f"[INFO] {label}: {score:.2f}")

cv2.putText(orig, f"Label: {labels[np.argmax(scores)]}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

cv2.imshow("Image", orig)
cv2.waitKey(0)
