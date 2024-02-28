from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception  # TensorFlow ONLY
from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.utils.image_utils import img_to_array
from keras.utils.image_utils import load_img
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-m", "--model", type=str, default="vgg16", help="Name of pretrained network to use")
args = vars(ap.parse_args())

models = {
    "vgg16": VGG16,
    "vgg19": VGG19,
    "inception": InceptionV3,
    "xception": Xception,
    "resnet": ResNet50
}

if args["model"] not in models.keys():
    raise AssertionError(f"The --model command line argument is invalid. "
                         f"It should be one of the following {models.keys()}")

# VGG16, VGG19 and ResNet accept 224x224 input image shapes
# InceptionV3 and Xception accept 299x299 input image shapes
if args["model"] in ("inception", "xception"):
    input_shape = (299, 299)
    preprocess = preprocess_input
else:
    input_shape = (224, 224)
    preprocess = imagenet_utils.preprocess_input

print(f"[INFO] loading {args['model']}")
Network = models[args["model"]]
model = Network(weights="imagenet")

print("[INFO] loading and pre-processing image...")
image = load_img(args["image"], target_size=input_shape)
image = img_to_array(image)

# our input image is now represented as a NumPy array of shape
# (inputShape[0], inputShape[1], 3) however we need to expand the
# dimension by making the shape (1, inputShape[0], inputShape[1], 3)
# so we can pass it through the network
image = np.expand_dims(image, axis=0)
image = preprocess(image)

print(f"[INFO] classifying image with \'{args['model']}\'")
preds = model.predict(image)
P = imagenet_utils.decode_predictions(preds)

for (i, (imagenetID, label, prob)) in enumerate(P[0]):
    print(f"{i + 1}. {label}: {prob * 100:.2f}")

orig = cv2.imread(args["image"])
(imagenetID, label, prob) = P[0][0]
cv2.putText(orig, f"Label: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
cv2.imshow("Classification", orig)
cv2.waitKey(0)
