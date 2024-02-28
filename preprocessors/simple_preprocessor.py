import cv2


class SimplePreprocessor:
    def __init__(self, width: int, height: int, inter=cv2.INTER_AREA):
        # store target img width, height and interpolation
        # method used when resizing
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        # resize img to fixed size, ignore aspect ratio
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)

