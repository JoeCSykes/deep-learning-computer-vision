from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Activation, Flatten, Dense
from keras import backend as K


class ShallowNet:
    @staticmethod
    def build(width, height, depth, classes):
        # init model along with inout shape to be "channels last"
        model = Sequential()

        if K.image_data_format == "channels first":
            input_shape = (depth, height, width)
        else:
            input_shape = (height, width, depth)

        model.add(
            Conv2D(32, (3, 3), padding="same", input_shape=input_shape)
        )
        model.add(Activation("relu"))

        # final layer
        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model
