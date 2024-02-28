from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
from keras.regularizers import l2
from keras import backend as K


class AlexNet:
    @staticmethod
    def build(width, height, depth, classes, reg=0.0002):
        model = Sequential()
        input_shape = (height, width, depth)
        chan_dim = -1

        if K.image_data_format() == "channels_first":
            input_shape = (depth, height, width)
            chan_dim = 1

        model.add(Conv2D(96, (11, 11), strides=(4, 4), input_shape=input_shape,
                         padding="same", kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(256, (5, 5), padding="same", kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(0.25))

        filters = [384, 384, 256]
        for filter_size in filters:
            model.add(Conv2D(filter_size, (3, 3), padding="same", kernel_regularizer=l2(reg)))
            model.add(Activation("relu"))
            model.add(BatchNormalization(axis=chan_dim))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(4096, kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(4096, kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # FC dropout values are standard for CNNs

        model.add(Dense(classes, kernel_regularizer=l2(reg)))
        model.add(Activation("softmax"))

        return model

