from keras.layers import BatchNormalization, Conv2D, AveragePooling2D, MaxPooling2D, Activation, Dropout, Dense, \
    Flatten, Input, concatenate
from keras.models import Model
from keras import backend as K


class MiniGoogleNet:

    @staticmethod
    def conv_module(x, k, k_x, k_y, stride, chan_dim, padding="same"):
        """
        Module to apply 2D convolution layer followed by an activation layer and finally a batch normalization layer
        :param x: input layer
        :param k: num filters for conv layer to learn
        :param k_x: size of x-axis of K filters to learn
        :param k_y: size of y-axis of K filters to learn
        :param stride: stride of conv layer
        :param chan_dim: channel dimension (derived from channels last or channels first ordering)
        :param padding: type of padding to be applied to conv layer
        :return: Keras layer
        """
        # def a CONV => RELU => BN pattern
        x = Conv2D(k, (k_x, k_y), strides=stride, padding=padding)(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chan_dim)(x)

        return x

    @staticmethod
    def inception_module(x, num_k_1x1, num_k_3x3, chan_dim):
        """
        Module to generate the mini inception module which applies a 1x1 conv_module and a 3x3 conv_module in parallel
        to the input and then merges the 2 results across the channel dimension to form the output
        :param x: input layer
        :param num_k_1x1: num 1x1 conv filters to generate
        :param num_k_3x3: num 3x3 conv filters to generate
        :param chan_dim: channel dimension (derived from channels last or channels first ordering)
        :return: Keras layer
        """
        conv_1x1 = MiniGoogleNet.conv_module(x, num_k_1x1, 1, 1, (1, 1), chan_dim)
        conv_3x3 = MiniGoogleNet.conv_module(x, num_k_3x3, 3, 3, (1, 1), chan_dim)
        x = concatenate([conv_1x1, conv_3x3], axis=chan_dim)

        return x

    @staticmethod
    def downsample_module(x, k, chan_dim):
        """
        Module to decrease the output volume size. Two methods are applied in parallel and the results are then
        merged together. The first down-sampling method is a 3x3 conv layer with a stride of 2x2 and the second is a
        max pooling layer with a 3x3 window and a stride of 2x2.
        :param x: input layer
        :param k: num filters
        :param chan_dim: channel dimension (derived from channels last or channels first ordering)
        :return: Keras layer
        """
        conv_3x3 = MiniGoogleNet.conv_module(x, k, 3, 3, (2, 2), chan_dim, padding="valid")
        pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = concatenate([conv_3x3, pool], axis=chan_dim)

        return x

    @staticmethod
    def build(width, height, depth, classes):
        inputShape = (height, width, depth)
        chan_dim = -1

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chan_dim = 1

        inputs = Input(shape=inputShape)
        x = MiniGoogleNet.conv_module(inputs, 96, 3, 3, (1, 1), chan_dim)

        x = MiniGoogleNet.inception_module(x, 32, 32, chan_dim)
        x = MiniGoogleNet.inception_module(x, 32, 48, chan_dim)
        x = MiniGoogleNet.downsample_module(x, 80, chan_dim)

        x = MiniGoogleNet.inception_module(x, 112, 48, chan_dim)
        x = MiniGoogleNet.inception_module(x, 96, 64, chan_dim)
        x = MiniGoogleNet.inception_module(x, 80, 80, chan_dim)
        x = MiniGoogleNet.inception_module(x, 48, 96, chan_dim)
        x = MiniGoogleNet.downsample_module(x, 96, chan_dim)

        x = MiniGoogleNet.inception_module(x, 176, 160, chan_dim)
        x = MiniGoogleNet.inception_module(x, 176, 160, chan_dim)
        x = AveragePooling2D((7, 7))(x)
        x = Dropout(0.5)(x)

        x = Flatten()(x)
        x = Dense(classes)(x)
        x = Activation("softmax")(x)

        model = Model(inputs, x, name="minigooglenet")

        return model
    

