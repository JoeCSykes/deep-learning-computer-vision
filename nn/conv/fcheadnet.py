from keras.layers import Dropout, Flatten, Dense


class FCHeadNet:
    @staticmethod
    def build(baseModel, classes: int, D: int):
        headModel = baseModel.output
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(D, activation="relu")(headModel)
        headModel = Dropout(0.5)(headModel)

        headModel = Dense(classes, activation="softmax")(headModel)

        return headModel
