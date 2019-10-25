from typing import Union

import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout


class ModelDDDD:
    """Sequential model with layers: Dense->Dropout->Dense->Dense"""

    def __init__(self, featuresCount: int):
        """Parameters:

        - featuresCount (int): A number of features of the data set.
        """
        super().__init__()

        model = Sequential()

        dense1 = int(featuresCount * 0.75)
        dense2 = int(featuresCount * 0.5)

        model.add(Dense(dense1, activation="relu", input_shape=(featuresCount,)))
        model.add(Dropout(0.1))
        model.add(Dense(dense2, activation="relu"))
        model.add(Dense(1, activation="sigmoid"))
        model.compile(
            optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"]
        )
        self.model = model

    def __call__(self) -> Union[Sequential]:
        """A factory method"""
        return self.model

    def summary(self):
        self.model.summary()
