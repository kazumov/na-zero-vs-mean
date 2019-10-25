"""The plot class"""

__author__ = "Ruben R. Kazumov"
__copyright__ = "Copyright 2019, Ruben R. Kazumov"
__credits__ = ["Ruben R. Kazumov"]
__license__ = "MIT"
__version__ = [3, 0, 0]
__maintainer__ = "Ruben R. Kazumov"
__email__ = "kazumov@gmail.com"
__status__ = "Production"

import os, sys

from uuid import uuid4

from abc import ABC

from keras.callbacks.callbacks import History

import matplotlib.pyplot as plt

import seaborn


class Plot(ABC):
    def __init__(self, path: str = "plots", fileName: str = ""):

        dirNameDefault = "plots"

        if path == "":
            self.path = os.getcwd() + os.path.sep + dirNameDefault
        else:
            self.path = os.getcwd() + os.path.sep + path

        # file name
        if fileName == "":
            # random
            self.url = self.path + os.path.sep + str(uuid4()) + ".png"
        else:
            # concrete name
            self.url = self.path + os.path.sep + fileName
            if os.path.exists(self.url):
                # remove old file
                os.remove(self.url)


class FittingAccuracy(Plot):
    """The fitting accuracy plot"""

    def __init__(self, dataSignature: str = "", path: str = "plots"):
        self.title = dataSignature
        # file name is alwais random
        super().__init__(path=path, fileName="")

    def plot(
        self,
        history: History,
        trainingDataSelector: str = "accuracy",
        testDataSelector: str = "val_accuracy",
    ) -> str:
        """Creates a fitting accuracy plot.
        
        Parameter:
        - history (History): A fitting history.
        - trainingDataSelector (str): A histroy data attribute name for the training data.
        - testDataSelector (str): A histroy data attribute name fro the testing data.

        Returns (str): URL to plot file."""

        plt.style.use("seaborn")

        seaborn.set_style("whitegrid")

        plt.figure()

        axes = plt.gca()

        axes.set_ylim([0.4, 1.0])

        plt.plot(history.history[trainingDataSelector])

        plt.plot(history.history[testDataSelector])

        plt.title(f"Accuracy plot for the data: ({self.title})")

        plt.ylabel("Accuracy")

        plt.xlabel("Epoch")

        plt.legend(["Train", "Test"], loc="upper left")

        plt.savefig(self.url)

        return self.url


class FittingLossFunction(Plot):
    """The loss function plot"""

    def __init__(self, dataSignature: str = "", path: str = "plots"):
        self.title = dataSignature
        # file name is alwais random
        super().__init__(path=path, fileName="")

    def plot(
        self,
        history: History,
        trainingDataSelector: str = "loss",
        testDataSelector: str = "val_loss",
    ) -> None:
        """Plots a history of the loss function value change

        Parameter:
        - history (History): A fitting history.
        - trainingDataSelector (str): A histroy data attribute name for the training data.
        - testDataSelector (str): A histroy data attribute name fro the testing data.
        
        Returns (str): URL to plot file."""

        plt.style.use("seaborn")

        seaborn.set_style("whitegrid")

        plt.figure()

        plt.plot(history.history[trainingDataSelector])

        plt.plot(history.history[testDataSelector])

        plt.title(f"Loss function plot for the data: ({self.title})")

        plt.ylabel("Loss")

        plt.xlabel("Epoch")

        plt.legend(["Train", "Test"], loc="upper left")

        plt.savefig(self.url)

        return self.url
