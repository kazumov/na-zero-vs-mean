from __future__ import annotations

"""The data set class"""

__author__ = "Ruben R. Kazumov"
__copyright__ = "Copyright 2019, Ruben R. Kazumov"
__credits__ = ["Ruben R. Kazumov"]
__license__ = "MIT"
__version__ = [3, 0, 0]
__maintainer__ = "Ruben R. Kazumov"
__email__ = "kazumov@gmail.com"
__status__ = "Production"

import os, sys, functools, math, uuid, itertools

from typing import List, Callable, Union

import logging, logging.config

import pickle

import numpy as np

from pandas import DataFrame

from sklearn.model_selection import train_test_split as trainTestSplit

from lab.target import TargetGenerator, TGAlpha

from lab.damager import Damager


class Data:
    """The data set builder

    Attributes:
    - x (ndarray): 2D list of features data.
    - xTrain (ndarray): 2D list of train set features values.
    - yTrain (ndarray): 2D list (column) of train set target class values.
    - xTest (ndarray): 2D list of train set features values.
    - yTest (ndarray): 2D list (column) of test set target class values."""

    def __init__(self):

        logging.basicConfig(
            filename="app.log",
            filemode="w",
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            level=logging.INFO,
        )

        self.logger = logging.getLogger(__class__.__name__)

    def setX(self, x: Union(List[[float]], np.ndarray)) -> Data:
        if type(x).__name__ == "ndarray":
            self.x = x

        if type(x).__name__ == "list":
            arr = np.array(x, dtype=float)
            if len(arr.shape) > 1:
                self.x = np.array(x, dtype=float)
            else:
                raise Exception("Two dimentional list expected.")
        try:
            del self.xTrain
        except AttributeError:
            pass

        try:
            del self.xTest
        except AttributeError:
            pass

        try:
            del self.yTest
        except AttributeError:
            pass

        try:
            del self.yTrain
        except AttributeError:
            pass

        return self

    def split(self, testSize: float = 0.3) -> Data:
        """Splits data set on test and training subsets

        Parameter:
        - testSize (float): Split proportion."""
        try:
            x = self.x
        except ArithmeticError:
            raise Exception(
                "The features data is empty. Set the data with method SetX()."
            )

        try:
            y = self.y
        except ArithmeticError:
            raise Exception(
                "The targets data is empty. Set the targets with methods SetY() or makeTarget()."
            )

        self.xTrain, self.xTest, self.yTrain, self.yTest = trainTestSplit(
            x, y, test_size=testSize
        )

        return self

    def makeTarget(self, targetGenerator: TargetGenerator) -> Data:
        """Creates target classes

        Parameters:
        - targetGenerator (TargetGenerator): Callable, calculates y by feature values."""
        try:
            x = self.x
        except AttributeError:
            raise Exception(
                "The features data is empty. Set the data with method SetX()."
            )

        y = np.apply_along_axis(targetGenerator, axis=1, arr=x)

        self.y = y.reshape(len(y), 1)

        self.logger.info(
            "makeTarget() calculated target classes for {0} observations. The targets histogram: {1}".format(
                x.shape, str(np.histogram(y))
            )
        )
        return self

    def save(self, path: str) -> str:
        """Saves the data set on a disk
        
        Parameters:
        - path (str): Path to directory where the data set will be stored."""
        fileName = str(uuid.uuid4()) + ".pkl"

        url = path + os.sep + fileName

        with open(url, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

        self.logger.info("save(): saved the data in file: `{0}`".format(url))

        return url

    def read(self, url: str) -> Data:
        with open(url, "rb") as f:
            d = pickle.load(f)
            try:
                self.x = d.x
            except AttributeError as e:
                self.logger.info(
                    "Data.read(): file `" + url + "` does not consist x data."
                )

            try:
                self.xTrain = d.xTrain
            except AttributeError as e:
                self.logger.info(
                    "Data.read(): file `" + url + "` does not consist xTrain data."
                )

            try:
                self.xTest = d.xTest
            except AttributeError as e:
                self.logger.info(
                    "Data.read(): file `" + url + "` does not consist xTest data."
                )

            try:
                self.yTrain = d.yTrain
            except AttributeError as e:
                self.logger.info(
                    "Data.read(): file `" + url + "` does not consist yTrain data."
                )

            try:
                self.yTest = d.yTest
            except AttributeError as e:
                self.logger.info(
                    "Data.read(): file `" + url + "` does not consist yTest data."
                )

        self.logger.info("Data.read(): got data from `" + url + "`")

        return self

    def damage(self, damager: Damager, quantity: Union(int, float)) -> Data:
        try:
            x = self.x
        except AttributeError:
            raise Exception(
                "The features data is empty. Set the data with method SetX()."
            )

        self.x = damager(x, quantity)

        self.logger.info("Data.damage(): damaged by " + damager.__class__.__name__)

        return self

    def fix(self, fixer: Fixer) -> Data:
        try:
            x = self.x
        except AttributeError:
            raise Exception(
                "The features data is empty. Set the data with method SetX()."
            )

        self.x = fixer(x)

        self.logger.info("Data.fix(): fixed by " + fixer.__class__.__name__)

        return self

    def info(self, expanded: bool = True) -> None:
        """Outputs the short dumps of the train and test data"""
        self.logger.info("Content:")

        for attr in ["x", "xTrain", "yTrain", "xTest", "yTest"]:
            try:
                source = getattr(self, attr)

                self.logger.info(
                    "The `{0}` attribute shape is {1}.".format(attr, source.shape)
                )

                if expanded:
                    self.logger.info(
                        "The `{0}` attribute content is:\n{1}".format(attr, source)
                    )

            except AttributeError:
                self.logger.info("The `{0}` attribute is empty.")

        return None


class DataRandom(Data):
    def __init__(self, features: int = 10, observations: int = 10000):
        """Creates splitted data set 
        Parameters:
        - observations (int): Number of observations.
        - features (int): Number of features."""

        super().__init__()

        self.x = np.random.rand(observations, features)

        self.logger.info(
            "data set [{0}, {1}] created, x generated.".format(observations, features)
        )

        self.logger = logging.getLogger(__class__.__name__)
