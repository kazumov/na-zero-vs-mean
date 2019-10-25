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

import numpy as np

import keras

from keras import Sequential

from keras.callbacks.callbacks import History

from lab.data import DataRandom

from lab.model import ModelDDDD


class Fitter:
    """fits model to data"""

    def __init__(self):
        logging.basicConfig(
            filename="app.log",
            filemode="w",
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            level=logging.INFO,
        )

        self.logger = logging.getLogger(__class__.__name__)

    def fit(self, data: Data, batchSize: int = 512, epochs: int = 500) -> History:
        """ Fits model to data
        
        Parameters:
        - model (Model): The model to fit (keras.Sequential)
        - batchSize (int): A bath size for fitting.
        - epochs (int): A number of fitting epochs. 

        Returns: (History): The fitting history object."""

        m = ModelDDDD(featuresCount=data.xTrain.shape[1])()

        history = m.fit(
            data.xTrain,
            data.yTrain,
            batch_size=batchSize,
            epochs=epochs,
            verbose=1,
            validation_data=(data.xTest, data.yTest),
        )

        return history
