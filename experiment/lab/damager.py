#!/usr/bin/env python3
from __future__ import annotations

"""The data set damager class"""

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

from abc import ABC, abstractmethod

import numpy as np


class Damager(ABC):
    """The data damager"""

    def __init__(self):
        logging.basicConfig(
            filename="app.log",
            filemode="w",
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            level=logging.INFO,
        )

        self.logger = logging.getLogger(__class__.__name__)

    @abstractmethod
    def __call__(self, data: ndarray, quantity: Union(float, int)) -> ndarray:
        pass


class DMGNA(Damager):
    """Damages the data array by replacing cells in the training data by NA"""

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__class__.__name__)

    def __call__(self, data: ndarray, quantity: Union(float, int)) -> ndarray:
        """The damager callable function
        
        Parameters:
        - data (ndarray): Data to damage.
        - amount (float): Proportion of the damaged to undamaged elements."""
        numberOfNaN = int(data.shape[0] * data.shape[1] * quantity)
        data.ravel()[np.random.choice(data.size, numberOfNaN, replace=False)] = np.nan
        self.logger.info(
            "damaged {0} elements of the {1}".format(
                numberOfNaN, data.__class__.__name__
            )
        )
        return data


class DMGNoiseFeatures(Damager):
    """Damages the data by adding noise features"""

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__class__.__name__)

    def __call__(self, data: ndarray, quantity: int) -> ndarray:
        """The damager callable function
    
        Properties:
        - data (ndarray): Data.
        - quantity (int): Number of features with a noise content."""
        noise = np.random.rand(data.shape[0], quantity)
        data = np.concatenate((data, noise), axis=1)
        self.logger.info(
            "{1} damaged by appending {0} noise features".format(
                quantity, data.__class__.__name__
            )
        )
        return data


class DMGNoiseValues(Damager):
    """Damages the data by replacing the real values by noise"""

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__class__.__name__)

    def __call__(self, data: ndarray, quantity: float) -> ndarray:
        """The damager callable function
        
        Properties:
        - data (ndarray): Data.
        - quantity (float): Proportion damaged/undamaged of elements."""
        numberOfNoiseElements = int(data.shape[0] * data.shape[1] * quantity)

        raise Exception("The metod `DMGNoiseValues.__call__()` implemented")

        self.logger.info(
            "damaged {0} elements of the {1}".format(
                numberOfNoiseElements, data.__class__.__name__
            )
        )
        return data
