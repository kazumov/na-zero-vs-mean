from __future__ import annotations

"""The data set fixer class"""

__author__ = "Ruben R. Kazumov"
__copyright__ = "Copyright 2019, Ruben R. Kazumov"
__credits__ = ["Ruben R. Kazumov"]
__license__ = "MIT"
__version__ = [3, 0, 0]
__maintainer__ = "Ruben R. Kazumov"
__email__ = "kazumov@gmail.com"
__status__ = "Production"

# import os, sys, functools, math, uuid, itertools

from abc import ABC, abstractmethod

import numpy as np


class Fixer(ABC):
    """The data fixer/patcher"""

    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, data: ndarray) -> ndarray:
        pass


class FXZero(Fixer):
    """Fixes NaN values by 0.0 replacement"""

    def __call__(self, data: ndarray) -> ndarray:
        """Replaces NaN with 0.0
        
        Parameters:
        - data (ndarray): The data with NaN elements."""
        data[np.isnan(data)] = 0.0
        # print(
        #     Icon.info,
        #     __class__.__name__,
        #     "replaced NaN values in",
        #     data.__class__.__name__,
        #     "with 0.0",
        # )
        return data


class FXMean(Fixer):
    def __call__(self, data: ndarray) -> ndarray:
        """Replaces NaN with the feature mean value
        
        Parameters:
        - data (ndarray): The data with NaN elements."""

        colMean = np.nanmean(data, axis=0)

        isNanIdx = np.where(np.isnan(data))

        data[isNanIdx] = np.take(colMean, isNanIdx[1])

        # print(
        #     Icon.info,
        #     __class__.__name__,
        #     "replaced each NaN value in",
        #     data.__class__.__name__,
        #     "with the feature mean",
        # )
        return data
