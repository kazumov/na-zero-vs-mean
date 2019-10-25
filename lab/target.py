from __future__ import annotations

# The data set builder class

__author__ = "Ruben R. Kazumov"
__copyright__ = "Copyright 2019, Ruben R. Kazumov"
__credits__ = ["Ruben R. Kazumov"]
__license__ = "MIT"
__version__ = [3, 0, 0]
__maintainer__ = "Ruben R. Kazumov"
__email__ = "kazumov@gmail.com"
__status__ = "Production"

import os, sys, functools, math

from typing import List

import logging, logging.config

from abc import ABC, abstractmethod

from enum import Enum

import numpy as np


class TargetGenerator(ABC):
    def __init__(self):
        logging.basicConfig(
            filename="app.log",
            filemode="w",
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            level=logging.INFO,
        )

        self.logger = logging.getLogger(__class__.__name__)

    @abstractmethod
    def __call__(self, x: List[float]) -> float:
        raise NotImplementedError(
            "The method __call__() must be overriden in child class!"
        )
        return None

    @abstractmethod
    def LaTeX(self) -> str:
        raise NotImplementedError(
            "The method LaTeX() must be overriden in child class!"
        )
        return None


class TGAlpha(TargetGenerator):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__class__.__name__)
        self.logger.info("instantiated")

    def __call__(self, x: List[float]) -> float:
        """Generates target value
        
        Parameters:
        - x (List[float]): The feature values. 

        Returns: (float) target value.
        """
        assert len(x) > 0, "no features"

        wave = list(map(lambda f: math.sin(f), x))  # sin of each feature

        tops = list(map(lambda f: (f > 0.5) * 1.0, wave))  # is bigger than > 0.5

        accumulate = functools.reduce(lambda f1, f2: f1 + f2, tops)  # sum of tops

        avg = math.sin(0.5) * len(x)

        target = (accumulate > avg) * 1.0  # step

        return target

    def LaTeX(self) -> str:
        return r"$t=((\sum_{i}{(\sin{f_i}>0.5)\times1.0})>\mu_f)\times1.0$"
