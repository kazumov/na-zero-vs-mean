import unittest
import os, sys, copy

import numpy as np

from data import Data, DataRandom
from target_generator import TGAlpha
from damager import DMGNA, DMGNoiseFeatures, DMGNoiseValues
from fixer import Fixer, FXZero, FXMean


class FixersClassTests(unittest.TestCase):
    def setUp(self):
        self.data = Data()
        self.dmgNA = DMGNA()
        self.x = np.arange(25, dtype=float).reshape((5, 5))
        self.data.setX(self.x).damage(self.dmgNA, 0.5)

    def test_meanZeroInit(self):
        z = FXZero()
        self.assertIsInstance(z, FXZero, "instantiation failed")

    def test_meanFixerInit(self):
        m = FXMean()
        self.assertIsInstance(m, FXMean, "instantiation failed")

    def test_zeroFix(self):
        d = copy.deepcopy(self.data)
        d.info()  # before fix
        z = FXZero()
        d.fix(z)
        d.info()  # after fix

    def test_meanFix(self):
        d = copy.deepcopy(self.data)
        d.info()  # before fix
        m = FXMean()
        d.fix(m)
        d.info()  # after fix
