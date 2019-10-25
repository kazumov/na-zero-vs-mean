import unittest
import os, sys
from copy import deepcopy

import numpy as np

from data import Data, DataRandom
from target_generator import TGAlpha
from damager import Damager, DMGNA, DMGNoiseFeatures, DMGNoiseValues


class DamagersClassTests(unittest.TestCase):
    def setUp(self):
        self.data = Data()
        x = np.arange(25, dtype=float).reshape((5, 5))
        self.data.setX(x)

    def test_prevention_of_the_abstract_damager_instatination(self):
        with self.assertRaises(Exception, msg="abstract class instantiated"):
            dmg = Damager()

    def test_instatination_of_the_DMGNA_damager(self):
        dmg = DMGNA()
        self.assertIsInstance(dmg, DMGNA, "it has wrong type")

    def test_instatination_of_the_DMGNoiseFeatures_damager(self):
        dmg = DMGNoiseFeatures()
        self.assertIsInstance(dmg, DMGNoiseFeatures, "it has wrong type")

    def test_instatination_of_the_DMGNoiseValues_damager(self):
        dmg = DMGNoiseValues()
        self.assertIsInstance(dmg, DMGNoiseValues, "it has wrong type")

    def test_data_damaged_by_DMGNA_damager(self):
        d = deepcopy(self.data)
        dmg = DMGNA()
        d.damage(dmg, 0.5)
        self.assertNotEqual(
            hash("{}".format(d.x)),
            hash("{}".format(self.data.x)),
            "damager did not change the data",
        )

    def test_data_damaged_by_DMGNoiseFeatures_damager(self):
        d = deepcopy(self.data)
        dmg = DMGNoiseFeatures()
        noiseFeaturesCount = 10  # testing constant
        d.damage(dmg, noiseFeaturesCount)
        self.assertEqual(
            d.x.shape[1] - noiseFeaturesCount,
            self.data.x.shape[1],
            "damager created wrong number of features",
        )

    @unittest.skip  # not implemented
    def test_data_damaged_by_DMGNoiseValues_damager(self):
        d = deepcopy(self.data)
        dmg = DMGNoiseValues()
        d.damage(dmg, 0.5)
        self.assertNotEqual(
            hash("{}".format(d.x)),
            hash("{}".format(self.data.x)),
            "damager did not change the data",
        )


if __name__ == "__main__":
    unittest.main()
