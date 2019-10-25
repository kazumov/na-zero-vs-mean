import unittest
import os, sys
from data import Data, DataRandom
from target_generator import TGAlpha
from numpy import ndarray


class DataClassTests(unittest.TestCase):
    def setUp(self):
        self.data = Data()
        self.dataStoragePath = "data"
        # return super().setUp()

    def test_instantiation(self):
        self.assertIsInstance(self.data, Data, "it has wrong type")

    def test_setArrayOfIntegersAsX(self):
        self.data.setX([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
        self.assertIsInstance(self.data, Data, "does not accept array of integers")

    def test_typeConvertionForX(self):
        self.data.setX([[1, 2, 3], [1, 2, 3]])
        self.assertIsInstance(
            self.data.x[1, 1],
            float,
            "the type of the data element is different than `float`",
        )

    def test_stringInArrayAsX(self):
        with self.assertRaises(ValueError):
            self.data.setX([[1, "a", 3], [1, 2, 3]])

    def test_oneDimentionalArrayAsX(self):
        with self.assertRaises(Exception):
            self.data.setX([1, 2, 3, 4, 5, 6])

    def test_storeInFile(self):
        url = self.data.save(self.dataStoragePath)
        self.assertGreater(len(url), 1, "url to file empty")
        size = os.stat(url).st_size
        self.assertGreater(size, 1, "file empty")

    def test_readFromFile(self):
        x = [[100.0, 200.0, 300.0], [1.0, 2.0, 3.0]]
        self.data.setX(x)
        url = self.data.save(self.dataStoragePath)

        d = Data()
        d.read(url)
        self.assertListEqual(
            list(self.data.x.flatten()),
            list(d.x.flatten()),
            "the data is damaged during store/read operation",
        )

    def test_makeTarget(self):
        self.data.setX([[1, 2, 3], [2, 2, 4], [3, 4, 2]])
        tg = TGAlpha()
        self.data.makeTarget(tg)
        self.assertIsInstance(self.data.y, ndarray, "target is not an array")

    def test_targetValues(self):
        self.data.setX([[1, 2, 3], [2, 2, 4], [3, 4, 2]])
        tg = TGAlpha()
        self.data.makeTarget(tg)
        self.assertListEqual(
            list(set(self.data.y.flatten())),
            [0.0, 1.0],
            "target class not in set {0., 1.}",
        )


class RandomDataClassTests(unittest.TestCase):
    def setUp(self):
        self.data = DataRandom()

    def test_instantiationWithDefaults(self):
        self.assertEqual(self.data.x.shape[0], 10000, "wrong observations number")
        self.assertEqual(self.data.x.shape[1], 10, "wrong features number")


if __name__ == "__main__":
    unittest.main()
