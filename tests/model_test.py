import unittest

import keras

from keras import Sequential

from lab.model import ModelDDDD


class ModelDDDDClassTests(unittest.TestCase):
    def setUp(self):
        self.model = ModelDDDD(40)

    def test_model_instantiation(self):
        self.assertIsInstance(self.model, ModelDDDD, "wrong type of object instance")

    def test_model_content_instance_type(self):
        self.assertIsInstance(
            self.model.model, keras.engine.sequential.Sequential, "wrong type of model"
        )

    def test_model_comtiling(self):
        s = self.model().summary()
        self.assertGreater(len(s), 300, "model is not a valid model")
        self.assertRegexpMatches(
            s, "Total params", "the missing term `Total params` in the model summary"
        )
