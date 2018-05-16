
import unittest
from dbc.classifier import Classifier


class TestClassifier(unittest.TestCase):


    def test_interact(self):
        cls = Classifier()
        result = cls.interact('/home/stas/dev/udacity/ml/machine-learning/projects/dog-project/images/Welsh_springer_spaniel_08203.jpg')
        print result
        self.assertIsNotNone(result)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()