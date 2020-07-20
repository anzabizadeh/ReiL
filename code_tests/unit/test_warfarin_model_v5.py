import unittest

from rl.subjects import WarfarinModel_v5

class testRLData(unittest.TestCase):
    def test_object_creation(self):
        w = WarfarinModel_v5()
        print(w)
        self.assertEqual(w.is_terminated, False)

