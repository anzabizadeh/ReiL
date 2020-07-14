import unittest

from rl.rlbase import RLBase

class testRLData(unittest.TestCase):
    def test_persistent(self):
        base_1 = RLBase(att_1=10, att_2=20)
        base_1.save(filename='test_persistent')
        base_2 = RLBase(att_1=100, att_2=200, persistent_attributes=['att_1', 'att_2'])
        base_2.load(filename='test_persistent')

        self.assertEqual(base_2._att_1, 100)
        self.assertEqual(base_2._att_2, 200)


if __name__ == "__main__":
    unittest.main()