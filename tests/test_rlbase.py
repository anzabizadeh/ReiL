import unittest

from reil.reilbase import ReilBase


class testReilData(unittest.TestCase):
    def test_persistent(self):
        base_1 = ReilBase(name='test_persistent', version=1.0)
        base_1.save(filename='test_persistent')
        base_2 = ReilBase(name='base_2', version=0.6,
                        persistent_attributes=['name', 'version'])
        base_2.load(filename='test_persistent')

        self.assertEqual(base_2._name, 'base_2')
        self.assertEqual(base_2._version, 0.6)

    def test_inheritance(self):
        class interited(ReilBase):
            def __init__(self, myarg: str='hello', **kwargs) -> None:
                self._myarg = myarg
                super().__init__(**kwargs)

        test = interited(another_arg=1, name='inherited')

        self.assertEqual(test._name, 'inherited')
        self.assertIn('_another_arg', test.__dict__)



if __name__ == "__main__":
    unittest.main()
