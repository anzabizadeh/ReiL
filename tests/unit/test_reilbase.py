from typing import Any
import unittest

from reil.reilbase import ReilBase


class testReilBase(unittest.TestCase):
    # def test_persistent(self):
    #     base_1 = ReilBase(name='test_persistent', save_zipped=True)
    #     base_1.save(filename='test_persistent')
    #     base_2 = ReilBase(name='base_2', save_zipped=True,
    #                       persistent_attributes=['name', 'version'])
    #     base_2.load(filename='test_persistent')

    #     self.assertEqual(base_2._name, 'base_2')
    #     self.assertEqual(base_2._version, 0.6)  # type: ignore

    def test_inheritance(self):
        class interited(ReilBase):
            def __init__(self, myarg: str = 'hello', **kwargs: Any) -> None:
                self._myarg = myarg
                super().__init__(**kwargs)

        test = interited(myarg='1', name='inherited')

        self.assertEqual(test._name, 'inherited')
        self.assertIn('_myarg', test.__dict__)

    def test_serialization(self):
        a = ReilBase(name='test', logger_name='logger')
        b = ReilBase.from_config(a.get_config())
        self.assertEqual(a._name, b._name)
        self.assertEqual(a._path, b._path)
        self.assertEqual(a._logger._name, b._logger._name)

    # def test_yaml(self):
    #     config = '''
    #         base:
    #             reil.ReilBase:
    #                 name: test
    #                 path: .
    #     '''
    #     y = YAML().load(config)
    #     obj = ReilBase.parse_yaml(y['base'])
    #     self.assertIsInstance(obj, ReilBase)
    #     self.assertEqual(obj._name, 'test')  # type: ignore


if __name__ == "__main__":
    unittest.main()
