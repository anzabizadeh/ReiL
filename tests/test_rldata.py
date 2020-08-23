import unittest
from random import randint, sample

from rl import rldata


class testRLData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._numerical_data = [randint(-100, 100) for _ in range(10)]
        cls._categorical_data = sample(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 10)

    def test_create_baserldata(self):
        rl_value = rldata.BaseRLData(name='test',
                        value=self._numerical_data,
                        categorical=False,
                        normalizer=lambda _: 1,  # type: ignore
                        lazy_evaluation=True)
        self.assertEqual(rl_value.name, 'test')
        self.assertEqual(rl_value.value, self._numerical_data)
        self.assertEqual(rl_value.normalized, 1)
        self.assertEqual(rl_value.lazy_evaluation, True)

        rl_value = rldata.BaseRLData(name='test',
                        value=self._categorical_data,
                        categorical=False,
                        normalizer=lambda _: 1,  # type: ignore
                        lazy_evaluation=True)
        self.assertEqual(rl_value.name, 'test')
        self.assertEqual(rl_value.value, self._categorical_data)
        self.assertEqual(rl_value.normalized, 1)
        self.assertEqual(rl_value.lazy_evaluation, True)

    def test_modify_baserldata(self):
        rl_value = rldata.BaseRLData(name='test',
                        categorical=False,
                        value=self._numerical_data.copy(),
                        normalizer=lambda x: 0 if isinstance(x.value, str) else 1,  # type: ignore
                        lazy_evaluation=False)

        rl_value.value[0] = self._numerical_data[0] * 10
        self.assertEqual(rl_value.value, [self._numerical_data[0] * 10] + self._numerical_data[1:])

        rl_value.value = 'hello'
        self.assertEqual(rl_value.value, 'hello')
        self.assertEqual(rl_value.normalized, 0)

        rl_value.value = 100
        self.assertEqual(rl_value.value, 100)
        self.assertEqual(rl_value.normalized, 1)

        rl_value.lazy_evaluation = True
        self.assertEqual(rl_value._normalized, None)
        rl_value.lazy_evaluation = False
        self.assertEqual(rl_value._normalized, 1)
        rl_value.lazy_evaluation = True
        self.assertEqual(rl_value._normalized, None)
        rl_value.lazy_evaluation = False
        self.assertEqual(rl_value._normalized, 1)

    def test_create_rangeddata(self):
        # categorical = False
        rl_value = rldata.RangedData(name='numerical',
                                        categorical = False,
                                        value=self._numerical_data)

        self.assertIsNone(rl_value.lower)
        self.assertIsNone(rl_value.upper)
        self.assertIsNone(rl_value.normalized)

        rl_value = rldata.RangedData(name='numerical',
                                        categorical = False,
                                        value=self._numerical_data,
                                        lower=min(self._numerical_data),
                                        upper=max(self._numerical_data))

        self.assertEqual(rl_value.lower, min(self._numerical_data))
        self.assertEqual(rl_value.upper, max(self._numerical_data))
        self.assertIsNotNone(rl_value.normalized)

        # categorical = True
        rl_value = rldata.RangedData(name='categorical',
                                        categorical = True,
                                        value=self._categorical_data)

        self.assertIsNone(rl_value.categories)
        self.assertIsNone(rl_value.normalized)

        rl_value = rldata.RangedData(name='categorical',
                                        categorical = True,
                                        value=self._categorical_data,
                                        categories=list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))

        self.assertEqual(rl_value.categories, list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
        self.assertIsNotNone(rl_value.normalized)

    def test_modify_rangeddata(self):
        # categorical = False
        rl_value = rldata.RangedData(name='numerical',
                                        categorical = False,
                                        value=self._numerical_data.copy())

        rl_value.lower = min(self._numerical_data)
        rl_value.upper = max(self._numerical_data)
        self.assertEqual(rl_value.lower, min(self._numerical_data))
        self.assertEqual(rl_value.upper, max(self._numerical_data))

        rl_value.value[0] = self._numerical_data[0] * 10
        self.assertEqual(rl_value.value, [self._numerical_data[0] * 10] + self._numerical_data[1:])

        with self.assertRaises(ValueError):
           rl_value.lower = min(self._numerical_data) + 1
           rl_value.lower = max(self._numerical_data) - 1

        with self.assertRaises(TypeError):
           rl_value.lower = 'x'
           rl_value.upper = 'x'

        # categorical = True
        rl_value = rldata.RangedData(name='numerical',
                                        categorical = True,
                                        value=self._categorical_data.copy())

        self.assertIsNone(rl_value.categories)
        self.assertIsNone(rl_value.normalized)

        rl_value.categories = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        self.assertEqual(rl_value.categories, list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
        self.assertIsNotNone(rl_value.normalized)

        with self.assertRaises(ValueError):
           rl_value.categories = ['x']

        with self.assertRaises(TypeError):
           rl_value.categories = 'x'

    def test_categorical_normalizer(self):
        rl_value = rldata.RangedData(name='categorical',
                                        categorical = True,
                                        value='A',
                                        categories=list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))

        self.assertEqual(rl_value.normalized, [1] + [0]*25)

        rl_value.value = 'Z'
        self.assertEqual(rl_value.normalized, [0]*25 + [1])

        rl_value.value = ['A', 'Z']
        self.assertEqual(rl_value.normalized, [1] + [0]*25 + [0]*25 + [1])

        rl_value = rldata.RangedData(name='categorical_tuple',
                                        categorical = True,
                                        value=('A', 'A'),
                                        categories=list((x, y) for x in 'ABCD' for y in 'ABCD'))

        self.assertEqual(rl_value.normalized, [1] + [0]*15)

        rl_value.value = ('D', 'D')
        self.assertEqual(rl_value.normalized, [0]*15 + [1])

        rl_value.value = [('A', 'B'), ('B', 'A')]
        self.assertEqual(rl_value.normalized, [0, 1] + [0]*14 + [0]*4 + [1] + [0]*11)

    def test_indexing_rangeddata(self):
        # categorical = False
        rl_value = rldata.RangedData(name='numerical',
                                        categorical = False,
                                        value=self._numerical_data,
                                        lower=min(self._numerical_data),
                                        upper=max(self._numerical_data))

        rl_value[0] = (min(self._numerical_data) + max(self._numerical_data)) / 2
        self.assertEqual(rl_value.value[0], self._numerical_data[0])

        rl_value.value = (min(self._numerical_data) + max(self._numerical_data)) / 2
        rl_value[0] = min(self._numerical_data)
        self.assertEqual(rl_value.value[0], self._numerical_data[0])

        # categorical = True
        rl_value = rldata.RangedData(name='categorical',
                                        categorical = True,
                                        value=self._categorical_data,
                                        categories=list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))

        rl_value[0] = 'Z'

        self.assertEqual(rl_value.value[0], self._numerical_data[0])
        # rl_value = RLData.RangedData(name='categorical',
        #                                 categorical = True,
        #                                 value=self._categorical_data)

        # self.assertIsNone(rl_value.categories)
        # self.assertIsNone(rl_value.normalized)


        # self.assertEqual(rl_value.categories, list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
        # self.assertIsNotNone(rl_value.normalized)



if __name__ == "__main__":
    unittest.main()
