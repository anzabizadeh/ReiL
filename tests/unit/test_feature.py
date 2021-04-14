import unittest
from random import randint, sample

from reil.datatypes import feature


class testFeature(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._numerical_data = [randint(-100, 100) for _ in range(10)]
        cls._categorical_data = sample(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 10)

    def test_create_feature(self):
        rl_value = feature.Feature(
            name='test', value=self._numerical_data, is_numerical=True)
        self.assertEqual(rl_value.name, 'test')
        self.assertEqual(rl_value.value, self._numerical_data)
        self.assertEqual(rl_value.is_numerical, True)

        lower = min(self._numerical_data)
        upper = max(self._numerical_data)

        def midpoint(f):
            return (f.lower + f.upper)/2

        numerical = feature.FeatureGenerator.numerical(
            name='num', lower=lower,
            upper=upper, generator=midpoint, randomized=True)

        self.assertEqual(numerical.name, 'num')
        self.assertEqual(numerical().value, (lower + upper)/2)
        self.assertEqual(numerical.is_numerical, True)

        categorical = feature.FeatureGenerator.categorical(
            name='cat', categories=('A', 'B', 'C'), probabilities=(.3, .6, .1),
            generator=lambda f: 'A', randomized=True)

        self.assertEqual(categorical.name, 'cat')
        self.assertEqual(categorical().value, 'A')
        self.assertEqual(categorical.is_numerical, False)

    def test_add_features(self):
        rl_values = [feature.Feature(
            name='test', value=v, is_numerical=True)
            for v in self._numerical_data]
        rl_new = sum(rl_values[1:], rl_values[0])
        self.assertEqual(rl_new.name, 'test')
        self.assertEqual(rl_new.value, sum(self._numerical_data))
        self.assertEqual(rl_new.is_numerical, True)

    def test_add_numerical(self):
        numericals = feature.FeatureGenerator.numerical(name='test')
        rl_values = [numericals(v) for v in self._numerical_data]

        rl_new = sum(rl_values[1:], rl_values[0])
        self.assertEqual(rl_new.name, 'test')
        self.assertEqual(rl_new.value, sum(self._numerical_data))
        self.assertEqual(rl_new.is_numerical, True)

        rl_values = [numericals((v,)) for v in self._numerical_data]

        rl_new = sum(rl_values[1:], rl_values[0])
        self.assertEqual(rl_new.name, 'test')
        self.assertEqual(rl_new.value, tuple(self._numerical_data))
        self.assertEqual(rl_new.is_numerical, True)

    def test_add_categorical(self):
        categoricals = feature.FeatureGenerator.categorical(
            name='test', categories=tuple(self._categorical_data))
        rl_values = [categoricals(v) for v in self._categorical_data]
        self.assertEqual(
            rl_values[0].normalized,
            tuple([1] + [0]*(len(self._categorical_data) - 2)))
        self.assertEqual(
            rl_values[-1].normalized,
            tuple([0]*(len(self._categorical_data) - 1)))

        rl_new = sum(rl_values[1:], rl_values[0])
        self.assertEqual(rl_new.name, 'test')
        self.assertEqual(rl_new.value, ''.join(self._categorical_data))
        self.assertEqual(rl_new.is_numerical, False)

        rl_values = [categoricals((v,)) for v in self._categorical_data]

        rl_new = sum(rl_values[1:], rl_values[0])
        self.assertEqual(rl_new.name, 'test')
        self.assertEqual(rl_new.value, tuple(self._categorical_data))
        self.assertEqual(rl_new.is_numerical, False)

    def test_feature_array(self):
        categoricals = feature.FeatureGenerator.categorical(
            name='cats', categories=tuple(self._categorical_data))
        numericals = feature.FeatureGenerator.numerical(
            name='nums', lower=min(self._numerical_data),
            upper=max(self._numerical_data))

        with self.assertRaises(KeyError):
            feature.FeatureArray(
                [categoricals(v) for v in self._categorical_data])
            # print(rl_array.categories)

        reilarray = feature.FeatureArray([
            categoricals(self._categorical_data[0]),
            numericals(self._numerical_data[0]),
        ])

        self.assertEqual(reilarray.lower,
                         {reilarray['cats'].name: reilarray['cats'].lower,
                          reilarray['nums'].name: reilarray['nums'].lower})
        self.assertEqual(reilarray.upper,
                         {reilarray['cats'].name: reilarray['cats'].upper,
                          reilarray['nums'].name: reilarray['nums'].upper})
        self.assertEqual(
            reilarray.categories,
            {reilarray['cats'].name: reilarray['cats'].categories,
             reilarray['nums'].name: reilarray['nums'].categories})

        self.assertEqual(reilarray.normalized.flatten(),
                         list(reilarray['cats'].normalized) +
                         [reilarray['nums'].normalized])

        self.assertEqual(
            reilarray,
            feature.FeatureArray([categoricals(self._categorical_data[0])]) +
            feature.FeatureArray([numericals(self._numerical_data[0])])
            )


if __name__ == "__main__":
    unittest.main()
