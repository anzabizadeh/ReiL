import random as rand
import numpy as np
from sklearn import exceptions
from sklearn.neural_network import MLPRegressor

clf_warm = MLPRegressor(solver='lbfgs', alpha=0.1, hidden_layer_sizes=(10, 5), max_iter=2, warm_start=True)
# clf_cold = MLPRegressor(solver='lbfgs', alpha=0.1, hidden_layer_sizes=(10,), max_iter=1, warm_start=False)

def f(x):
    value = list(i ** 2 for i in x)
    return np.sum(value, axis=1)

print('training')
for _ in range(5000):
    X = np.array([[rand.randint(0, 10) for _ in range(3)] for _ in range(5)])
    Y = np.array(f(X))
    # clf_cold.fit(X, Y)
    clf_warm.fit(X, Y)
    # Y_cold = clf_cold.predict(X)
    # Y_warm = clf_warm.predict(X)
    # print((Y-Y_warm)/Y, end=', ')

print('\ntesting')
for _ in range(10):
    X = np.array([[rand.randint(0, 10) for _ in range(3)]])
    Y = np.array([f(X)])
    # clf_cold.fit(X, Y)
    # clf_warm.fit(X, Y)
    # Y_cold = clf_cold.predict(X)
    Y_warm = clf_warm.predict(X)
    print(Y, Y_warm, (Y-Y_warm)/Y)
