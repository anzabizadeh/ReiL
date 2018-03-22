'''
Learning signal (for intrrupt)
'''
# import signal
# import sys
# def signal_handler(signal, frame):
#         print('You pressed Ctrl+C!')
#         sys.exit(0)
# signal.signal(signal.SIGINT, signal_handler)
# print('Press Ctrl+C')
# i=0
# while True:
#     i+=1
#     print(i)


'''
Learning Profiling
'''
# import timeit

# test1 = {'a1': {'a11': 10, 'a12': 20}, 'b1': {'a12': 1000, 'z': 5}}
# test2 = {'b1': {'b11': 30, 'a12': 40}}

# test3 = {('a1', 'a11'): 10, ('a1', 'a12'): 20, ('b1', 'a12'): 1000, ('b1', 'z'): 5}
# test4 = {('b1', 'b11'): 30, ('b1', 'a12'): 40}


# def code_nested():
#     for _ in range(100000):
#         dif=0
#         keylist = set(list(test1.keys()) + list(test2.keys()))
#         for k in keylist:
#             keylist2 = set(list(test1.get(k, {}).keys()) + list(test2.get(k, {}).keys()))
#             for l in keylist2:
#                 dif += abs(test1.get(k, {}).get(l, 0) - test2.get(k, {}).get(l, 0))
#     return dif

# def code_tuple():
#     for _ in range(100000):
#         dif=0
#         keylist = set(list(test3.keys()) + list(test4.keys()))
#         for k in keylist:
#             dif += abs(test3.get(k, 0) - test2.get(k, 0))
#     return dif

# print(timeit.timeit(code_nested, number=10000))
# print(timeit.timeit(code_tuple, number=10000))

# import cProfile, pstats, io
# pr = cProfile.Profile()
# pr.enable()
# # ... do something ...
# code_tuple()
# pr.disable()
# s = io.StringIO()
# sortby = 'cumulative'
# ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
# ps.print_stats()
# print(s.getvalue())
# pr = cProfile.Profile()
# pr.enable()
# # ... do something ...
# code_nested()
# pr.disable()
# s = io.StringIO()
# sortby = 'cumulative'
# ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
# ps.print_stats()
# print(s.getvalue())



'''
Learning Neural Networks!
'''
# import random as rand
# import numpy as np
# from sklearn import exceptions
# from sklearn.neural_network import MLPRegressor

# clf_warm = MLPRegressor(solver='lbfgs', alpha=0.1, hidden_layer_sizes=(10, 5), max_iter=2, warm_start=True)
# # clf_cold = MLPRegressor(solver='lbfgs', alpha=0.1, hidden_layer_sizes=(10,), max_iter=1, warm_start=False)

# def f(x):
#     value = list(i ** 2 for i in x)
#     return np.sum(value, axis=1)

# print('training')
# for _ in range(5000):
#     X = np.array([[rand.randint(0, 10) for _ in range(3)] for _ in range(5)])
#     Y = np.array(f(X))
#     # clf_cold.fit(X, Y)
#     clf_warm.fit(X, Y)
#     # Y_cold = clf_cold.predict(X)
#     # Y_warm = clf_warm.predict(X)
#     # print((Y-Y_warm)/Y, end=', ')

# print('\ntesting')
# for _ in range(10):
#     X = np.array([[rand.randint(0, 10) for _ in range(3)]])
#     Y = np.array([f(X)])
#     # clf_cold.fit(X, Y)
#     # clf_warm.fit(X, Y)
#     # Y_cold = clf_cold.predict(X)
#     Y_warm = clf_warm.predict(X)
#     print(Y, Y_warm, (Y-Y_warm)/Y)
