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


import copy

mylist = {'a1': {'a11': 10, 'a12': 20}, 'b1': {'a12': 30, 'z': 40}}

mylist_copy = copy.deepcopy(mylist)
print(mylist_copy)
print(mylist)

mylist['b1'].update({'z': 100000})
print(mylist_copy)
print(mylist)
