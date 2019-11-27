class dtype(list):
    def __setitem__(self, key, value):
        print(key, type(key), value)
        return super().__setitem__(key, value)

a = dtype([1, 2, 3])
a[0] = 100
a[:] = [2, 3]

print(slice(None))



# # import numpy as np
# import pandas as pd
# from time import time
# from collections import deque

# entries = 1000

# t = time()

# agent_list = ['a', 'b', 'c']
# h = dict((agent_info, deque([])) for agent_info in agent_list)

# for i in range(entries):
#     for agent_name in agent_list:
#         h[agent_name].append({'state': tuple(range(10)), 'action': 2.0, 'reward': 3.0})

# for agent_name in agent_list:
#     for i in range(len(h[agent_name])):
#         state = h[agent_name][i]['state']
#         action = h[agent_name][i]['action']
#         reward = h[agent_name][i]['reward']

# print(time() - t)


# t = time()

# gent_list = ['a', 'b', 'c']
# h = dict((agent_info, []) for agent_info in agent_list)

# for i in range(entries):
#     for agent_name in agent_list:
#         h[agent_name].append([tuple(range(10)), 2.0, 3.0])

# for agent_name in agent_list:
#     for i in range(len(h[agent_name])):
#         state = h[agent_name][i][0]
#         action = h[agent_name][i][1]
#         reward = h[agent_name][i][2]

# print(time() - t)

# t = time()

# gent_list = ['a', 'b', 'c']
# h = dict((agent_info, deque([])) for agent_info in agent_list)

# for i in range(entries):
#     for agent_name in agent_list:
#         h[agent_name].append([tuple(range(10)), 2.0, 3.0])

# for agent_name in agent_list:
#     for i in range(len(h[agent_name])):
#         state = h[agent_name][i][0]
#         action = h[agent_name][i][1]
#         reward = h[agent_name][i][2]

# print(time() - t)
