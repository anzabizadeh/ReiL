# import numpy as np
import pandas as pd
from time import time

entries = 10

agent_list = ['a', 'b', 'c']
h = dict((agent_info, []) for agent_info in agent_list)

t = time()
for i in range(entries):
    for agent_name in agent_list:
        h[agent_name].append([tuple(range(10)), 2.0, 3.0])

history2 = {}
for agent_name in agent_list:
    history2[agent_name] = pd.DataFrame(h[agent_name], columns=['state', 'action', 'reward'])

print(time() - t)


agent_list = ['a', 'b', 'c']
history = dict((agent_info, pd.DataFrame(columns=['state', 'action', 'reward'])) for agent_info in agent_list)

t = time()
for i in range(entries):
    for agent_name in agent_list:
        history[agent_name].loc[len(history[agent_name].index)] = [tuple(range(10)), 2.0, 3.0]

print(time() - t)

