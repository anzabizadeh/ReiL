#%% warfarin with lookahead

from rl.subjects.warfarin_model_v5 import WarfarinModel_v5
from rl.rldata import RLData
from copy import deepcopy

class WarfarinLookAhead(WarfarinModel_v5):
    def __init__(self, **kwargs):
        self._lookahead_duration = kwargs.get('lookahead_duration', 7)
        kwargs['max_day'] = kwargs.get('max_day', 90) + self._lookahead_duration
        super().__init__(**kwargs)
        self._INR_penalty_coef = kwargs.get('INR_penalty_coef', 1)
        self._lookahead_penalty_coef = kwargs.get('lookahead_penalty_coef', 0)

    @property
    def is_terminated(self):
        return self._day >= self._max_day - self._lookahead_duration

    def take_effect(self, action, _id=None):
        self._current_dose = action[0]
        self._dosing_intervals.append(self._d_current)
        self._dosing_intervals.popleft()
        self._dose_list.append(self._current_dose)
        self._dose_list.popleft()

        # self._patient.dose = {self._day: self._current_dose}
        if (self._initial_phase_duration == -1 and self._therapeutic_range[0] <= self._INR[-1] <= self._therapeutic_range[1]) \
            or self._day == self._initial_phase_duration:
                self._phase = 'maintenance'
        if self._phase == 'initial':
            self._d_current = 1  # action.value[1]
            self._patient.dose = {self._day: self._current_dose}
        else:
            self._d_current = min(self._maintenance_day_interval, self._max_day-self._day)
            self._patient.dose = dict(tuple((i + self._day, self._current_dose) for i in range(self._d_current)))

        temp_patient = deepcopy(self._patient)
        temp_patient.dose = dict(tuple((i + self._day, self._current_dose) for i in range(1, self._lookahead_duration + 1)))
        lookahead_penalty = -sum(((2 / self._INR_range * (self._INR_mid - INRi)) ** 2
                                for INRi in temp_patient.INR(list(range(self._day + 1, self._day + self._lookahead_duration + 1)))))

        self._day += self._d_current

        self._INR.append(self._patient.INR(self._day)[-1])
        self._INR.popleft()

        try:
            INR_penalty = -sum(((2 / self._INR_range * (self._INR_mid - self._INR[-2] - (self._INR[-1]-self._INR[-2])/self._d_current*j)) ** 2
                                for j in range(1, self._d_current + 1)))  # negative squared distance as reward (used *2/range to normalize)
            dose_change_penalty = - self._dose_change_penalty_func(self._dose_list)
            reward = self._INR_penalty_coef * INR_penalty \
                    + self._dose_change_penalty_coef * dose_change_penalty \
                    + self._lookahead_penalty_coef * lookahead_penalty
        except TypeError:
            reward = 0

        # return TTR*self._d_current
        return RLData(reward, normalizer=lambda x: x)

if __name__ == '__main__':
    w = WarfarinLookAhead(lookahead_penalty_coef=1, INR_penalty_coef=0)
    for i in range(90):
        reward = w.take_effect(RLData(10, lower=0, upper=15))
        print(reward[0], w._INR[-1])

#%% Truing to define a new datatype

# class dtype(list):
#     def __setitem__(self, key, value):
#         print(key, type(key), value)
#         return super().__setitem__(key, value)

# a = dtype([1, 2, 3])
# a[0] = 100
# a[:] = [2, 3]

# print(slice(None))


#%% Testing the speed of different datatypes

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
