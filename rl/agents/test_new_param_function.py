# class Test():
#     def __init__(self, **kwargs):
#         self.__defaults = {'gamma': 1,
#                            'alpha': 1,
#                            'epsilon': 0,
#                            'default_actions': (),
#                            'state_action_list': {}}

#     def setparam(self, **kwargs):
#         self.__dict__.update(('_'+key, kwargs.get(key, self.__defaults[key]))
#                               for key in self.__defaults if key in kwargs)


# t = Test()
# t.setparam(gamma=10, boogh=20)
# print(t.__dict__)
# t.setparam(alpha=100, boogh=20)
# print(t.__dict__)
# t.setparam(gamma=100, boogh=20)
# print(t.__dict__)

from rl.agents import Agent

a = Agent()