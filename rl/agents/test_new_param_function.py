# from rl.subjects.mnkgame import MNKGame

class Test():
    def __init__(self, **kwargs):
        pass
        # self.__defaults = {'gamma': 1,
        #                    'alpha': 1,
        #                    'epsilon': 0,
        #                    'default_actions': (),
        #                    'state_action_list': {}}

    def setparam(self, **kwargs):
        self.__dict__.update(('_'+key, kwargs.get(key, self.__defaults[key]))
                              for key in self.__defaults if key in kwargs)

    def set_defaults(self, **params):
        if not hasattr(self, '__defaults'):
            self.__defaults = {}
        for key, value in params.items():
            self.__defaults[key] = value


# board = MNKGame()
# print(board.printable())
# t = Test()
# t.set_defaults(gamma=10)
# t.setparam(gamma=10, boogh=20)
# print(t.__dict__)
# t.setparam(alpha=100, boogh=20)
# print(t.__dict__)
# t.setparam(gamma=100, boogh=20)
# print(t.__dict__)

from rl.agents import QAgent
from rl.valueset import ValueSet

a = QAgent()
print(a.act(ValueSet(0)))
print(a.__dict__)