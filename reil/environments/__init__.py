# -*- coding: utf-8 -*-
'''
environments module for reinforcement learning
==============================================

This module contains classes that provides tools to load/save objects and
environments, add/remove agents/subjects, assign agents to subjects and run models.

Classes
-------
EnvironmentStaticMap: an environment with static interaction map.

Environment: (disabled) the base class that provides minimum required
functionality for a reinforcement learning environment.

Experiment: (disabled) an environment to explore performance of trained
agents on subjects.

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

# TODO: see if Experiment is still useful.

from .environment_static_map import EnvironmentStaticMap
# from .environment import Environment
# from .experiment import Experiment
