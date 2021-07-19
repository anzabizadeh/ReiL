# -*- coding: utf-8 -*-
'''
environments module for reinforcement learning
==============================================

This module contains classes that provides tools to load/save objects and
environments, add/remove `agents`/`subjects`, assign `agents` to `subjects` and
run models.

Classes
-------
Environment:
    The base class of all environment classes.

EnvironmentStaticMap:
    An environment with static interaction sequence.

Task:
    A class to define a `ReiL` task.

Session:
    A class that accepts `Agent`s, `Subject`s, and `Demon`s, and
    runs a set of `Task`s.

SessionBuilder:
    A tool to create a `Session` from one or a list of `YAML` config files.
'''

from .environment import (  # noqa: W0611
    Environment, EntityType, EntityGenType, AgentSubjectTuple)
from .environment_static_map import EnvironmentStaticMap  # noqa: W0611
from .task import Task  # noqa: W0611
from .session import Session  # noqa: W0611
from .session_builder import SessionBuilder  # noqa: W0611
from .trajectory import Trajectory  # noqa: W0611
