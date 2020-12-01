# -*- coding: utf-8 -*-
'''
datatypes module for reinforcement learning
===========================================

This module contains datatypes used in `reil`

Classes
-------
Feature: a datatype that accepts initial value and feature generator and generates new values.

InteractionProtocol: a datatype that specifies how an agent and a subject interact in an environment.

ReilData: the main datatype used to communicate states, actions, and rewards, between objects in `reil`.

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

from .feature import Feature, FeatureType
from .interaction_protocol import InteractionProtocol
from .reildata import ReilData
