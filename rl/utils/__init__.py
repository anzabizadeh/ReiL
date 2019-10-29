# -*- coding: utf-8 -*-
'''
subjects module for reinforcement learning
==========================================

This module provides different subjects in reinforcement learning
context.

Classes
-------
    MNKBoard: an m-by-n board in which k similar horizontal, vertical, or
        diagonal sequence is a win.
    Patient: the Warfarin PK/PD model based on Hamberg et al (2006)
    WekaClusterer: a clustering class based on Weka's clustering capabilities

@author: Sadjad Anzabi Zadeh (sadjad-anzabizadeh@uiowa.edu)
'''

from .mnkboard import MNKBoard
from .patient import Patient
from .weka_clustering import WekaClusterer
