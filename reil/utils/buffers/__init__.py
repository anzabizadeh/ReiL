# -*- coding: utf-8 -*-
'''
Buffers module for reinforcement learning
==========================================

This module provides different types of buffer.

Classes
-------
Buffer: base class for all buffers. It provides basic functionality for setup,
adding to and picking from the buffer.

CircularBuffer: a buffer that overwrites the oldest items if more than its size
is added.

EndlessBuffer: a buffer without an end! This buffer keeps appending new items,
so it should be used with memory and performance constraints in mind.

VanilaExperienceReplay: a `CircularBuffer` that only picks randomly and returns
values only if it is full.


'''

from .buffer import Buffer, PickModes, T
from .circular_buffer import CircularBuffer
from .endless_buffer import EndlessBuffer
from .vanilla_experience_replay import VanillaExperienceReplay
