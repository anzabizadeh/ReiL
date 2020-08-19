  
# -*- coding: utf-8 -*-

# By using `os.path.dirname(__file__)` you can run the test from any directory/cwd and it will find the right path to the sample module. Without it you might accidentally import it from the wrong directory (i.e. if you have a development version of your project checked out, but you also have one installed in the system path, you may think that your tests pass when in fact they were testing the system installation rather than the development version).


import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import rl
