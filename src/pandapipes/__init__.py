# Copyright (c) 2020-2024 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import importlib.metadata

__version__ = importlib.metadata.version("pandapipes")
__format_version__ = '0.11.0'

import pandas as pd
import os

pd.options.mode.chained_assignment = None  # default='warn'
pp_dir = os.path.dirname(os.path.realpath(__file__))

from pandapipes.properties.fluids import *
from pandapipes.create import *
from pandapipes.io.file_io import *
from pandapipes.pipeflow import *
from pandapipes.toolbox import *
from pandapipes.pf.pipeflow_setup import *
from pandapipes.std_types import *
import pandapipes.plotting
