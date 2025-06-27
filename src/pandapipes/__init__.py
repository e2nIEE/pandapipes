# Copyright (c) 2020-2025 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import importlib.metadata

try:
    __version__ = importlib.metadata.version("pandapipes")
except importlib.metadata.PackageNotFoundError:
    # if the package is not installed, try reading the toml itself
    import tomllib
    from pathlib import Path
    toml_file = Path(__file__).parent / "../../pyproject.toml"
    if toml_file.exists() and toml_file.is_file():
        with toml_file.open("rb") as f:
            data = tomllib.load(f)
            if "project" in data and "version" in data["project"]:
                __version__ = data["project"]["version"]

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
