# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
import pandapipes
import pandas as pd
import pytest
from pandapipes.component_models import Pipe, Junction
from pandapipes.idx_node import PINIT, TINIT
from pandapipes.pipeflow_setup import get_lookup
from pandapipes.test.pipeflow_internals import internals_data_path
from pandapipes.internals_toolbox import select_from_pit


def test_select_from_pit():
    """

    :return:
    :rtype:
    """

    input_array = np.array([2,4,5])
    table_index_array = np.array([1,2,3,4,5])
    data = np.array([10,11,12,13,14])

    ret = select_from_pit(table_index_array,input_array,data)
    expected_result = np.array([11, 13,14])

    assert np.all(ret == expected_result)

