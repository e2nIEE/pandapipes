# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
from pandapipes.pf.internals_toolbox import select_from_pit


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

