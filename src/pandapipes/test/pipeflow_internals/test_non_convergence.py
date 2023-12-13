# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
import numpy as np
import pandas as pd
import pytest

import pandapipes
from pandapipes.networks.simple_gas_networks import gas_versatility
from pandapipes.pipeflow import PipeflowNotConverged
from pandapipes.properties.fluids import FluidPropertyConstant


@pytest.mark.parametrize("use_numba", [True, False])
def test_pipeflow_non_convergence(use_numba):
    net = gas_versatility()
    pandapipes.get_fluid(net).add_property("molar_mass", FluidPropertyConstant(16.6))

    pandapipes.pipeflow(net, use_numba=use_numba)
    for comp in net["component_list"]:
        table_name = comp.table_name()
        assert np.all(net["res_" + table_name].index == net[table_name].index)
        if table_name == "valve":
            continue
        assert np.all(pd.notnull(net["res_" + table_name]))

    pandapipes.create_sink(net, 6, 100)
    with pytest.raises(PipeflowNotConverged):
        pandapipes.pipeflow(net, use_numba=use_numba)

    for comp in net["component_list"]:
        table_name = comp.table_name()
        assert np.all(net["res_" + table_name].index == net[table_name].index)
        assert np.all(pd.isnull(net["res_" + table_name]))
