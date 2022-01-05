# Copyright (c) 2020-2021 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
import pandas as pd
from pandapipes.networks.simple_gas_networks import gas_versatility
import pandapipes as pp
import pytest
from pandapipes.pipeflow import PipeflowNotConverged
import numpy as np
from pandapipes.properties.fluids import FluidPropertyConstant


def test_pipeflow_non_convergence():
    net = gas_versatility()
    pp.get_fluid(net).add_property("molar_mass", FluidPropertyConstant(16.6))

    pp.pipeflow(net)
    for comp in net["component_list"]:
        table_name = comp.table_name()
        assert np.all(net["res_" + table_name].index == net[table_name].index)
        if table_name == "valve":
            continue
        assert np.all(pd.notnull(net["res_" + table_name]))

    pp.create_sink(net, 6, 100)
    with pytest.raises(PipeflowNotConverged):
        pp.pipeflow(net)

    try:
        pp.pipeflow(net)
    except PipeflowNotConverged:
        pass

    for comp in net["component_list"]:
        table_name = comp.table_name()
        assert np.all(net["res_" + table_name].index == net[table_name].index)
        assert np.all(pd.isnull(net["res_" + table_name]))
