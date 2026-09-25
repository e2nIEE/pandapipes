# Copyright (c) 2020-2026 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
import numpy as np
import pandas as pd
import pytest

import pandapipes
from pandapipes.networks.simple_gas_networks import gas_versatility
from pandapipes.pf.pipeflow_setup import PipeflowNotConverged
from pandapipes.properties.fluids import FluidPropertyConstant


@pytest.mark.parametrize("use_numba", [True, False])
def test_pipeflow_non_convergence(use_numba):
    net = gas_versatility()
    pandapipes.get_fluid(net).add_property("molar_mass", FluidPropertyConstant(16.6))

    max_iter_hyd = 10 if use_numba else 10
    pandapipes.pipeflow(net, use_numba=use_numba, max_iter_hyd=max_iter_hyd)
    for comp in net["component_list"]:
        table_name = comp.table_name()
        assert np.all(net["res_" + table_name].index == net[table_name].index)
        if table_name == "valve":
            continue
        assert np.all(pd.notnull(net["res_" + table_name]))

    pandapipes.create_sink(net, 6, 100)
    with pytest.raises(PipeflowNotConverged):
        pandapipes.pipeflow(net, max_iter_hyd=max_iter_hyd,
                            use_numba=use_numba)

    for comp in net["component_list"]:
        table_name = comp.table_name()
        assert np.all(net["res_" + table_name].index == net[table_name].index)
        assert np.all(pd.isnull(net["res_" + table_name]))


def test_heat_mode_rejects_unconverged_hydraulics():
    """Regression test: initialize_pit()'s standalone mode="heat" branch used to reuse
    net["_pit"] just by checking it exists, without checking net.converged - if a caller caught
    PipeflowNotConverged from a failed hydraulics run and later ran mode="heat" on the same net
    (e.g. while batch-processing many nets, or just re-trying with different heat options), the
    heat solve would silently build on the unconverged, physically meaningless mdot/p values that
    non-convergent run still left behind - and then overwrite net.converged with the HEAT run's
    own (unrelated) outcome, erasing any sign that the underlying hydraulics never converged."""
    net = pandapipes.create_empty_network("net", add_stdtypes=False, fluid="water")
    j1, j2 = pandapipes.create_junctions(net, 2, pn_bar=5, tfluid_k=300)
    pandapipes.create_pipe_from_parameters(net, j1, j2, length_km=1, inner_diameter_mm=100)
    pandapipes.create_ext_grid(net, j1, p_bar=5, t_k=300, type="pt")
    pandapipes.create_sink(net, j2, mdot_kg_per_s=1)

    with pytest.raises(PipeflowNotConverged):
        pandapipes.pipeflow(net, mode="hydraulics", max_iter_hyd=1)
    assert not net.converged
    assert "_pit" in net  # the failed run still left a (non-converged) pit behind

    with pytest.raises(PipeflowNotConverged):
        pandapipes.pipeflow(net, mode="heat")
    assert not net.converged  # not silently overwritten by an unrelated heat-run outcome
