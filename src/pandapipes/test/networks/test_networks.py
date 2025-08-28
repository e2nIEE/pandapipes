# Copyright (c) 2020-2025 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
import numpy as np
import pandas as pd
from pandapower.control import ConstControl

import pandapipes
import pytest
from pandapipes import networks
from pandapower.timeseries import OutputWriter
from pandapower.timeseries.data_sources.frame_data import DFData

from pandapipes.timeseries.run_time_series import run_timeseries
from pandapipes import topology as top


def test_schutterwald():
    net = networks.schutterwald_gas(True, None)
    max_iter_hyd = 3
    pandapipes.pipeflow(net, max_iter_hyd=max_iter_hyd)
    assert net.converged

    net2 = networks.schutterwald_gas(False, None)
    assert net2.sink.empty
    assert len(net2.pipe.loc[net2.pipe.type == "house_connection"]) == 0
    max_iter_hyd = 2
    pandapipes.pipeflow(net2, max_iter_hyd=max_iter_hyd)
    assert net2.converged

    net3 = networks.schutterwald_gas(True, 30)
    assert len(net3.sink) == 1506
    assert net3.pipe.loc[net3.pipe.in_service & (net3.pipe.type == "house_connection"),
                         "length_km"].max() <= 0.03
    max_iter_hyd = 3
    pandapipes.pipeflow(net3, max_iter_hyd=max_iter_hyd)
    assert net3.converged


def test_schutterwald_heat():
    net = networks.schutterwald_heat(80)
    pandapipes.pipeflow(net, mode="sequential")
    assert np.all(net.res_junction.p_bar.to_numpy() > 0)
    assert np.all(net.res_junction.t_k.to_numpy() > net.user_pf_options["ambient_temperature"] - 0.01)


def test_schutterwald_heat_transient():
    net = networks.schutterwald_heat(80)
    dt_simulation = 120

    np.random.seed(30)
    mdot = np.cumsum(np.random.random([200, 4]) - 0.5, axis=0)
    mdot -= np.min(mdot, axis=0)
    mdot /= np.max(mdot, axis=0)
    mdot_df = pd.DataFrame(mdot)

    ts0_c = np.where(mdot == 0)
    ts0 = [ts0_c[0][list(ts0_c[1]).index(i)] for i in range(len(ts0_c[0]))]

    ds = DFData(mdot_df)

    initial_mdot = net["heat_consumer"]["controlled_mdot_kg_per_s"].to_numpy().copy()
    initial_qext = net["heat_consumer"]["qext_w"].to_numpy().copy()

    columns = np.tile(np.arange(4), int(np.ceil(len(net.heat_consumer) / 4)))[:len(net.heat_consumer)]

    ConstControl(net, "heat_consumer", "controlled_mdot_kg_per_s", net.heat_consumer.index,
                 profile_name=columns, data_source=ds, scale_factor=initial_mdot)
    ConstControl(net, "heat_consumer", "qext_w", net.heat_consumer.index,
                 profile_name=columns, data_source=ds, scale_factor=initial_qext)

    ow = OutputWriter(net, time_steps=None, log_variables=[
        ("res_junction", "t_k"),
        ("res_junction", "p_bar"),
        ("res_pipe", "v_mean_m_per_s"),
        ("res_pipe", "t_outlet_k"),
        ("res_heat_consumer", "qext_w"),
        ("res_heat_consumer", "t_outlet_k"),
    ])

    run_timeseries(net, continue_on_divergence=False, verbose=True,
                   mode="sequential", transient=True, dt=dt_simulation, iter=200, use_numba=True)

    mg = top.create_nxgraph(net, include_heat_consumers=False, include_pressure_circ_pumps=False)

    assert ow.output["res_pipe.v_mean_m_per_s"].abs().max().max() < 3
    assert np.allclose(
        ow.output["res_heat_consumer.t_outlet_k"].loc[ts0[0], columns == 0],
        ow.output["res_heat_consumer.t_outlet_k"].loc[ts0[0] - 1, columns == 0],
    )
    for c in range(4):
        juncs0 = [fj for fj in net.heat_consumer.loc[columns == c, "from_junction"] if len(mg[fj]) == 1]
        assert np.all(
            ow.output["res_junction.t_k"].loc[ts0[c], juncs0]
            < ow.output["res_junction.t_k"].loc[ts0[c] - 1, juncs0]
        )



if __name__ == '__main__':
    n = pytest.main(["test_networks.py"])