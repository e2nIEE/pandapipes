# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import pandapipes as pp
import os
import pytest
import pandapipes.networks.release_control_test_network as nw
import pandapipes.timeseries
from pandapower.timeseries import OutputWriter
from pandapipes.pipeflow import logger as pf_logger

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
pf_logger.setLevel(logging.WARNING)


def test_release_control_pipeflow_json():
    """
    Normal network calculation with pipeflow and the loaded network from the json file.

    """
    net = nw.release_control_test_network(use_json=True)

    pp.pipeflow(net, stop_condition="tol", iter=100, tol_p=1e-7, tol_v=1e-7, friction_model="colebrook",
                mode="all", transient=False, nonlinear_method="automatic")

    # results
    '''
    print("\n junctions \n", net.res_junction)
    print("\n pipes \n", net.res_pipe)
    print("\n valves \n", net.res_valve)
    print("\n heat exchanger \n", net.res_heat_exchanger)
    print("\n pump \n", net.res_pump)
    print("\n circ. pump const mass flow \n", net.res_circ_pump_mass)
    print("\n circ. pump pressure \n", net.res_circ_pump_pressure)
    '''


def test_release_control_pipeflow_without_json():
    """
    Normal network calculation with pipeflow and the directly created pandapipes network.

    """
    net = nw.release_control_test_network()

    pp.pipeflow(net, stop_condition="tol", iter=100, tol_p=1e-7, tol_v=1e-7, friction_model="colebrook",
                mode="all", transient=False, nonlinear_method="automatic")


@pytest.mark.xfail
def test_release_control_ts_json():
    """
    Time series calculation with loaded net from the json file.

    """
    # get net
    net = nw.release_control_test_network(use_json=True, use_time_series=True)

    # preparation for run time series
    time_steps = range(4)

    log_variables = [
        ('res_junction', 'p_bar'), ('res_pipe', 'v_mean_m_per_s'),
        ('res_pipe', 'reynolds'), ('res_pipe', 'lambda'),
        ('res_sink', 'mdot_kg_per_s'), ('res_source', 'mdot_kg_per_s'),
        ('res_ext_grid', 'mdot_kg_per_s')]
    ow = OutputWriter(net, time_steps, output_path=None, log_variables=log_variables)

    # run time series
    pp.timeseries.run_time_series.run_timeseries_ppipe(net, time_steps, output_writer=ow)

    # results
    '''
    print("pressure \n", ow.np_results["res_junction.p_bar"])
    print("mean velocity \n", ow.np_results["res_pipe.v_mean_m_per_s"])
    print("reynolds number \n", ow.np_results["res_pipe.reynolds"])
    print("lambda \n", ow.np_results["res_pipe.lambda"])
    print("mass flow sink \n", ow.np_results["res_sink.mdot_kg_per_s"])
    print("mass flow source \n", ow.np_results["res_source.mdot_kg_per_s"])
    print("mass flow ext. grid \n", ow.np_results["res_ext_grid.mdot_kg_per_s"])
    '''


@pytest.mark.xfail
def test_release_control_ts_without_json():
    """
    Time series calculation with directly created pandapipes network.

    """
    # get net
    net = nw.release_control_test_network(use_time_series=True)

    # preparation for run time series
    time_steps = range(4)

    log_variables = [
        ('res_junction', 'p_bar'), ('res_pipe', 'v_mean_m_per_s'),
        ('res_pipe', 'reynolds'), ('res_pipe', 'lambda'),
        ('res_sink', 'mdot_kg_per_s'), ('res_source', 'mdot_kg_per_s'),
        ('res_ext_grid', 'mdot_kg_per_s')]
    ow = OutputWriter(net, time_steps, output_path=None, log_variables=log_variables)

    # run time series
    pp.timeseries.run_time_series.run_timeseries_ppipe(net, time_steps, output_writer=ow)


if __name__ == "__main__":
    pytest.main([os.path.join(os.path.dirname(__file__), "release_control_test.py")])
