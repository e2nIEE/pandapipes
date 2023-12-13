# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
import pandas as pd
import pytest

import pandapipes
import pandapower
from pandapipes.multinet.control.controller.multinet_control import coupled_p2g_const_control
from pandapipes.multinet.create_multinet import create_empty_multinet, add_nets_to_multinet
from pandapipes.multinet.timeseries.run_time_series_multinet import run_timeseries
from pandapipes.test import runpp_with_mark, pipeflow_with_mark
from pandapipes.test.multinet.test_control_multinet import get_gas_example, get_power_example_simple
from pandapower.control.controller.const_control import ConstControl
from pandapower.timeseries.data_sources.frame_data import DFData
from pandapower.timeseries.output_writer import OutputWriter


def test_time_series_p2g_control(get_gas_example, get_power_example_simple):
    net_gas = get_gas_example
    net_power = get_power_example_simple

    pandapipes.create_source(net_gas, 5, 0.003)
    pandapipes.create_sink(net_gas, 3, 0.003)
    pandapipes.create_sink(net_gas, 4, 0.003)

    pandapower.create_load(net_power, 6, 0.004)
    pandapower.create_load(net_power, 5, 0.004)
    pandapower.create_sgen(net_power, 4, 0.004)

    mn = create_empty_multinet('coupled net')

    add_nets_to_multinet(mn, power=net_power, gas=net_gas)

    data_p2g_power = pd.DataFrame(np.concatenate([np.array([1.0] * 5), np.array([0.75] * 2), np.array([1.0] * 3)]),
                                  columns=['load_p2g_power'])
    data_const_load = pd.DataFrame(np.concatenate([np.array([1.2] * 5), np.array([1.75] * 2), np.array([1.2] * 3)]),
                                   columns=['load_const'])
    data_const_sgen = pd.DataFrame(np.concatenate([np.array([0.8] * 5), np.array([0.9] * 2), np.array([0.8] * 3)]),
                                   columns=['sgen_const'])
    data_const_sink = np.zeros([10, 2])
    data_const_sink[:5, :] = [0.6, 0.4]
    data_const_sink[5:7, :] = [1.2, 0.3]
    data_const_sink[7:, :] = [0.6, 0.4]
    data_const_sink = pd.DataFrame(data_const_sink, columns=['sink_const_0', 'sink_const_1'])

    ds = DFData(data_p2g_power)
    _, p2g = coupled_p2g_const_control(mn, 0, 0, 0.6, initial_run=True, profile_name='load_p2g_power', data_source=ds)
    ds = DFData(data_const_sink)
    ConstControl(net_gas, 'sink', 'mdot_kg_per_s', [0, 1], profile_name=['sink_const_0', 'sink_const_1'],
                 data_source=ds)
    ds = DFData(data_const_load)
    ConstControl(net_power, 'load', 'p_mw', 1, profile_name=['load_const'], data_source=ds)
    ds = DFData(data_const_sgen)
    ConstControl(net_power, 'sgen', 'p_mw', 0, profile_name=['sgen_const'], data_source=ds)

    log_variables = [('res_source', 'mdot_kg_per_s'),
                     ('res_sink', 'mdot_kg_per_s')]

    ow_gas = OutputWriter(net_gas, range(10), log_variables=log_variables)

    log_variables = [('res_load', 'p_mw'),
                     ('res_sgen', 'p_mw')]

    ow_power = OutputWriter(net_power, range(10), log_variables=log_variables)
    run_timeseries(mn, range(10))

    gas_res = ow_gas.np_results
    power_res = ow_power.np_results
    assert np.all(gas_res['res_sink.mdot_kg_per_s'] == data_const_sink.values)
    assert np.all(gas_res['res_source.mdot_kg_per_s'] == data_p2g_power.values * \
                  p2g.conversion_factor_mw_to_kgps() * p2g.efficiency)
    assert np.all(power_res['res_load.p_mw'][:, 1] == data_const_load.values.T)
    assert np.all(power_res['res_sgen.p_mw'] == data_const_sgen.values)


def test_time_series_p2g_control_run_parameter(get_gas_example, get_power_example_simple):
    net_gas = get_gas_example
    net_power = get_power_example_simple

    pandapipes.create_source(net_gas, 5, 0.003)
    pandapipes.create_sink(net_gas, 3, 0.003)
    pandapipes.create_sink(net_gas, 4, 0.003)

    pandapower.create_load(net_power, 6, 0.004)
    pandapower.create_load(net_power, 5, 0.004)
    pandapower.create_sgen(net_power, 4, 0.004)

    mn = create_empty_multinet('coupled net')

    add_nets_to_multinet(mn, power=net_power, gas=net_gas)

    data_p2g_power = pd.DataFrame(np.concatenate([np.array([1.0] * 5), np.array([0.75] * 2), np.array([1.0] * 3)]),
                                  columns=['load_p2g_power'])
    data_const_load = pd.DataFrame(np.concatenate([np.array([1.2] * 5), np.array([1.75] * 2), np.array([1.2] * 3)]),
                                   columns=['load_const'])
    data_const_sgen = pd.DataFrame(np.concatenate([np.array([0.8] * 5), np.array([0.9] * 2), np.array([0.8] * 3)]),
                                   columns=['sgen_const'])
    data_const_sink = np.zeros([10, 2])
    data_const_sink[:5, :] = [0.6, 0.4]
    data_const_sink[5:7, :] = [1.2, 0.3]
    data_const_sink[7:, :] = [0.6, 0.4]
    data_const_sink = pd.DataFrame(data_const_sink, columns=['sink_const_0', 'sink_const_1'])

    ds = DFData(data_p2g_power)
    _, p2g = coupled_p2g_const_control(mn, 0, 0, 0.6, initial_run=True, profile_name='load_p2g_power', data_source=ds)
    ds = DFData(data_const_sink)
    ConstControl(net_gas, 'sink', 'mdot_kg_per_s', [0, 1], profile_name=['sink_const_0', 'sink_const_1'],
                 data_source=ds)
    ds = DFData(data_const_load)
    ConstControl(net_power, 'load', 'p_mw', 1, profile_name=['load_const'], data_source=ds)
    ds = DFData(data_const_sgen)
    ConstControl(net_power, 'sgen', 'p_mw', 0, profile_name=['sgen_const'], data_source=ds)

    log_variables = [('res_source', 'mdot_kg_per_s'),
                     ('res_sink', 'mdot_kg_per_s')]

    OutputWriter(net_gas, range(10), log_variables=log_variables)

    log_variables = [('res_load', 'p_mw'),
                     ('res_sgen', 'p_mw')]

    OutputWriter(net_power, range(10), log_variables=log_variables)
    run_timeseries(mn, range(1), run={"power": runpp_with_mark, "gas": pipeflow_with_mark})

    assert net_power.mark == "runpp"
    assert net_gas.mark == "pipeflow"


if __name__ == '__main__':
    pytest.main(['-xs', __file__])
