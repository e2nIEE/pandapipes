# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

import pandapower.control as control
from pandapipes import networks as nw
from pandapipes import pp_dir
from pandapipes.timeseries import run_timeseries, init_default_outputwriter
from pandapower.timeseries import OutputWriter, DFData

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)

path = os.path.join(pp_dir, 'test', 'pipeflow_internals', 'data', 'test_time_series_results')


def _prepare_grid(net):
    """
    Writing the DataSources of sinks and sources to the net with ConstControl.

    :param net: Previously created or loaded pandapipes network
    :type net: pandapipesNet
    :return: Prepared network for time series simulation
    :rtype: pandapipesNet
    """

    ds_sink, ds_source = _data_source()
    control.ConstControl(net, element='sink', variable='mdot_kg_per_s',
                         element_index=net.sink.index.values, data_source=ds_sink,
                         profile_name=net.sink.index.values.astype(str))
    control.ConstControl(
        net, element='source', variable='mdot_kg_per_s', element_index=net.source.index.values,
        data_source=ds_source, profile_name=net.source.index.values.astype(str))


def _save_profiles_csv(net):
    """

    :param net:
    :type net:
    :return:
    :rtype:
    """
    rand = np.random.random([25, len(net.sink)])
    profiles = net.sink.mdot_kg_per_s.values * rand
    profiles = pd.DataFrame(profiles, columns=[net.sink.index.values])
    profiles.to_csv(os.path.join(pp_dir, 'test', 'pipeflow_internals', 'data',
                                 'test_time_series_sink_profiles.csv'))

    rand = np.random.random([25, len(net.source)])
    profiles = net.source.mdot_kg_per_s.values * rand
    profiles = pd.DataFrame(profiles, columns=[net.source.index.values])
    profiles.to_csv(os.path.join(pp_dir, 'test', 'pipeflow_internals', 'data',
                                 'test_time_series_source_profiles.csv'))


def _data_source():
    """
    Read out existing time series (csv files) for sinks and sources.

    :return: Time series values from csv files for sink and source
    :rtype: DataFrame
    """
    profiles_sink = pd.read_csv(os.path.join(pp_dir, 'test', 'pipeflow_internals', 'data',
                                             'test_time_series_sink_profiles.csv'), index_col=0)
    profiles_source = pd.read_csv(os.path.join(pp_dir, 'test', 'pipeflow_internals', 'data',
                                               'test_time_series_source_profiles.csv'), index_col=0)
    ds_sink = DFData(profiles_sink)
    ds_source = DFData(profiles_source)
    return ds_sink, ds_source


def _compare_results(ow):
    test_res_ext_grid = pd.read_csv(os.path.join(
        pp_dir, 'test', 'pipeflow_internals', 'data', 'test_time_series_results', 'res_ext_grid',
        'mdot_kg_per_s.csv'), sep=';', index_col=0)
    res_ext_grid = ow.np_results["res_ext_grid.mdot_kg_per_s"]
    res_ext_grid = res_ext_grid[~np.isclose(res_ext_grid, 0)]
    test_res_ext_grid = test_res_ext_grid.values[~np.isclose(test_res_ext_grid.values, 0)]
    diff = 1 - res_ext_grid.round(9) / test_res_ext_grid.round(9)
    check = diff < 0.0001
    assert (np.all(check))
    test_res_junction = pd.read_csv(os.path.join(
        pp_dir, 'test', 'pipeflow_internals', 'data', 'test_time_series_results', 'res_junction',
        'p_bar.csv'), sep=';', index_col=0)
    res_junction = ow.np_results["res_junction.p_bar"]
    res_junction = res_junction[~np.isclose(res_junction, 0)]
    test_res_junction = test_res_junction.values[~np.isclose(test_res_junction.values, 0)]
    diff = 1 - res_junction.round(9) / test_res_junction.round(9)
    check = diff < 0.0001
    assert (np.all(check))
    test_res_pipe = pd.read_csv(os.path.join(
        pp_dir, 'test', 'pipeflow_internals', 'data', 'test_time_series_results', 'res_pipe',
        'v_mean_m_per_s.csv'), sep=';', index_col=0)
    res_pipe = ow.np_results["res_pipe.v_mean_m_per_s"]
    res_pipe = res_pipe[~np.isclose(res_pipe, 0)]
    test_res_pipe = test_res_pipe.values[~np.isclose(test_res_pipe.values, 0)]
    diff = 1 - res_pipe.round(9) / test_res_pipe.round(9)
    check = diff < 0.0001
    assert (np.all(check))
    test_res_sink = pd.read_csv(os.path.join(
        pp_dir, 'test', 'pipeflow_internals', 'data', 'test_time_series_results', 'res_sink',
        'mdot_kg_per_s.csv'), sep=';', index_col=0)
    res_sink = ow.np_results["res_sink.mdot_kg_per_s"]
    res_sink = res_sink[~np.isclose(res_sink, 0)]
    test_res_sink = test_res_sink.values[~np.isclose(test_res_sink.values, 0)]
    diff = 1 - res_sink.round(9) / test_res_sink.round(9)
    check = diff < 0.0001
    assert (np.all(check))
    test_res_source = pd.read_csv(os.path.join(
        pp_dir, 'test', 'pipeflow_internals', 'data', 'test_time_series_results',
        'res_source', 'mdot_kg_per_s.csv'), sep=';', index_col=0)
    res_source = ow.np_results["res_source.mdot_kg_per_s"]
    res_source = res_source[~np.isclose(res_source, 0)]
    test_res_source = test_res_source.values[~np.isclose(test_res_source.values, 0)]
    diff = 1 - res_source.round(9) / test_res_source.round(9)
    check = diff < 0.0001
    assert (np.all(check))


def _output_writer(net, time_steps, ow_path=None):
    """
    Creating an output writer.

    :param net: Prepared pandapipes net
    :type net: pandapipesNet
    :param time_steps: Time steps to calculate as a list or range
    :type time_steps: list, range
    :param ow_path: Path to a folder where the output is written to.
    :type ow_path: string, default None
    :return: Output writer
    :rtype: pandapower.timeseries.output_writer.OutputWriter
    """
    log_variables = [
        ('res_junction', 'p_bar'), ('res_pipe', 'v_mean_m_per_s'),
        ('res_pipe', 'reynolds'), ('res_pipe', 'lambda'),
        ('res_sink', 'mdot_kg_per_s'), ('res_source', 'mdot_kg_per_s'),
        ('res_ext_grid', 'mdot_kg_per_s')]
    ow = OutputWriter(net, time_steps, output_path=ow_path, log_variables=log_variables)
    return ow


def test_time_series():
    """

    :return:
    :rtype:
    """
    net = nw.gas_versatility()
    _prepare_grid(net)
    time_steps = range(25)
    # _output_writer(net, time_steps)  # , path=os.path.join(ppipe.pp_dir, 'results'))
    _output_writer(net, time_steps, ow_path=tempfile.gettempdir())
    run_timeseries(net, time_steps, calc_compression_power = False)
    ow = net.output_writer.iat[0, 0]
    _compare_results(ow)


def test_time_series_default_ow():
    """

    :return:
    :rtype:
    """
    net = nw.gas_versatility()
    _prepare_grid(net)
    time_steps = range(25)
    init_default_outputwriter(net, time_steps)
    run_timeseries(net, time_steps, calc_compression_power = False)
    ow = net.output_writer.iat[0, 0]
    _compare_results(ow)


if __name__ == "__main__":
    pytest.main(test_time_series())
