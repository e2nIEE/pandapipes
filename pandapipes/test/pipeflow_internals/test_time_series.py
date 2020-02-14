# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import os
import numpy as np
import pytest
import pandapower.control as control
import pandas as pd
from pandapipes import networks as nw
from pandapipes.timeseries.run_time_series import run_timeseries_ppipe, \
    get_default_output_writer_ppipe
from pandapower.timeseries import DFData
from pandapower.timeseries import OutputWriter
from pandapipes import pp_dir

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)

path = os.path.join(pp_dir, 'test', 'pipeflow_internals', 'data', 'test_time_series_results')


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

    :return:
    :rtype:
    """
    profiles_sink = pd.read_csv(os.path.join(pp_dir, 'test', 'pipeflow_internals', 'data',
                                             'test_time_series_sink_profiles.csv'), index_col=0)
    profiles_source = pd.read_csv(os.path.join(pp_dir, 'test', 'pipeflow_internals', 'data',
                                               'test_time_series_source_profiles.csv'), index_col=0)
    ds_sink = DFData(profiles_sink)
    ds_source = DFData(profiles_source)
    return ds_sink, ds_source


def _preparte_grid(net):
    """

    :param net:
    :type net:
    :return:
    :rtype:
    """

    ds_sink, ds_source = _data_source()
    const_sink = control.ConstControl(net, element='sink', variable='mdot_kg_per_s',
                                      element_index=net.sink.index.values, data_source=ds_sink,
                                      profile_name=net.sink.index.values.astype(str))
    const_source = control.ConstControl(net, element='source', variable='mdot_kg_per_s',
                                        element_index=net.source.index.values,
                                        data_source=ds_source,
                                        profile_name=net.source.index.values.astype(str))
    del const_sink.initial_powerflow
    const_sink.initial_pipeflow = False
    del const_source.initial_powerflow
    const_source.initial_pipeflow = False
    return net


def _compare_results(ow):
    test_res_ext_grid = pd.read_csv(os.path.join(
        pp_dir, 'test', 'pipeflow_internals', 'data', 'test_time_series_results', 'res_ext_grid',
        'mdot_kg_per_s.csv'), sep=';', index_col=0)
    res_ext_grid = ow.np_results["res_ext_grid.mdot_kg_per_s"]
    check = res_ext_grid.round(9) == test_res_ext_grid.values.round(9)
    assert (np.all(check))
    test_res_junction = pd.read_csv(os.path.join(
        pp_dir, 'test', 'pipeflow_internals', 'data', 'test_time_series_results', 'res_junction',
        'p_bar.csv'), sep=';', index_col=0)
    res_junction = ow.np_results["res_junction.p_bar"]
    check = res_junction.round(9) == test_res_junction.values.round(9)
    assert (np.all(check))
    test_res_pipe = pd.read_csv(os.path.join(
        pp_dir, 'test', 'pipeflow_internals', 'data', 'test_time_series_results', 'res_pipe',
        'v_mean_m_per_s.csv'), sep=';', index_col=0)
    res_pipe = ow.np_results["res_pipe.v_mean_m_per_s"]
    check = res_pipe.round(9) == test_res_pipe.values.round(9)
    assert (np.all(check))
    test_res_sink = pd.read_csv(os.path.join(
        pp_dir, 'test', 'pipeflow_internals', 'data', 'test_time_series_results', 'res_sink',
        'mdot_kg_per_s.csv'), sep=';', index_col=0)
    res_sink = ow.np_results["res_sink.mdot_kg_per_s"]
    check = res_sink.round(9) == test_res_sink.values.round(9)
    assert (np.all(check))
    test_res_source = pd.read_csv(os.path.join(
        pp_dir, 'test', 'pipeflow_internals', 'data', 'test_time_series_results',
        'res_source', 'mdot_kg_per_s.csv'), sep=';', index_col=0)
    res_source = ow.np_results["res_source.mdot_kg_per_s"]
    check = res_source.round(9) == test_res_source.values.round(9)
    assert (np.all(check))


def _output_writer(net, time_steps, path=None):
    """

    :param net:
    :type net:
    :param time_steps:
    :type time_steps:
    :param path:
    :type path:
    :return:
    :rtype:
    """
    log_variables = [
        ('res_junction', 'p_bar'), ('res_pipe', 'v_mean_m_per_s'),
        ('res_pipe', 'reynolds'), ('res_pipe', 'lambda'),
        ('res_sink', 'mdot_kg_per_s'), ('res_source', 'mdot_kg_per_s'),
        ('res_ext_grid', 'mdot_kg_per_s')]
    ow = OutputWriter(net, time_steps, output_path=path, log_variables=log_variables)
    return ow


def test_time_series():
    """

    :return:
    :rtype:
    """
    net = nw.gas_versatility()
    net = _preparte_grid(net)
    time_steps = range(25)
    ow = _output_writer(net, time_steps)  # , path=os.path.join(ppipe.pp_dir, 'results'))
    run_timeseries_ppipe(net, time_steps, output_writer=ow)
    _compare_results(ow)


def test_time_series_default_ow():
    """

    :return:
    :rtype:
    """
    net = nw.gas_versatility()
    net = _preparte_grid(net)
    time_steps = range(25)
    ow = get_default_output_writer_ppipe(net, time_steps)
    run_timeseries_ppipe(net, time_steps, output_writer=ow)
    _compare_results(ow)


if __name__ == "__main__":
    pytest.main(test_time_series())
