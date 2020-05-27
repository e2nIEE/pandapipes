# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import tempfile

import pandapipes as ppipes
from pandapipes.pipeflow import PipeflowNotConverged
from pandapower.control.util.diagnostic import control_diagnostic
from pandapower.timeseries.output_writer import OutputWriter
from pandapower.timeseries.run_time_series import init_time_series as init_time_series_pp, cleanup, run_loop

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.WARNING)


def init_default_outputwriter(net, time_steps, **kwargs):
    """
    Creates a default output writer for the time series calculation.

    :param net: The pandapipes format network
    :type net: pandapipesNet
    :param timesteps: timesteps to calculate as list
    :type timesteps: list
    :return: output_writer - The default output_writer
    :rtype: ?
    """
    output_writer = kwargs.get("output_writer", None)
    if output_writer is not None:
        # write the output_writer to net
        logger.warning("deprecated: output_writer should not be given to run_timeseries(). "
                       "This overwrites the stored one in net.output_writer.")
        net.output_writer.iat[0, 0] = output_writer
    if "output_writer" not in net or net.output_writer.iat[0, 0] is None:
        ow = OutputWriter(net, time_steps, output_path=tempfile.gettempdir(), log_variables=[])
        ow.log_variable('res_sink', 'mdot_kg_per_s')
        ow.log_variable('res_source', 'mdot_kg_per_s')
        ow.log_variable('res_ext_grid', 'mdot_kg_per_s')
        ow.log_variable('res_pipe', 'v_mean_m_per_s')
        ow.log_variable('res_junction', 'p_bar')
        ow.log_variable('res_junction', 't_k')
        logger.info("No output writer specified. Using default:")
        logger.info(ow)


def pf_not_converged(time_step, ts_variables):
    """
    Todo: Fill out parameters.

    :param time_step: time_step to be calculated
    :type time_step: int
    :param ts_variables: contains settings for controller and time series simulation. \n
                                  See init_time_series()
    :type ts_variables: dict
    :return:
    :rtype:
    """
    logger.error('PipeflowNotConverged at time step %s' % time_step)
    if not ts_variables["continue_on_divergence"]:
        raise PipeflowNotConverged


def init_time_series(net, time_steps, output_writer=None, continue_on_divergence=False,
                     verbose=True, **kwargs):
    """
    Inits the time series calculation.
    Creates the dict ts_variables, which includes necessary variables for the time series / control
    function.

    :param net: The pandapipes format network
    :type net: pandapipesNet
    :param time_steps: time_steps to calculate as list or tuple (start, stop) if None, all time
                        steps from provided data source are simulated
    :type time_steps: list or tuple
    :param output_writer: A predefined output writer. If None the a default one is created with
                            get_default_output_writer()
    :type output_writer: ?, default None
    :param continue_on_divergence: If True time series calculation continues in case of errors.
    :type continue_on_divergence: bool, default False
    :param verbose: prints progess bar or logger debug messages
    :type verbose: bool, default True
    :param kwargs:
    :type kwargs:
    :return:
    :rtype:
    """

    run = kwargs.get("run", ppipes.pipeflow)
    init_default_outputwriter(net, time_steps, **kwargs)

    ts_variables = init_time_series_pp(net, time_steps, continue_on_divergence, verbose, run=run, **kwargs)

    return ts_variables, kwargs


def run_timeseries(net, time_steps=None, output_writer=None, continue_on_divergence=False,
                   verbose=True, **kwargs):
    """
    Time Series main function
    Runs multiple PANDAPOWER AC power flows based on time series in controllers
    Optionally other functions than the pp power flow can be called by setting the run function in
    kwargs.
    **NOTE: refers to pandapower power flow.
    :param net: The pandapipes format network
    :type net: pandapipesNet
    :param time_steps: time_steps to calculate as list or tuple(start, stop) if None, all time steps
                        from provided data source are simulated
    :type time_steps: list or tuple, default None
    :param output_writer: A predefined output writer. If None the a default one is created with
                            get_default_output_writer()
    :type output_writer: ?, default None
    :param continue_on_divergence: If True time series calculation continues in case of errors.
    :type continue_on_divergence: bool, default False
    :param verbose: prints progress bar or if logger.level == Debug it prints debug  messages
    :type verbose: bool, default True
    :param kwargs: Keyword arguments for run_control and runpp
    :return: No output.
    """
    ts_variables, kwargs = init_time_series(net, time_steps, output_writer,
                                            continue_on_divergence, verbose, **kwargs)

    control_diagnostic(net)
    run_loop(net, ts_variables, **kwargs)

    # cleanup functions after the last time step was calculated
    cleanup(ts_variables)
