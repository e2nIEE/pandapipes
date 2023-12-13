# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import tempfile

from pandapower.control import NetCalculationNotConverged

from pandapipes.pipeflow import PipeflowNotConverged, pipeflow
from pandapower.control.util.diagnostic import control_diagnostic
from pandapower.timeseries.output_writer import OutputWriter
from pandapower.timeseries.run_time_series import init_time_series as init_time_series_pp, cleanup,\
    run_loop

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.WARNING)


def init_default_outputwriter(net, time_steps, **kwargs):
    """
    Creates a default output writer for the time series calculation.

    :param net: The pandapipes format network
    :type net: pandapipesNet
    :param time_steps: Time steps to calculate as list
    :type time_steps: list
    :return: ow - The default output writer
    :rtype: pandapower.timeseries.output_writer.OutputWriter
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

    :param time_step: Time step to be calculated
    :type time_step: int
    :param ts_variables: Contains settings for controller and time series simulation. \n
                         See init_time_series()
    :type ts_variables: dict
    :return: No output
    """
    logger.error('PipeflowNotConverged at time step %s' % time_step)
    if not ts_variables["continue_on_divergence"]:
        raise PipeflowNotConverged


def init_time_series(net, time_steps, continue_on_divergence=False, verbose=True, **kwargs):
    """
    Initializes the time series calculation.

    Creates the dict ts_variables, which includes necessary variables for the time series /
    control function.

    :param net: The pandapipes format network
    :type net: pandapipesNet
    :param time_steps: Time steps to calculate as list or tuple (start, stop). If None, all time
                       steps from provided data source are simulated.
    :type time_steps: list or tuple
    :param continue_on_divergence: If True, time series calculation continues in case of errors.
    :type continue_on_divergence: bool, default False
    :param verbose: Prints progress bar or logger debug messages
    :type verbose: bool, default True
    :param kwargs: Keyword arguments for run_control and runpp
    :type kwargs: dict
    :return: ts_variables, kwargs
    :rtype: dict, dict
    """

    run = kwargs.pop("run", pipeflow)
    init_default_outputwriter(net, time_steps, **kwargs)

    ts_variables = init_time_series_pp(net, time_steps, continue_on_divergence, verbose, run=run,
                                       **kwargs)

    ts_variables["errors"] = tuple([PipeflowNotConverged, NetCalculationNotConverged])

    return ts_variables


def run_timeseries(net, time_steps=None, continue_on_divergence=False, verbose=True, **kwargs):
    """
    Time Series main function

    Execution of pipe flow calculations for a time series using controllers.
    Optionally other functions than pipeflow can be called by setting the run function in kwargs.

    .. note:: Refers to pandapower power flow.

    :param net: The pandapipes format network
    :type net: pandapipesNet
    :param time_steps: Time steps to calculate as list or tuple (start, stop). If None, all time \
            steps from provided data source are simulated.
    :type time_steps: list or tuple, default None
    :param continue_on_divergence: If True, time series calculation continues in case of errors.
    :type continue_on_divergence: bool, default False
    :param verbose: Prints progress bar or if *logger.level == Debug*, it prints debug messages
    :type verbose: bool, default True
    :param kwargs: Keyword arguments for run_control and runpp
    :type kwargs: dict
    :return: No output
    """
    ts_variables = init_time_series(net, time_steps, continue_on_divergence, verbose, **kwargs)

    control_diagnostic(net)
    run_loop(net, ts_variables, **kwargs)

    # cleanup functions after the last time step was calculated
    cleanup(net, ts_variables)
