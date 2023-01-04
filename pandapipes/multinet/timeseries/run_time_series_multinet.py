# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import tqdm
from pandapower import pandapowerNet
from pandapower.control.util.diagnostic import control_diagnostic
from pandapower.timeseries.run_time_series import get_recycle_settings, init_time_steps, \
    output_writer_routine, cleanup, run_loop, init_default_outputwriter as init_default_ow_pp, \
    init_output_writer

from pandapipes import pandapipesNet
from pandapipes.multinet.control.run_control_multinet import prepare_run_ctrl, run_control
from pandapipes.timeseries.run_time_series import init_default_outputwriter as init_default_ow_pps

try:
    import pandaplan.core.pplog as pplog
except ImportError:
    import logging as pplog

logger = pplog.getLogger(__name__)
logger.setLevel(level=pplog.WARNING)


def _call_output_writer(multinet, time_step, pf_converged, ctrl_converged, ts_variables):
    """
    Calling the output writer routine for each net in multinet.

    :param multinet: multinet with multinet controllers, net distinct controllers and several pandapipes/pandapower nets
    :type multinet: pandapipes.Multinet
    :param time_step: the results of each time step, which shall be retrieved by the output writer
    :type time_step: sequence of array_like
    :param pf_converged: did powerflow converge
    :type pf_converged: bool
    :param ctrl_converged: did all controller converge
    :type ctrl_converged: bool
    :param ts_variables: contains all relevant information and boundaries required for time series and control analyses
    :type ts_variables: dict
    :return: calling each output writer in order to save the results which are retrieved
    :rtype: None
    """
    for net_name in multinet['nets'].keys():
        net = multinet['nets'][net_name]
        output_writer_routine(net, time_step, pf_converged, ctrl_converged, ts_variables['nets'][net_name]["recycle_options"])


def init_time_series(multinet, time_steps, continue_on_divergence=False, verbose=True,
                     **kwargs):
    """
    Initializes the time series calculation.
    Besides it creates the dict ts_variables, which includes necessary variables for the time series / control loop.

    :param multinet: multinet with multinet controllers, net distinct controllers and several pandapipes/pandapower nets
    :type multinet: pandapipes.Multinet
    :param time_steps: the number of times a time series calculation shall be conducted
    :type time_steps: sequence of array_like
    :param continue_on_divergence: What to do if loadflow/pipeflow is not converging, fires control_repair
    :type continue_on_divergence: bool, default: False
    :param verbose: prints progess bar or logger debug messages
    :type verbose: bool, default: True
    :param kwargs: additional keyword arguments handed to each run function
    :type kwargs: dict
    :return: ts_variables which contains all relevant information and boundaries required for time series and
    control analyses
    :rtype: dict
    """
    time_steps = init_time_steps(multinet, time_steps, **kwargs)
    run = kwargs.get('run', None)

    ts_variables = prepare_run_ctrl(multinet, **kwargs)

    for net_name in multinet['nets'].keys():
        net = multinet['nets'][net_name]
        if isinstance(net, pandapowerNet):
            init_default_ow_pp(net, time_steps, **kwargs)
        elif isinstance(net, pandapipesNet):
            init_default_ow_pps(net, time_steps, **kwargs)
        else:
            raise ValueError('the given nets are neither pandapipes nor pandapower nets')
        recycle_options = None
        if hasattr(run, "__name__") and run.__name__ == "runpp":
            # use faster runpp options if possible
            recycle_options = get_recycle_settings(net, **kwargs)
        ts_variables['nets'][net_name]['run'] = run[net_name] if run is not None else ts_variables['nets'][net_name]['run']
        ts_variables['nets'][net_name]['recycle_options'] = recycle_options
        init_output_writer(net, time_steps)

    # time steps to be calculated (list or range)
    ts_variables["time_steps"] = time_steps
    # If True, a diverged run is ignored and the next step is calculated
    ts_variables["continue_on_divergence"] = continue_on_divergence
    # print settings
    ts_variables["verbose"] = verbose

    if logger.level != 10 and verbose:
        # simple progress bar
        ts_variables['progress_bar'] = tqdm.tqdm(total=len(time_steps))

    return ts_variables


def run_timeseries(multinet, time_steps=None, continue_on_divergence=False,
                   verbose=True, **kwargs):
    """
    Time Series main function.
    Runs multiple run functions for each net in multinet. Within each time step several controller loops are conducted
    till all controllers and each net is converged.
    A normal pp.runpp/pps.pipeflow can be optionally replaced by other run functions by setting the run function in
    kwargs.

    :param multinet: multinet with multinet controllers, net distinct controllers and several pandapipes/pandapower nets
    :type multinet: pandapipes.Multinet
    :param time_steps: the number of times a time series calculation shall be conducted
    :type time_steps: sequence of array_like, default: None
    :param continue_on_divergence: What to do if loadflow/pipeflow is not converging, fires control_repair
    :type continue_on_divergence: bool, default: False
    :param verbose: prints progess bar or logger debug messages
    :type verbose: bool, default: True
    :param kwargs: additional keyword arguments handed to each run function
    :type kwargs: dict
    :return: runs the time series loop
    :rtype: None
    """
    ts_variables = init_time_series(multinet, time_steps, continue_on_divergence, verbose, **kwargs)

    for net_name in multinet['nets'].keys():
        control_diagnostic(multinet['nets'][net_name])

    run_loop(multinet, ts_variables, run_control, _call_output_writer, **kwargs)

    # cleanup functions after the last time step was calculated
    for net_name in multinet['nets'].keys():
        cleanup(multinet['nets'][net_name], ts_variables['nets'][net_name])
