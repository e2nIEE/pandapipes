# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from pandapipes.multinet.control.run_control_multinet import prepare_run_ctrl, run_control
from pandapower.control.util.diagnostic import control_diagnostic
from pandapower.timeseries.run_time_series import get_recycle_settings, init_time_steps, init_output_writer, \
    output_writer_routine, print_progress_bar, cleanup, \
    run_loop

try:
    import pplog
except ImportError:
    import logging as pplog

logger = pplog.getLogger(__name__)
logger.setLevel(level=pplog.WARNING)


def _call_output_writer(multinet, time_step, pf_converged, ctrl_converged, ts_variables):
    for net_name in multinet['nets'].keys():
        net = multinet['nets'][net_name]
        output_writer_routine(net, time_step, pf_converged, ctrl_converged, ts_variables[net_name]["recycle_options"])


def init_time_series(multinet, time_steps, continue_on_divergence=False, verbose=True,
                     **kwargs):
    """
    inits the time series calculation
    creates the dict ts_variables, which includes necessary variables for the time series / control function

    INPUT:
        **net** - The pandapower format network
        **time_steps** (list or tuple, None) - time_steps to calculate as list or tuple (start, stop)
                                                if None, all time steps from provided data source are simulated
    OPTIONAL:
        **output_writer** - A predefined output writer. If None the a default one is created with
                            get_default_output_writer()
        **continue_on_divergence** (bool, False) - If True time series calculation continues in case of errors.
        **verbose** (bool, True) - prints progess bar or logger debug messages
    """

    time_steps = init_time_steps(multinet, time_steps, **kwargs)
    run = kwargs.get('run', None)

    ts_variables = prepare_run_ctrl(multinet, None)

    for net_name in multinet['nets'].keys():
        net = multinet['nets'][net_name]
        recycle_options = None
        if hasattr(run, "__name__") and run.__name__ == "runpp":
            # use faster runpp options if possible
            recycle_options = get_recycle_settings(net, **kwargs)
        ts_variables[net_name]['run'] = run['net_name'] if run is not None else ts_variables[net_name]['run']
        ts_variables[net_name]['recycle_options'] = recycle_options
        init_output_writer(net, time_steps)

    # time steps to be calculated (list or range)
    ts_variables["time_steps"] = time_steps
    # If True, a diverged run is ignored and the next step is calculated
    ts_variables["continue_on_divergence"] = continue_on_divergence
    # print settings
    ts_variables["verbose"] = verbose

    if logger.level is not 10 and verbose:
        # simple progress bar
        print_progress_bar(0, len(time_steps), prefix='Progress:', suffix='Complete', length=50)

    return ts_variables


def run_timeseries(multinet, time_steps=None, continue_on_divergence=False,
                   verbose=True, **kwargs):
    """
    Time Series main function
    Runs multiple PANDAPOWER AC power flows based on time series in controllers
    Optionally other functions than the pp power flow can be called by setting the run function in kwargs

    INPUT:
        **net** - The pandapower format network

    OPTIONAL:
        **time_steps** (list or tuple, None) - time_steps to calculate as list or tuple (start, stop)
                                                if None, all time steps from provided data source are simulated
        **output_writer** - A predefined output writer. If None the a default one is created with
                            get_default_output_writer()
        **continue_on_divergence** (bool, False) - If True time series calculation continues in case of errors.
        **verbose** (bool, True) - prints progress bar or if logger.level == Debug it prints debug messages
        **kwargs** - Keyword arguments for run_control and runpp
    """

    ts_variables = init_time_series(multinet, time_steps, continue_on_divergence, verbose, **kwargs)

    for net_name in multinet['nets'].keys():
        control_diagnostic(multinet['nets'][net_name])

    run_loop(multinet, ts_variables, run_control, _call_output_writer, **kwargs)

    # cleanup functions after the last time step was calculated
    for net_name in multinet['nets'].keys():
        cleanup(ts_variables[net_name])
