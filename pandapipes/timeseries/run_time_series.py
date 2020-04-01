# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import tempfile

import numpy as np

from pandapower import LoadflowNotConverged, OPFNotConverged
from pandapower.control.run_control import ControllerNotConverged, get_controller_order
from pandapower.control.util.diagnostic import control_diagnostic
from pandapower.timeseries.output_writer import OutputWriter
import pandapipes as ppipe
from pandapipes.control.run_control import check_for_initial_pipeflow, run_control_ppipe

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.WARNING)


def get_default_output_writer_ppipe(net, timesteps):
    """
    Creates a default output writer for the time series calculation.

    :param net: The pandapipes format network
    :type net: pandapipesNet
    :param timesteps: timesteps to calculate as list
    :type timesteps: list
    :return: output_writer - The default output_writer
    :rtype: ?
    """

    ow = OutputWriter(net, timesteps, output_path=tempfile.gettempdir(), log_variables=[])
    ow.log_variable('res_sink', 'mdot_kg_per_s')
    ow.log_variable('res_source', 'mdot_kg_per_s')
    ow.log_variable('res_ext_grid', 'mdot_kg_per_s')
    ow.log_variable('res_pipe', 'v_mean_m_per_s')
    ow.log_variable('res_junction', 'p_bar')
    ow.log_variable('res_junction', 't_k')
    return ow


def init_outputwriter_ppipe(net, time_steps, output_writer=None):
    """
    Initilizes output writer. If output_writer is None, default output_writer is created.

    :param net: The pandapipes format network
    :type net: pandapipesNet
    :param time_steps: timesteps to calculate as list
    :type time_steps: list
    :param output_writer: An output_writer
    :type output_writer: ?
    :return: output_writer - The initialized output_writer
    :rtype: ?
    """

    if output_writer is None:
        output_writer = get_default_output_writer_ppipe(net, time_steps)
        logger.info("No output writer specified. Using default which writes to: {}"
                    .format(output_writer.output_path))
    else:
        # inits output writer before time series calculation
        output_writer.time_steps = time_steps
        output_writer.init_all()
    return output_writer


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
     Call in a loop to create terminal progress bar.
    the idea was mentioned in :
    https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console

    :param iteration: Current iteration
    :type iteration: int
    :param total: total iterations
    :type total: int
    :param prefix: prefix string
    :type prefix: str
    :param suffix: suffix string
    :type suffix: str
    :param decimals: positive number of decimals in percent complete
    :type decimals: int
    :param length: character length of bar
    :type length: int
    :param fill: bar fill character
    :type fill: str
    :return: No output.
    """

    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    # logger.info('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix))
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end="")
    # Print New Line on Complete
    if iteration == total:
        print("\n")


def controller_not_converged(net, time_step, ts_variables):
    """
    Todo: Fill out parameters.

    :param net:
    :type net:
    :param time_step:
    :type time_step:
    :param ts_variables:
    :type ts_variables:
    :return:
    :rtype:
    """

    logger.error('ControllerNotConverged at time step %s' % time_step)
    if not ts_variables["continue_on_divergence"]:
        raise ControllerNotConverged


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
        raise LoadflowNotConverged


def run_time_step(net, time_step, ts_variables, **kwargs):
    """
    Time Series step function
    Should be called to run the PANDAPOWER AC power flows based on time series in controllers
    (or other functions).
    **NOTE: Description refers to pandapower power flow.

    :param net: The pandapipes format network
    :type net: pandapipesNet
    :param time_step: time_step to be calculated
    :type time_step: int
    :param ts_variables: contains settings for controller and time series simulation. \n
                                  See init_time_series()
    :type ts_variables: dict
    :param kwargs:
    :return: No output.
    """

    ctrl_converged = True
    pf_converged = True
    output_writer = ts_variables["output_writer"]
    # update time step for output writer
    output_writer.time_step = time_step
    # run time step function for each controller
    for levelorder in ts_variables["controller_order"]:
        for ctrl in levelorder:
            ctrl.time_step(time_step)

    try:
        # calls controller init, control steps and run function (runpp usually is called in here)
        run_control_ppipe(net, ctrl_variables=ts_variables, **kwargs)
    except ControllerNotConverged:
        ctrl_converged = False
        # If controller did not converge do some stuff
        controller_not_converged(net, time_step, ts_variables)
    except (LoadflowNotConverged, OPFNotConverged):
        # If power flow did not converge simulation aborts or continues if continue_on_divergence
        # is True
        pf_converged = False
        pf_not_converged(time_step, ts_variables)

    # save
    output_writer.save_results(time_step, pf_converged=pf_converged, ctrl_converged=ctrl_converged)


def all_controllers_recycleable(net):
    """

    :param net:
    :type net:
    :return:
    :rtype:
    """
    # checks if controller are recycleable
    recycleable = np.alltrue(net["controller"]["recycle"].values)
    if not recycleable:
        logger.warning("recycle feature not supported by some controllers in net. I have to "
                       "deactive recycle")
    return recycleable


def get_run_function(**kwargs):
    """
    Todo: Fill out parameters.
    checks if "run" is specified in kwargs and calls this function in time series loop.
    if "recycle" is in kwargs we use the TimeSeriesRunpp class.

    :param kwargs:
    :type kwargs:
    :return: run - the run function to be called (default is pp.runpp())
             recycle_class - class to recycle implementation
    :rtype:
    """

    recycle_class = None

    if "run" in kwargs:
        run = kwargs.pop("run")
    else:
        run = ppipe.pipeflow
    return run, recycle_class


def init_time_steps_ppipe(net, time_steps, **kwargs):
    """
    Todo: Fill out parameters.

    :param net: The pandapipes format network
    :type net: pandapipesNet
    :param time_steps: time_steps to calculate as list
    :type time_steps:
    :param kwargs:
    :type kwargs:
    :return:
    :rtype:
    """
    # initializes time steps if as a range
    if not (isinstance(time_steps, list) or isinstance(time_steps, range)):
        if time_steps is None and ("start_step" in kwargs and "stop_step" in kwargs):
            logger.warning("start_step and stop_step are depricated. Please use a tuple like "
                           "time_steps = (start_step, stop_step) instead or a list")
            time_steps = range(kwargs["start_step"], kwargs["stop_step"] + 1)
        elif isinstance(time_steps, tuple):
            time_steps = range(time_steps[0], time_steps[1])
        else:
            logger.warning("No time steps to calculate are specified. I'll check the datasource of"
                           " the first controller for avaiable time steps")
            max_timestep = net.controller.loc[0].controller.data_source.get_time_steps_len()
            time_steps = range(max_timestep)
    return time_steps


def init_time_series_ppipe(net, time_steps, output_writer=None, continue_on_divergence=False,
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

    time_steps = init_time_steps_ppipe(net, time_steps, **kwargs)

    ts_variables = dict()

    output_writer = init_outputwriter_ppipe(net, time_steps, output_writer)
    level, order = get_controller_order(net)
    # use faster runpp if timeseries possible
    run, recycle_class = get_run_function(**kwargs)

    # True at default. Initial power flow is calculated before each control step
    # (some controllers need inits)
    ts_variables["initial_pipeflow"] = check_for_initial_pipeflow(order)
    ts_variables["initial_powerflow"] = ts_variables["initial_pipeflow"]
    # order of controller (controllers are called in a for loop.)
    ts_variables["controller_order"] = order
    # run function to be called in run_control - default is pp.runpp, but can be runopf or whatever
    # you like
    ts_variables["run"] = run
    # recycle class function, which stores some NR variables. Only used if recycle == True
    ts_variables["recycle_class"] = recycle_class
    # output writer, which logs information during the time series simulation
    ts_variables["output_writer"] = output_writer
    # time steps to be calculated (list or range)
    ts_variables["time_steps"] = time_steps
    # If True, a diverged power flow is ignored and the next step is calculated
    ts_variables["continue_on_divergence"] = continue_on_divergence

    if (logger.level != 10) and verbose:
        # simple progress bar
        print_progress_bar(0, len(time_steps), prefix='Progress:', suffix='Complete', length=50)

    if "recycle" in kwargs:
        kwargs.pop("recycle")

    return ts_variables, kwargs


def cleanup(ts_variables):
    """

    :param ts_variables:
    :type ts_variables:
    :return:
    :rtype:
    """
    if ts_variables["recycle_class"] is not None:
        ts_variables["recycle_class"].cleanup()


def print_progress(i, time_step, time_steps, verbose, **kwargs):
    """
    Todo: Fill out parameters.
    :param i:
    :type i:
    :param time_step:
    :type time_step:
    :param time_steps:
    :type time_steps:
    :param verbose:
    :type verbose:
    :param kwargs:
    :type kwargs:
    """
    # simple status print in each time step.
    if (logger.level != 10) and verbose:
        len_timesteps = len(time_steps)
        print_progress_bar(i + 1, len_timesteps, prefix='Progress:', suffix='Complete', length=50)

    # print debug info
    if logger.level == logging.DEBUG and verbose:
        logger.debug("run time step %i" % time_step)

    # print luigi pipeline progress
    if "luigi_progress" in kwargs and i % 365 == 0:
        # print only every 365 time steps
        message = kwargs["luigi_progress"]["message"]
        progress = kwargs["luigi_progress"]["progress"]
        len_timesteps = len(time_steps)
        message("Progress: %d / %d" % (i, len_timesteps))
        progress_percentage = int(((i + 1) / len_timesteps) * 100)
        progress(progress_percentage)


def run_timeseries_ppipe(net, time_steps=None, output_writer=None, continue_on_divergence=False,
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
    ts_variables, kwargs = init_time_series_ppipe(net, time_steps, output_writer,
                                                  continue_on_divergence, verbose, **kwargs)

    control_diagnostic(net)
    for i, time_step in enumerate(ts_variables["time_steps"]):
        print_progress(i, time_step, ts_variables["time_steps"], verbose, **kwargs)
        run_time_step(net, time_step, ts_variables, **kwargs)

    # cleanup functions after the last time step was calculated
    cleanup(ts_variables)
