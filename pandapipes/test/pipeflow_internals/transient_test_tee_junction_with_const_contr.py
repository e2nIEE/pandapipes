import pandapipes as pp
import numpy as np
import copy
import matplotlib.pyplot as plt
import time
import tempfile
# create empty net
import pandas as pd
import os
import pandapower.control as control
from pandapipes.component_models import Pipe
from pandapipes.timeseries import run_timeseries, init_default_outputwriter
from pandapower.timeseries import OutputWriter, DFData
from pandapipes import pp_dir
from types import MethodType
import matplotlib
from datetime import datetime


class OutputWriterTransient(OutputWriter):
    def _save_single_xls_sheet(self, append):
        raise NotImplementedError("Sorry not implemented yet")

    def _init_log_variable(self, net1, table, variable, index=None, eval_function=None,
                           eval_name=None):
        if table == "res_internal":
            index = np.arange(net1.pipe.sections.sum() - 1)
        return super()._init_log_variable(net1, table, variable, index, eval_function, eval_name)


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

    if transient_transfer:
        log_variables = [
            ('res_junction', 't_k'), ('res_junction', 'p_bar'), ('res_pipe', 't_to_k'), ('res_internal', 't_k')
        ]
    else:
        log_variables = [
            ('res_junction', 't_k'), ('res_junction', 'p_bar'), ('res_pipe', 't_to_k')
        ]
    owtrans = OutputWriterTransient(net, time_steps, output_path=ow_path, output_file_type=".csv",
                                    log_variables=log_variables)
    return owtrans

def _prepare_net(net):
    """
        Writing the DataSources of sinks and sources to the net with ConstControl.

        :param net: Previously created or loaded pandapipes network
        :type net: pandapipesNet
        :return: Prepared network for time series simulation
        :rtype: pandapipesNet
        """

    ds_sink, ds_ext_grid = _data_source()
    control.ConstControl(net, element='sink', variable='mdot_kg_per_s',
                         element_index=net.sink.index.values, data_source=ds_sink,
                         profile_name=net.sink.index.values.astype(str))

    control.ConstControl(net, element='ext_grid', variable='t_k',
                         element_index=net.ext_grid.index.values,
                         data_source=ds_ext_grid, profile_name=net.ext_grid.index.values.astype(str))
    control.ConstControl(net, element='sink', variable='mdot_kg_per_s',
                         element_index=net.sink.index.values,
                         data_source=ds_sink, profile_name=net.sink.index.values.astype(str))
    return net


def _data_source():
    """
    Read out existing time series (csv files) for sinks and sources.

    :return: Time series values from csv files for sink and source
    :rtype: DataFrame
    """
    profiles_sink = pd.read_csv(os.path.join(pp_dir, 'files',
                                             'tee_junction_timeseries_sinks.csv'), index_col=0)
    profiles_source = pd.read_csv(os.path.join(pp_dir, 'files',
                                               'tee_junction_timeseries_ext_grid.csv'), index_col=0)
    ds_sink = DFData(profiles_sink)
    ds_ext_grid = DFData(profiles_source)
    return ds_sink, ds_ext_grid


# define the network
transient_transfer = True

net = pp.create_empty_network(fluid="water")

# create junctions
j1 = pp.create_junction(net, pn_bar=1.05, tfluid_k=293.15, name="Junction 1")
j2 = pp.create_junction(net, pn_bar=1.05, tfluid_k=293.15, name="Junction 2")
j3 = pp.create_junction(net, pn_bar=1.05, tfluid_k=293.15, name="Junction 3")
j4 = pp.create_junction(net, pn_bar=1.05, tfluid_k=293.15, name="Junction 4")

# create junction elements
ext_grid = pp.create_ext_grid(net, junction=j1, p_bar=5, t_k=330, name="Grid Connection")
sink = pp.create_sink(net, junction=j3, mdot_kg_per_s=2, name="Sink")
sink = pp.create_sink(net, junction=j4, mdot_kg_per_s=2, name="Sink")

# create branch elements
sections = 3
nodes = 4
length = 0.1
k_mm = 0.0472

pp.create_pipe_from_parameters(net, j1, j2, length, 75e-3, k_mm=k_mm, sections=sections,
                               alpha_w_per_m2k=5, text_k=293.15)
pp.create_pipe_from_parameters(net, j2, j3, length, 75e-3, k_mm=k_mm, sections=sections,
                               alpha_w_per_m2k=5, text_k=293.15)
pp.create_pipe_from_parameters(net, j2, j4, length, 75e-3, k_mm=k_mm, sections=sections,
                               alpha_w_per_m2k=5, text_k=293.15)

# prepare grid with controllers
_prepare_net(net)

# define time steps, iterations and output writer for time series simulation
dt = 1
time_steps = range(10)
iterations = 20
output_directory = os.path.join(tempfile.gettempdir()) #, "time_series_example"
ow = _output_writer(net, len(time_steps), ow_path=output_directory)

# run the time series
run_timeseries(net, time_steps, transient=transient_transfer, mode="all", iter=iterations, dt=dt)


# print and plot results

print("v: ", net.res_pipe.loc[0, "v_mean_m_per_s"])
print("timestepsreq: ", ((length * 1000) / net.res_pipe.loc[0, "v_mean_m_per_s"]) / dt)

if transient_transfer:
    res_T = ow.np_results["res_internal.t_k"]
else:
    res_T = ow.np_results["res_junction.t_k"]

# pipe1 = np.zeros(((sections + 1), res_T.shape[0]))
# pipe2 = np.zeros(((sections + 1), res_T.shape[0]))
# pipe3 = np.zeros(((sections + 1), res_T.shape[0]))
#
# pipe1[0, :] = copy.deepcopy(res_T[:, 0])
# pipe1[-1, :] = copy.deepcopy(res_T[:, 1])
# pipe2[0, :] = copy.deepcopy(res_T[:, 1])
# pipe2[-1, :] = copy.deepcopy(res_T[:, 2])
# pipe3[0, :] = copy.deepcopy(res_T[:, 1])
# pipe3[-1, :] = copy.deepcopy(res_T[:, 3])
# if transient_transfer:
#     pipe1[1:-1, :] = np.transpose(copy.deepcopy(res_T[:, nodes:nodes + (sections - 1)]))
#     pipe2[1:-1, :] = np.transpose(
#         copy.deepcopy(res_T[:, nodes + (sections - 1):nodes + (2 * (sections - 1))]))
#     pipe3[1:-1, :] = np.transpose(
#         copy.deepcopy(res_T[:, nodes + (2 * (sections - 1)):nodes + (3 * (sections - 1))]))
#
# # datap1 = pd.read_csv(os.path.join(pp_dir, 'Temperature.csv'),
# #                      sep=';',
# #                      header=1, nrows=5, keep_default_na=False)
# # datap2 = pd.read_csv(os.path.join(pp_dir, 'Temperature.csv'),
# #                      sep=';',
# #                      header=8, nrows=5, keep_default_na=False)
# # datap3 = pd.read_csv(os.path.join(pp_dir, 'Temperature.csv'),
# #                      sep=';',
# #                      header=15, nrows=5, keep_default_na=False)
#
# from IPython.display import clear_output
#
# plt.ion()
#
# fig = plt.figure()
# ax = fig.add_subplot(221)
# ax1 = fig.add_subplot(222)
# ax2 = fig.add_subplot(223)
# ax.set_title("Pipe 1")
# ax1.set_title("Pipe 2")
# ax2.set_title("Pipe 3")
# ax.set_ylabel("Temperature [K]")
# ax1.set_ylabel("Temperature [K]")
# ax2.set_ylabel("Temperature [K]")
# ax.set_xlabel("Length coordinate [m]")
# ax1.set_xlabel("Length coordinate [m]")
# ax2.set_xlabel("Length coordinate [m]")
#
# show_timesteps = [1, 5, 9]
#
# line1, = ax.plot(np.arange(0, sections + 1, 1) * length * 1000 / sections, pipe1[:, show_timesteps[0]], color="black",
#                  marker="+", label="Time step " + str(show_timesteps[0]), linestyle="dashed")
# line11, = ax.plot(np.arange(0, sections + 1, 1) * length * 1000 / sections, pipe1[:, show_timesteps[1]], color="black",
#                   linestyle="dotted", label="Time step " + str(show_timesteps[1]))
# line12, = ax.plot(np.arange(0, sections + 1, 1) * length * 1000 / sections, pipe1[:, show_timesteps[2]], color="black",
#                   linestyle="dashdot", label="Time step" + str(show_timesteps[2]))
#
# line2, = ax1.plot(np.arange(0, sections + 1, 1) * length * 1000 / sections, pipe2[:, show_timesteps[0]], color="black",
#                   marker="+", linestyle="dashed")
# line21, = ax1.plot(np.arange(0, sections + 1, 1) * length * 1000 / sections, pipe2[:, show_timesteps[1]], color="black",
#                    linestyle="dotted")
# line22, = ax1.plot(np.arange(0, sections + 1, 1) * length * 1000 / sections, pipe2[:, show_timesteps[2]], color="black",
#                    linestyle="dashdot")
#
# # line3, = ax2.plot(np.arange(0, sections + 1, 1) * length * 1000 / sections, pipe3[:, show_timesteps[0]], color="black",
# #                   marker="+", linestyle="dashed")
# # line31, = ax2.plot(np.arange(0, sections + 1, 1) * length * 1000 / sections, pipe3[:, show_timesteps[1]], color="black",
# #                    linestyle="dotted")
# # line32, = ax2.plot(np.arange(0, sections + 1, 1) * length * 1000 / sections, pipe3[:, show_timesteps[2]], color="black",
# #                    linestyle="dashdot")
#
# # if length == 1 and sections == 4 and k_mm == 0.1 and dt == 60:
#     # d1 = ax.plot(np.arange(0, sections + 1, 1) * length * 1000 / sections, datap1["T"], color="black")
#     # d2 = ax1.plot(np.arange(0, sections + 1, 1) * length * 1000 / sections, datap2["T"], color="black")
#     # d3 = ax2.plot(np.arange(0, sections + 1, 1) * length * 1000 / sections, datap3["T"], color="black")
#
# ax.set_ylim((280, 335))
# ax1.set_ylim((280, 335))
# ax2.set_ylim((280, 335))
# ax.legend()
# fig.canvas.draw()
# plt.show()
# # plt.savefig("files/output/tee_junction"+"sections"+str(sections)+"total_length_m"+str(length*1000)+"dt"+str(dt)+"iter"+str(iterations)+"k_mm"+str(k_mm)+".png") #+datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
#
# for phase in time_steps:
#     line1.set_ydata(pipe1[:, phase])
#     line2.set_ydata(pipe2[:, phase])
#     #    line3.set_ydata(pipe3[:, phase])
#     fig.canvas.draw()
#     fig.canvas.flush_events()
#     plt.pause(.01)


# print("Results can be found in your local temp folder: {}".format(output_directory))
