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
from types import MethodType
import matplotlib


class OutputWriterTransient(OutputWriter):
    def _save_single_xls_sheet(self, append):
        raise NotImplementedError("Sorry not implemented yet")

    def _init_log_variable(self, net, table, variable, index=None, eval_function=None,
                           eval_name=None):
        if table == "res_internal":
            index = np.arange(len(net.junction) + net.pipe.sections.sum() - len(net.pipe))
        return super()._init_log_variable(net, table, variable, index, eval_function, eval_name)


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
    ow = OutputWriterTransient(net, time_steps, output_path=ow_path, log_variables=log_variables)
    return ow


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
sections = 36
nodes = 4
length = 1
k_mm = 0.1  # 0.0472

pp.create_pipe_from_parameters(net, j1, j2, length, 75e-3, k_mm=k_mm, sections=sections,
                               alpha_w_per_m2k=5, text_k=293.15)
pp.create_pipe_from_parameters(net, j2, j3, length, 75e-3, k_mm=k_mm, sections=sections,
                               alpha_w_per_m2k=5, text_k=293.15)
pp.create_pipe_from_parameters(net, j2, j4, length, 75e-3, k_mm=k_mm, sections=sections,
                               alpha_w_per_m2k=5, text_k=293.15)

time_steps = range(100)
dt = 60
iterations = 20
ow = _output_writer(net, time_steps, ow_path=tempfile.gettempdir())
run_timeseries(net, time_steps, dynamic_sim=True, transient=transient_transfer, mode="all", dt=dt,
               reuse_internal_data=True, iter=iterations)

res_T = ow.np_results["res_internal.t_k"]

pipe1 = np.zeros(((sections + 1), res_T.shape[0]))
pipe2 = np.zeros(((sections + 1), res_T.shape[0]))
pipe3 = np.zeros(((sections + 1), res_T.shape[0]))

pipe1[0, :] = copy.deepcopy(res_T[:, 0])
pipe1[-1, :] = copy.deepcopy(res_T[:, 1])
pipe2[0, :] = copy.deepcopy(res_T[:, 1])
pipe2[-1, :] = copy.deepcopy(res_T[:, 2])
pipe3[0, :] = copy.deepcopy(res_T[:, 1])
pipe3[-1, :] = copy.deepcopy(res_T[:, 3])
pipe1[1:-1, :] = np.transpose(copy.deepcopy(res_T[:, nodes:nodes + (sections - 1)]))
pipe2[1:-1, :] = np.transpose(
    copy.deepcopy(res_T[:, nodes + (sections - 1):nodes + (2 * (sections - 1))]))
pipe3[1:-1, :] = np.transpose(
    copy.deepcopy(res_T[:, nodes + (2 * (sections - 1)):nodes + (3 * (sections - 1))]))


plt.ion()

fig = plt.figure()
ax = fig.add_subplot(221)
ax1 = fig.add_subplot(222)
ax2 = fig.add_subplot(223)
ax.set_title("Pipe 1")
ax1.set_title("Pipe 2")
ax2.set_title("Pipe 3")
ax.set_ylabel("Temperature [K]")
ax1.set_ylabel("Temperature [K]")
ax2.set_ylabel("Temperature [K]")
ax.set_xlabel("Length coordinate [m]")
ax1.set_xlabel("Length coordinate [m]")
ax2.set_xlabel("Length coordinate [m]")

show_timesteps = [10, 25, 40]
line1, = ax.plot(np.arange(0, sections + 1, 1) * length * 1000 / sections, pipe1[:, show_timesteps[0]], color="black",
                 marker="+", label="Time step " + str(show_timesteps[0]), linestyle="dashed")
line11, = ax.plot(np.arange(0, sections + 1, 1) * length * 1000 / sections, pipe1[:, show_timesteps[1]], color="black",
                  linestyle="dotted", label="Time step " + str(show_timesteps[1]))
line12, = ax.plot(np.arange(0, sections + 1, 1) * length * 1000 / sections, pipe1[:, show_timesteps[2]], color="black",
                  linestyle="dashdot", label="Time step" + str(show_timesteps[2]))

line2, = ax1.plot(np.arange(0, sections + 1, 1) * length * 1000 / sections, pipe2[:, show_timesteps[0]], color="black",
                  marker="+", linestyle="dashed")
line21, = ax1.plot(np.arange(0, sections + 1, 1) * length * 1000 / sections, pipe2[:, show_timesteps[1]], color="black",
                   marker="+", linestyle="dotted")
line22, = ax1.plot(np.arange(0, sections + 1, 1) * length * 1000 / sections, pipe2[:, show_timesteps[2]], color="black",
                   marker="+", linestyle="dashdot")

line3, = ax2.plot(np.arange(0, sections + 1, 1) * length * 1000 / sections, pipe3[:, show_timesteps[0]], color="black",
                  marker="+", linestyle="dashed")
line31, = ax2.plot(np.arange(0, sections + 1, 1) * length * 1000 / sections, pipe3[:, show_timesteps[1]], color="black",
                   marker="+", linestyle="dotted")
line32, = ax2.plot(np.arange(0, sections + 1, 1) * length * 1000 / sections, pipe3[:, show_timesteps[2]], color="black",
                   marker="+", linestyle="dashdot")


# if sections == 4:
#     datap1 = pd.read_csv(os.path.join(os.getcwd(), 'pandapipes', 'pandapipes', 'files', 'Temperature.csv'),
#                                       sep=';',
#                                       header=1, nrows=5, keep_default_na=False)
#     datap2 = pd.read_csv(os.path.join(os.getcwd(), 'pandapipes', 'pandapipes', 'files', 'Temperature.csv'),
#                                       sep=';',
#                                       header=8, nrows=5, keep_default_na=False)
#     datap3 = pd.read_csv(os.path.join(os.getcwd(), 'pandapipes', 'pandapipes', 'files', 'Temperature.csv'),
#                                       sep=';',
#                                       header=15, nrows=5, keep_default_na=False)
#
#     d1 = ax.plot(np.arange(0, sections + 1, 1) * 1000 / sections, datap1["T"], color="black")
#     d2 = ax1.plot(np.arange(0, sections + 1, 1) * 1000 / sections, datap2["T"], color="black")
#     d3 = ax2.plot(np.arange(0, sections + 1, 1) * 1000 / sections, datap3["T"], color="black")

ax.set_ylim((280, 335))
ax1.set_ylim((280, 335))
ax2.set_ylim((280, 335))
ax.legend()
fig.canvas.draw()
plt.show()
