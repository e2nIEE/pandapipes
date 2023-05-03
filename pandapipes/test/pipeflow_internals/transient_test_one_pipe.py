import pytest

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
from pandapipes.test.pipeflow_internals import internals_data_path
from types import MethodType


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
j1 = pp.create_junction(net, pn_bar=1.05, tfluid_k=293, name="Junction 1")
j2 = pp.create_junction(net, pn_bar=1.05, tfluid_k=293, name="Junction 2")

# create junction elements
ext_grid = pp.create_ext_grid(net, junction=j1, p_bar=5, t_k=330, name="Grid Connection")
sink = pp.create_sink(net, junction=j2, mdot_kg_per_s=2, name="Sink")

# create branch elements
sections = 74
nodes = 2
length = 1
pp.create_pipe_from_parameters(net, j1, j2, length, 75e-3, k_mm=.0472, sections=sections,
                               alpha_w_per_m2k=5, text_k=293)

# read in csv files for control of sources/sinks

time_steps = range(100)
dt = 60
iterations = 3000
ow = _output_writer(net, time_steps, ow_path=tempfile.gettempdir())
run_timeseries(net, time_steps, dynamic_sim=True, transient=transient_transfer, mode="all", dt=dt,
               reuse_internal_data=True, iter=iterations)

if transient_transfer:
    res_T = ow.np_results["res_internal.t_k"]
else:
    res_T = ow.np_results["res_junctions.t_k"]

res_T_df = pd.DataFrame(res_T)
res_T_df.to_excel('res_T.xlsx')

pipe1 = np.zeros(((sections + 1), res_T.shape[0]))

pipe1[0, :] = copy.deepcopy(res_T[:, 0])
pipe1[-1, :] = copy.deepcopy(res_T[:, 1])
if transient_transfer:
    pipe1[1:-1, :] = np.transpose(copy.deepcopy(res_T[:, nodes:nodes + (sections - 1)]))
# print(pipe1) # columns: timesteps, rows: pipe segments

plt.ion()

fig = plt.figure()
ax = fig.add_subplot(221)
ax.set_title("Pipe 1")
ax.set_ylabel("Temperature [K]")
ax.set_xlabel("Length coordinate [m]")

show_timesteps = [10, 30, 90]
line1, = ax.plot(np.arange(0, sections + 1, 1) * length * 1000 / sections, pipe1[:, show_timesteps[0]], color="black",
                 marker="+", label="Time step " + str(show_timesteps[0]), linestyle="dashed")
line11, = ax.plot(np.arange(0, sections + 1, 1) * length * 1000 / sections, pipe1[:, show_timesteps[1]], color="red",
                  linestyle="dotted", label="Time step " + str(show_timesteps[1]))
line12, = ax.plot(np.arange(0, sections + 1, 1) * length * 1000 / sections, pipe1[:, show_timesteps[2]], color="blue",
                  linestyle="dashdot", label="Time step " + str(show_timesteps[2]))

ax.set_ylim((280, 335))
ax.legend()
fig.canvas.draw()
plt.show()

print(net.res_internal)
