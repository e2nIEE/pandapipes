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
service = True

net = pp.create_empty_network(fluid="water")
# create junctions
j1 = pp.create_junction(net, pn_bar=1.05, tfluid_k=293, name="Junction 1")
j2 = pp.create_junction(net, pn_bar=1.05, tfluid_k=293, name="Junction 2")
#j3 = pp.create_junction(net, pn_bar=1.05, tfluid_k=293, name="Junction 3")

# create junction elements
ext_grid = pp.create_ext_grid(net, junction=j1, p_bar=5, t_k=330, name="Grid Connection")
sink = pp.create_sink(net, junction=j2, mdot_kg_per_s=10, name="Sink")

# create branch elements
sections = 9
nodes = 2
length = 0.1
pp.create_pipe_from_parameters(net, j1, j2, length, 75e-3, k_mm=.0472, sections=sections,
                               alpha_w_per_m2k=5, text_k=293)
# pp.create_pipe_from_parameters(net, j2, j3, length, 75e-3, k_mm=.0472, sections=sections,
#                                 alpha_w_per_m2k=5, text_k=293)
# pp.create_valve(net, from_junction=j2, to_junction=j3, diameter_m=0.310, opened=True, loss_coefficient=4.51378671)

# read in csv files for control of sources/sinks

profiles_source = pd.read_csv(os.path.join('pandapipes/files',
                                           'heat_flow_source_timesteps.csv'),
                              index_col=0)

ds_source = DFData(profiles_source)

const_source = control.ConstControl(net, element='ext_grid', variable='t_k',
                                    element_index=net.ext_grid.index.values,
                                    data_source=ds_source,
                                    profile_name=net.ext_grid.index.values.astype(str),
                                    in_service=service)

dt = 5
time_steps = range(ds_source.df.shape[0])
ow = _output_writer(net, time_steps, ow_path=tempfile.gettempdir())
run_timeseries(net, time_steps, mode="all", transient=transient_transfer, iter=30, dt=dt)

if transient_transfer:
    res_T = ow.np_results["res_internal.t_k"]
else:
    res_T = ow.np_results["res_junctions.t_k"]
pipe1 = np.zeros(((sections + 1), res_T.shape[0]))

pipe1[0, :] = copy.deepcopy(res_T[:, 0])
pipe1[-1, :] = copy.deepcopy(res_T[:, 1])
if transient_transfer:
    pipe1[1:-1, :] = np.transpose(copy.deepcopy(res_T[:, nodes:nodes + (sections - 1)]))
print(pipe1) # columns: timesteps, rows: pipe segments

print("v: ", net.res_pipe.loc[0, "v_mean_m_per_s"])
print("timestepsreq: ", ((length * 1000) / net.res_pipe.loc[0, "v_mean_m_per_s"]) / dt)

print("net.res_pipe:")
print(net.res_pipe)
print("net.res_junction:")
print(net.res_junction)
if transient_transfer:
    print("net.res_internal:")
    print(net.res_internal)
print("net.res_ext_grid")
print(net.res_ext_grid)

x = time_steps
fig = plt.figure()
plt.xlabel("time step")
plt.ylabel("temperature at both junctions [K]")
plt.title("junction results temperature transient")
plt.plot(x, pipe1[0,:], "r-o")
plt.plot(x, pipe1[-1,:], "y-o")
if transient_transfer:
    plt.plot(x, pipe1[1, :], "g-o")
    plt.legend(["Junction 0", "Junction 1", "Section 1"])
else:
    plt.legend(["Junction 0", "Junction 1"])
plt.grid()
plt.savefig("files/output/one_pipe_temperature_step_transient.png")
plt.show()
plt.close()
