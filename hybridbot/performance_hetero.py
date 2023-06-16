

import pandapipes as pp
from pandapipes import pipeflow
from pandapipes.networks.simple_gas_networks import gas_3parallel, gas_meshed_delta, schutterwald
from pandapipes.plotting import simple_plot
#from pandapipes.plotting.plotly.simple_plotly import simple_plotly

net = pp.create_empty_network("net", "methane")

# create junction
j1 = pp.create_junction(net, pn_bar=1.05, tfluid_k=280, name="Junction 1")
j2 = pp.create_junction(net, pn_bar=1.05, tfluid_k=280, name="Junction 2")
j3 = pp.create_junction(net, pn_bar=1.05, tfluid_k=280, name="Junction 3")
j4 = pp.create_junction(net, pn_bar=1.05, tfluid_k=280, name="Junction 4")

# create junction elements
ext_grid = pp.create_ext_grid(net, fluid="methane", junction=j1, p_bar=1.1, t_k=293.15, name="Grid Connection")
sink = pp.create_sink(net, junction=j3, mdot_kg_per_s=0.045, name="Sink")
source = pp.create_source(net, junction=j4, mdot_kg_per_s=0.005, name="Source", fluid="hydrogen")
# create branch element
pipe = pp.create_pipe_from_parameters(net, from_junction=j1, to_junction=j2, length_km=0.1, diameter_m=0.05,
                                      name="Pipe 1")
pipe1 = pp.create_pipe_from_parameters(net, from_junction=j2, to_junction=j3, length_km=0.1, diameter_m=0.05,
                                       name="Pipe 2")
pipe2 = pp.create_pipe_from_parameters(net, from_junction=j2, to_junction=j4, length_km=0.1, diameter_m=0.05,
                                       name="Pipe 3")

pipeflow(net)

simple_plot(net)
