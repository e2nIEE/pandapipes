##%
import pandas as pd

import pandapipes as pp
from pandapipes import pipeflow
from pandapipes.networks.simple_gas_networks import gas_3parallel, gas_meshed_delta, schutterwald
from pandapipes.plotting import simple_plot
#from pandapipes.plotting.plotly.simple_plotly import simple_plotly

net = pp.create_empty_network("net")
#net = pp.create_empty_network("net", "methane")
# create junction
j1 = pp.create_junction(net, pn_bar=1.05, tfluid_k=280, name="Junction 1")
j2 = pp.create_junction(net, pn_bar=1.05, tfluid_k=280, name="Junction 2")
j3 = pp.create_junction(net, pn_bar=1.05, tfluid_k=280, name="Junction 3")
j4 = pp.create_junction(net, pn_bar=1.05, tfluid_k=280, name="Junction 4")

# create junction elements
ext_grid = pp.create_ext_grid(net, fluid="hgas", junction=j1, p_bar=1.1, t_k=293.15, name="Grid Connection")
sink = pp.create_sink(net, junction=j3, mdot_kg_per_s=0.045, name="Sink")
source = pp.create_source(net, junction=j4, mdot_kg_per_s=0.01, name="Source", fluid="lgas")
# create branch element
pipe = pp.create_pipe_from_parameters(net, from_junction=j1, to_junction=j2, length_km=0.1, diameter_m=0.05,
                                      name="Pipe 1")
pipe1 = pp.create_pipe_from_parameters(net, from_junction=j2, to_junction=j3, length_km=0.1, diameter_m=0.05,
                                       name="Pipe 2")
pipe2 = pp.create_pipe_from_parameters(net, from_junction=j2, to_junction=j4, length_km=0.1, diameter_m=0.05,
                                       name="Pipe 3")



pipeflow(net)
simple_plot(net)

##%
node_column_names = ['TABLE_IDX','ELEMENT_IDX',
                    'NODE_TYPE','ACTIVE',
                    'PINIT','HEIGHT',
                    'TINIT','PAMB',
                    'LOAD_T','NODE_TYPE_T',
                    'EXT_GRID_OCCURENCE','EXT_GRID_OCCURENCE_T',
                    'FLUID','RHO',
                    'LOAD','SLACK']
for key in net.fluid:
    node_column_names.append(key + '_LOAD')
    node_column_names.append(key + '_W')
    node_column_names.append(key + '_RHO')
    node_column_names.append(key + '_JAC_DERIV_RHO_SAME_W')
    node_column_names.append(key + '_JAC_DERIV_RHO_DIFF_W')

branch_column_names= ['TABLE_IDX','ELEMENT_IDX',
                    'FROM_NODE','TO_NODE',
                    'ACTIVE','LENGTH',
                    'D','AREA',
                    'K','TINIT',
                    'VINIT','RE',
                    'LAMBDA','JAC_DERIV_DV',
                    'JAC_DERIV_DP','JAC_DERIV_DP1',
                    'LOAD_VEC_BRANCHES','JAC_DERIV_DV_NODE',
                    'LOAD_VEC_NODES','LOSS_COEFFICIENT',
                    'CP','ALPHA',
                    'JAC_DERIV_DT','JAC_DERIV_DT1',
                    'LOAD_VEC_BRANCHES_T','T_OUT',
                    'JAC_DERIV_DT_NODE','LOAD_VEC_NODES_T',
                    'VINIT_T','FROM_NODE_T',
                    'TO_NODE_T','QEXT',
                    'TEXT','STD_TYPE',
                    'PL','TL',
                    'BRANCH_TYPE','PRESSURE_RATIO',
                    'RHO','ETA',
                    'V_FROM_NODE','V_TO_NODE']

for key in net._fluid:
    branch_column_names.append(key + '_W')
    branch_column_names.append(key + '_RHO')
    branch_column_names.append(key + '_JAC_DERIV_RHO_SAME_W')
    branch_column_names.append(key + '_JAC_DERIV_RHO_DIFF_W')


def node_pit_to_pd(arr):
    return pd.DataFrame(arr, columns=node_column_names)

def branch_pit_to_pd(arr):
    return pd.DataFrame(arr, columns=branch_column_names)

node_pit_df = node_pit_to_pd(net["_pit"]['node'])
branch_pit_df = branch_pit_to_pd(net["_pit"]['branch'])

def p_mean(p_from, p_to):
    p_m = 2/3 * (p_from**3 - p_to**3) / (p_from**2 - p_to**2)
    return p_m

p_mean(net.res_pipe.p_from_bar, net.res_pipe.p_to_bar )

pd.concat([net.res_pipe.p_from_bar, net.res_pipe.p_to_bar], axis =1)