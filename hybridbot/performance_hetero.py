##%
import pandapipes as pp
import os
from pandapipes import pp_dir
import pandas as pd
from jedi.plugins import pytest
import numpy as np
from pandapipes import pipeflow, from_json
from pandapipes.networks.simple_gas_networks import gas_3parallel, gas_meshed_delta, schutterwald
from pandapipes.plotting import simple_plot
from pandapipes.properties.properties_toolbox import calculate_molar_fraction_from_mass_fraction
from hybridbot.compressibility_func import calculate_mixture_compressibility_fact
#from pandapipes.plotting.plotly.simple_plotly import simple_plotly



net = pp.create_empty_network("net")
# create junction
j1 = pp.create_junction(net, pn_bar=1.05, tfluid_k=283.15, name="Junction 1")
j2 = pp.create_junction(net, pn_bar=1.05, tfluid_k=283.15, name="Junction 2")
j3 = pp.create_junction(net, pn_bar=1.05, tfluid_k=283.15, name="Junction 3")
j4 = pp.create_junction(net, pn_bar=1.05, tfluid_k=283.15, name="Junction 4")
# create junction elements
ext_grid = pp.create_ext_grid(net, fluid="methane", junction=j1, p_bar=20, t_k=283.15, name="Grid Connection")
sink = pp.create_sink(net, junction=j3, mdot_kg_per_s=0.045, name="Sink")
source = pp.create_source(net, junction=j4, mdot_kg_per_s=0.01, name="Source", fluid="hydrogen")
# create branch element
pipe = pp.create_pipe_from_parameters(net, from_junction=j1, to_junction=j2, length_km=0.1, diameter_m=0.05,
                                      name="Pipe 1")
pipe1 = pp.create_pipe_from_parameters(net, from_junction=j2, to_junction=j3, length_km=0.1, diameter_m=0.05,
                                       name="Pipe 2")
pipe2 = pp.create_pipe_from_parameters(net, from_junction=j2, to_junction=j4, length_km=0.1, diameter_m=0.05,
                                       name="Pipe 3")


pipeflow(net)
simple_plot(net)

mass_fraction = net.res_junction[['w_hydrogen','w_methane']].values
pressure = net.res_junction['p_bar']
temperature = net.res_junction['t_k']

critical_data_list = [net.fluid[fluid].get_critical_data() for fluid in net._fluid]
molar_mass_list = [net.fluid[fluid].get_molar_mass() for fluid in net._fluid]
molar_fraction = calculate_molar_fraction_from_mass_fraction(mass_fraction.T, np.array(molar_mass_list))
compressibility_fact, compressibility_fact_norm = calculate_mixture_compressibility_fact(molar_fraction.T, pressure, temperature, critical_data_list)

np.all(np.isclose(compressibility_fact,
                             (0.95926, 1.00264,1.00263,1.01068), rtol=1.e-4, atol=1.e-4))
# from pp import networks as nets_pps
#
# net = nets_pps.schutterwald()
# pp.create_sources(net, [5, 168, 193], 6.6e-3, 'hydrogen')
# pp.pipeflow(net, iter=100)
#



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


# def p_mean(p_from, p_to):
#     p_m = 2/3 * (p_from**3 - p_to**3) / (p_from**2 - p_to**2)
#     return p_m
#
# p_mean(net.res_pipe.p_from_bar, net.res_pipe.p_to_bar )
#
# pd.concat([net.res_pipe.p_from_bar, net.res_pipe.p_to_bar], axis =1)