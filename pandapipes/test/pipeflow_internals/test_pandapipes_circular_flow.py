# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 09:59:19 2022

@author: tzoschke
"""

import pandapipes as pp
from pandapipes.component_models import Pipe
import os
import pandas as pd
import pandapower.control as control
from pandapower.timeseries import DFData
from pandapower.timeseries import OutputWriter
from pandapipes.timeseries import run_timeseries
from pandapipes import networks
import numpy as np

# create empty net
net = pp.create_empty_network(fluid ="water")

# create fluid
#pandapipes.create_fluid_from_lib(net, "water", overwrite=True)


j0 = pp.create_junction(net, pn_bar=5, tfluid_k=293.15, name="junction 0")
j1 = pp.create_junction(net, pn_bar=5, tfluid_k=293.15, name="junction 1")
j2 = pp.create_junction(net, pn_bar=5, tfluid_k=293.15, name="junction 2")
j3 = pp.create_junction(net, pn_bar=5, tfluid_k=293.15, name="junction 3")


#Pump
#this is a component, which consists of an external grid, connected to the junction specified via the from_junction-parameter and a sink, connected to the junction specified via the to_junction-parameter
pp.create_circ_pump_const_mass_flow(net, from_junction=j0, to_junction=j3, p_bar=5, mdot_kg_per_s=20, t_k=273.15+35)

#Heat exchanger
#Positiv heat value means that heat is withdrawn from the network and supplied to a consumer
pp.create_heat_exchanger(net, from_junction=j1, to_junction=j2, diameter_m=200e-3, qext_w = 100000)

Pipe1 = pp.create_pipe_from_parameters(net, from_junction=j0, to_junction=j1, length_km=1, diameter_m=200e-3, k_mm=.1, alpha_w_per_m2k=10, sections = 5, text_k=283)
Pipe2 = pp.create_pipe_from_parameters(net, from_junction=j2, to_junction=j3, length_km=1, diameter_m=200e-3, k_mm=.1, alpha_w_per_m2k=10, sections = 5, text_k=283)



pp.pipeflow(net, mode='all')

net.res_junction #results are written to res_junction within toolbox.py
print(net.res_junction)
net.res_pipe
print(net.res_pipe)
print(net.res_pipe.t_from_k)
print(net.res_pipe.t_to_k)
#print(Pipe1.get_internal_results(net, [0]))


pipe_results = Pipe.get_internal_results(net, [0])
print("Temperature at different sections for Pipe 1:")
print(pipe_results["TINIT"])
pipe_results = Pipe.get_internal_results(net, [1])
print("Temperature at different sections for Pipe 2:")
print(pipe_results["TINIT"])
#Pipe.plot_pipe(net, 0, pipe_results)

#Start time series simulation
#_____________________________________________________________________

#Load profile for mass flow
profiles_mass_flow = pd.read_csv('mass_flow_pump.csv', index_col=0)
print(profiles_mass_flow)
#digital_df = pd.DataFrame({'0': [20,20,20,50,50,50,60,20,20,20]})
#print(digital_df)
#Prepare as data source for controller
ds_massflow = DFData(profiles_mass_flow)
print(type(ds_massflow))
#ds_massflow = DFData(digital_df)

#profiles_temperatures = pd.DataFrame({'0': [308,307,306,305,304,303,302,301,300,299]})
#ds_temperature = DFData(profiles_temperatures)


# Pass mass flow values to pump for time series simulation
const_sink = control.ConstControl(net, element='circ_pump_mass', variable='mdot_kg_per_s',element_index=net.circ_pump_mass.index.values, data_source=ds_massflow,profile_name=net.circ_pump_mass.index.values.astype(str))



#Define number of time steps
time_steps = range(10)



#Output Writer
log_variables = [ ('res_pipe', 'v_mean_m_per_s'),('res_pipe', 't_from_k'),('res_pipe', 't_to_k')]
ow = OutputWriter(net, time_steps, output_path=None, log_variables=log_variables)

# Pass temperature values to pump for time series simulation
#const_sink = control.ConstControl(net, element='circ_pump_mass', variable='t_k',element_index=net.circ_pump_mass.index.values, data_source=ds_temperature,profile_name=net.circ_pump_mass.index.values.astype(str))
previous_temperature = pd.DataFrame(net.res_pipe.t_to_k[1], index=time_steps, columns=['0'])#range(1))
print(previous_temperature)
ds_temperature = DFData(previous_temperature)
print(ds_temperature)

# @Quentin: the data_source='Pass' option doesn't exist in the current version of pandapipes (I think, this is even part of pandapower), this is what I added later on. The pass_variable is the temperature. So this way the outlet temperature of the component before the pipe is passed to the component after the pipe as a starting value
# unofortunately I couldnt find the actual implementation, but it was just a small edit, we will be able to reproduce it.
# This is the old function:
#const_sink = control.ConstControl(net, element='circ_pump_mass', variable='t_k',element_index=net.circ_pump_mass.index.values, data_source=None,profile_name=net.circ_pump_mass.index.values.astype(str))
# new implementation:
const_sink = control.ConstControl(net, element='circ_pump_mass', variable='t_k',element_index=net.circ_pump_mass.index.values, data_source='Pass',profile_name=net.circ_pump_mass.index.values.astype(str), pass_element='res_pipe', pass_variable='t_to_k', pass_element_index=1)


#Run time series simulation
run_timeseries(net, time_steps, mode = "all")

print("volume flow pipe:")
print(ow.np_results["res_pipe.v_mean_m_per_s"])
print("temperature into pipe:")
print(ow.np_results["res_pipe.t_from_k"])
print("temperature out of pipe:")
print(ow.np_results["res_pipe.t_to_k"])