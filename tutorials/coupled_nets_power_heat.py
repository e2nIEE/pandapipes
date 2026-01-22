import pandapipes as pp
import pandapower as ppower
from pandapower.control.basic_controller import Controller
from pandapower import networks as pandasnet
from pandapipes import networks as pipenet
from pandapipes.multinet.create_multinet import create_empty_multinet, add_net_to_multinet
from pandapipes.multinet.control.run_control_multinet import run_control
import pandapower.plotting as pp_plot
import os
import matplotlib.pyplot as plt
from pandapower.plotting.plotly import simple_plotly
import pandas as pd
import pandapower.plotting.plotly as pplotly
import sys
from multinet_control_power2heat import *

import matplotlib.pyplot as plt
# ----------------------------------------
# Step 1: Creating the power network using pandapower's predefined simple network example
power_net = pandasnet.example_simple()  # Loading a predefined simple electrical network.

# ----------------------------------------
# Step 2: Creating a heat network using pandapipes
# This defines a thermal (fluid) network that models the heat system.

heat_net = pp.create_empty_network(fluid="water")  # Create an empty heat network with water as the fluid.

# Create junctions (nodes) in the heat network:
j0 = pp.create_junction(heat_net, pn_bar=5, tfluid_k=293.15, name="junction 0")  # Junction 0 at 5 bar pressure and 293.15 K temperature.
j1 = pp.create_junction(heat_net, pn_bar=5, tfluid_k=293.15, name="junction 1")  # Junction 1 with same pressure and temperature.
j2 = pp.create_junction(heat_net, pn_bar=5, tfluid_k=293.15, name="junction 2")  # Junction 2 with same pressure and temperature.
j3 = pp.create_junction(heat_net, pn_bar=5, tfluid_k=293.15, name="junction 3")  # Junction 3 with same pressure and temperature.

# Create a pump to circulate fluid between junctions j0 and j3, with a constant mass flow rate.
pp.create_circ_pump_const_mass_flow(heat_net, return_junction=j3, flow_junction=j0, p_flow_bar=5,
                                     mdot_flow_kg_per_s=20, t_flow_k=273.15+35)  # A pump for fluid flow at 20 kg/s with 5 bar pressure difference.

# Create a heat exchanger between junctions j1 and j2
pp.create_heat_exchanger(heat_net, from_junction=j1, to_junction=j2, diameter_m=200e-3, qext_w=100000)  # Heat exchanger between j1 and j2.

# Create pipes to connect the junctions and form the network.
pp.create_pipe_from_parameters(heat_net, from_junction=j0, to_junction=j1, length_km=1,
                               diameter_m=200e-3, k_mm=.1, alpha_w_per_m2k=10, sections=5, text_k=283)  # Pipe from j0 to j1.
pp.create_pipe_from_parameters(heat_net, from_junction=j2, to_junction=j3, length_km=1,
                               diameter_m=200e-3, k_mm=.1, alpha_w_per_m2k=10, sections=5, text_k=283)  # Pipe from j2 to j3.

# ----------------------------------------
# Step 3: Create a multinet (multi-network) and add power and heat networks
multinet = create_empty_multinet('multinet')  # Create an empty multinet system.
add_net_to_multinet(multinet, power_net, 'power')  # Add the power network to the multinet system.
add_net_to_multinet(multinet, heat_net, 'heat')  # Add the heat network to the multinet system.

# ----------------------------------------
# Step 4: Define conversion units (power-to-heat and heat-to-power) within the networks.
# These elements will facilitate energy conversions between the power and heat networks.

# Power-to-Heat (P2H) conversion: Creating a load in the power network (consuming power) and a heat exchanger in the heat network.
p2h_id_el = ppower.create_load(power_net, bus=6, p_mw=.0002, name="power to heat consumption")  # Load in the power network at bus 6 (0.2 MW).
p2h_id_heat = pp.create_heat_exchanger(heat_net, from_junction=0, to_junction=1, diameter_m=200e-3, qext_w=0, name="power to heat feed in")

# Heat-to-Power (H2P) conversion: Creating a heat exchanger in the heat network and a generator in the power network.
h2p_id_heat = pp.create_heat_exchanger(heat_net, from_junction=2, to_junction=3, diameter_m=200e-3, qext_w=200000, name="power to heat feed in")
h2p_id_el = ppower.create_sgen(power_net, bus=6, p_mw=0, name="fuel cell feed in")

# ----------------------------------------
# Step 5: Define control objects for the power-to-heat and heat-to-power conversions.

# Power-to-Heat control: A control object to manage the energy transfer between power and heat networks.
p2h_ctrl = P2HControlMultiEnergy(multinet, p2h_id_el, p2h_id_heat, efficiency=3,
                                 name_power_net="power", name_heat_net="heat")

# Heat-to-Power control: Another control object for managing the reverse flow (heat to power).
h2p_ctrl = H2PControlMultiEnergy(multinet, h2p_id_el, h2p_id_heat, efficiency=2,
                                 name_power_net="power", name_heat_net="heat")

# ----------------------------------------
# Step 6: Print initial values of power-to-heat and heat-to-power elements.
print(heat_net.heat_exchanger.loc[p2h_id_heat, 'qext_w'])  # Print the initial power-to-heat exchange rate.
print(power_net.sgen.loc[h2p_id_el, 'p_mw'])  # Print the initial power generation from the fuel cell.

# ----------------------------------------
# Step 7: Run the control simulation on the multinet system.
run_control(multinet)  # This runs the control logic for the power and heat networks based on the defined conversions.

# ----------------------------------------
# Step 8: Print updated values after running the simulation.
print(heat_net.heat_exchanger.loc[p2h_id_heat, 'qext_w'])  # Print the updated power-to-heat exchange rate after running control.
print(power_net.sgen.loc[h2p_id_el, 'p_mw'])  # Print the updated power generation after running control.
