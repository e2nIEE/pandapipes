import pandapipes.plotting as plot
from itertools import chain
from pandapower.control import ConstControl
from pandapipes.properties.fluids import get_fluid
from pandapower.control.basic_controller import Controller
from pandas.errors import InvalidIndexError

import matplotlib.pyplot as plt
import pandapipes.plotting as pp_plot

import sys
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_INSTALLED = True
except ImportError:
    MATPLOTLIB_INSTALLED = False

from pandapower.auxiliary import soft_dependency_error
from pandapower.plotting.plotting_toolbox import get_collection_sizes
from pandapower.plotting.collections import create_bus_collection, create_line_collection, \
    create_trafo_collection, create_trafo3w_collection, \
    create_line_switch_collection, draw_collections, create_bus_bus_switch_collection, create_ext_grid_collection, create_sgen_collection, \
    create_gen_collection, create_load_collection, create_dcline_collection
from pandapower.plotting.generic_geodata import create_generic_coordinates
from pandapipes.component_models.circulation_pump_mass_component import CirculationPumpMass
from pandapipes.component_models.circulation_pump_pressure_component import CirculationPumpPressure
from pandapipes.component_models.pump_component import Pump
from pandapipes.plotting.collections import create_junction_collection, create_pipe_collection, \
    create_valve_collection, create_source_collection, create_pressure_control_collection, \
    create_heat_exchanger_collection, create_sink_collection, create_pump_collection, \
    create_compressor_collection, create_flow_control_collection
from pandapipes.plotting.generic_geodata import create_generic_coordinates
from pandapipes.plotting.plotting_toolbox import get_collection_sizes

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)

import pandapower.plotting as pp_plot
import pandapipes.plotting as pipes_plot
import matplotlib.pyplot as plt


import pandapower.plotting as pp_plot
import pandapipes.plotting as pipes_plot
import matplotlib.pyplot as plt
import sys
from coupled_nets_power_heat import *
def plot_coupled_network_with_highlighted_bus(multinet, highlighted_bus):
    fig, ax = plt.subplots()

    # Plot pandapower network
    power_network = multinet["nets"]["power"]
    pp_plot.simple_plot(power_network, ax=ax)

    # Highlight the specific bus in pandapipes network
    heat_network = multinet["nets"]["heat"]
    highlighted_bus_color = 'red'  # Choose your desired color
    pipes_plot.simple_plot(heat_network, ax=ax, bus_color=highlighted_bus_color, highlighted_bus=highlighted_bus)

    plt.show()

plot_coupled_network_with_highlighted_bus(multinet, highlighted_bus=3)

def plot_coupled_network(multinet):
    fig, ax = plt.subplots()

    # Plot pandapower network
    power_network = multinet["nets"]["power"]
    pp_plot.simple_plot(power_network, ax=ax)

    # Plot pandapipes network
    heat_network = multinet["nets"]["heat"]
    pipes_plot.simple_plot(heat_network, ax=ax)

    plt.show()

# plotting coupled net
plot_coupled_network(multinet)



