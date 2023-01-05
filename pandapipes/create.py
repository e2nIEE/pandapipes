# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
import pandas as pd
from pandapower.create import _get_multiple_index_with_check, _get_index_with_check, _set_entries, \
    _check_node_element, _check_multiple_node_elements, _set_multiple_entries, \
    _add_multiple_branch_geodata, _check_branch_element, _check_multiple_branch_elements

from pandapipes.component_models import Junction, Sink, Source, Pump, Pipe, ExtGrid, \
    HeatExchanger, Valve, CirculationPumpPressure, CirculationPumpMass, PressureControlComponent, \
    Compressor, MassStorage
from pandapipes.component_models.component_toolbox import add_new_component
from pandapipes.component_models.flow_control_component import FlowControlComponent
from pandapipes.pandapipes_net import pandapipesNet, get_basic_net_entries, add_default_components
from pandapipes.properties import call_lib
from pandapipes.properties.fluids import Fluid, _add_fluid_to_net
from pandapipes.std_types.std_type_class import regression_function, PumpStdType
from pandapipes.std_types.std_types import add_basic_std_types, create_pump_std_type, \
    load_std_type

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def create_empty_network(name="", fluid=None, add_stdtypes=True):
    """
    This function initializes the pandapipes datastructure.

    :param name: Name for the network
    :type name: string, default None
    :param fluid: A fluid that can be added to the net from the start. Should be either of type\
            Fluid (c.f. pandapipes.properties.fluids.Fluid) or a string which refers to a standard\
            fluid type used to call `create_fluid_from_lib`. A fluid is required for pipeflow\
            calculations, but can also be added later.
    :type fluid: Fluid or str, default None
    :param add_stdtypes: Flag whether to add a dictionary of typical pump and pipe std types
    :type add_stdtypes: bool, default True
    :return: net - pandapipesNet with empty tables
    :rtype: pandapipesNet

    :Example:
        >>> net1 = create_empty_network("my_first_pandapipesNet", "lgas")
        >>> net2 = create_empty_network()

    """
    net = pandapipesNet(get_basic_net_entries())
    add_default_components(net, True)
    net['name'] = name
    if add_stdtypes:
        add_basic_std_types(net)

    if fluid is not None:
        if isinstance(fluid, Fluid):
            net["fluid"] = fluid
        elif isinstance(fluid, str):
            create_fluid_from_lib(net, fluid)
        else:
            logger.warning("The fluid %s cannot be added to the net. Only fluids of type Fluid or "
                           "strings can be used." % fluid)
    return net


def create_junction(net, pn_bar, tfluid_k, height_m=0, name=None, index=None, in_service=True,
                    type="junction", geodata=None, **kwargs):
    """
    Adds one junction in table net["junction"]. Junctions are the nodes of the network that
    all other elements connect to.

    :param net: The pandapipes network in which the element is created
    :type net: pandapipesNet
    :param pn_bar: The nominal pressure in [bar]. Used as an initial value for pressure calculation.
    :type pn_bar: float
    :param tfluid_k: The fluid temperature in [K]. Used as parameter for gas calculations and as\
            initial value for temperature calculations.
    :type tfluid_k: float
    :param height_m: Height of node above sea level in [m]
    :type height_m: float, default 0
    :param name: The name for this junction
    :type name: string, default None
    :param index: Force a specified ID if it is available. If None, the index one higher than the\
            highest already existing index is selected.
    :type index: int, default None
    :param in_service: True if the junction is in service or False if it is out of service
    :type in_service: boolean, default True
    :param type: not used yet - Designed for type differentiation on pandas lookups (e.g. household\
            connection vs. crossing)
    :type type: string, default "junction"
    :param geodata: Coordinates used for plotting
    :type geodata: (x,y)-tuple, default None
    :param kwargs: Additional keyword arguments will be added as further columns to the\
            net["junction"] table
    :return: index - The unique ID of the created element
    :rtype: int

    :Example:
        >>> create_junction(net, pn_bar=5, tfluid_k=320)
    """
    add_new_component(net, Junction)

    index = _get_index_with_check(net, "junction", index)

    cols = ["name", "pn_bar", "tfluid_k", "height_m", "in_service", "type"]
    vals = [name, pn_bar, tfluid_k, height_m, bool(in_service), type]

    _set_entries(net, "junction", index, **dict(zip(cols, vals)), **kwargs)

    if geodata is not None:
        if len(geodata) != 2:
            raise UserWarning("geodata must be given as (x, y) tuple")
        net["junction_geodata"].loc[index, ["x", "y"]] = geodata

    return index


def create_sink(net, junction, mdot_kg_per_s, scaling=1., name=None, index=None, in_service=True,
                type='sink', **kwargs):
    """
    Adds one sink in table net["sink"].

    :param net: The net for which this sink should be created
    :type net: pandapipesNet
    :param junction: The index of the junction to which the sink is connected
    :type junction: int
    :param mdot_kg_per_s: The required mass flow
    :type mdot_kg_per_s: float, default None
    :param scaling: An optional scaling factor to be set customly
    :type scaling: float, default 1
    :param name: A name tag for this sink
    :type name: str, default None
    :param index: Force a specified ID if it is available. If None, the index one higher than the\
            highest already existing index is selected.
    :type index: int, default None
    :param in_service: True for in service, False for out of service
    :type in_service: bool, default True
    :param type: Type variable to classify the sink
    :type type: str, default None
    :param kwargs: Additional keyword arguments will be added as further columns to the\
            net["sink"] table
    :return: index - The unique ID of the created element
    :rtype: int

    :Example:
        >>> new_sink_id = create_sink(net, junction=2, mdot_kg_per_s=0.1)

    """
    add_new_component(net, Sink)

    _check_junction_element(net, junction)
    index = _get_index_with_check(net, "sink", index)

    cols = ["name", "junction", "mdot_kg_per_s", "scaling", "in_service", "type"]
    vals = [name, junction, mdot_kg_per_s, scaling, bool(in_service), type]
    _set_entries(net, "sink", index, **dict(zip(cols, vals)), **kwargs)

    return index


def create_source(net, junction, mdot_kg_per_s, scaling=1., name=None, index=None, in_service=True,
                  type='source', **kwargs):
    """
    Adds one source in table net["source"].

    :param net: The net for which this source should be created
    :type net: pandapipesNet
    :param junction: The index of the junction to which the source is connected
    :type junction: int
    :param mdot_kg_per_s: The required mass flow
    :type mdot_kg_per_s: float, default None
    :param scaling: An optional scaling factor to be set customly
    :type scaling: float, default 1
    :param name: A name tag for this source
    :type name: str, default None
    :param index: Force a specified ID if it is available. If None, the index one higher than the\
            highest already existing index is selected.
    :type index: int, default None
    :param in_service: True for in service, False for out of service
    :type in_service: bool, default True
    :param type: Type variable to classify the source
    :type type: str, default None
    :param kwargs: Additional keyword arguments will be added as further columns to the\
            net["source"] table
    :return: index - The unique ID of the created element
    :rtype: int

    :Example:
        >>> create_source(net,junction=2,mdot_kg_per_s=0.1)

    """
    add_new_component(net, Source)

    _check_junction_element(net, junction)

    index = _get_index_with_check(net, "source", index)

    cols = ["name", "junction", "mdot_kg_per_s", "scaling", "in_service", "type"]
    vals = [name, junction, mdot_kg_per_s, scaling, bool(in_service), type]
    _set_entries(net, "source", index, **dict(zip(cols, vals)), **kwargs)

    return index

def create_mass_storage(net, junction, mdot_kg_per_s, init_m_stored_kg=0, min_m_stored_kg=0.,
                        max_m_stored_kg=np.inf, scaling=1., name=None, index=None,
                        in_service=True, type="mass_storage", **kwargs):
    """
    Adds one storage entry in table net["mass_storage"]. Not suitable for thermal storage tanks.

    :param net: The net for which this storage unit should be created
    :type net: pandapipesNet
    :param junction: The index of the junction to which the storage is connected
    :type junction: int
    :param mdot_kg_per_s: The stationary mass flow. (if fluid flows into storage: > 0,
                          if fluid flows from storage to net: < 0)
    :type mdot_kg_per_s: float, default None
    :param init_m_stored_kg: The initially stored mass in the storage
    :type init_m_stored_kg: float, default None
    :param min_m_stored_kg: Minimum amount of fluid that has to remain in the storage unit. (To be used
                   with controllers)
    :type min_m_stored_kg: float
    :param max_m_stored_kg: Maximum amount of fluid that can be stored in the storage unit. (To be used
                   with controllers)
    :type max_m_stored_kg: float, default np.inf
    :param scaling: An optional scaling factor to be set customly
    :type scaling: float, default 1
    :param name: A name tag for this storage unit
    :type name: str, default None
    :param index: Force a specified ID if it is available. If None, the index one higher than the\
            highest already existing index is selected.
    :type index: int, default None
    :param in_service: True for in service, False for out of service
    :type in_service: bool, default True
    :param type: Type variable to classify the storage
    :type type: str, default mass_storage
    :param kwargs: Additional keyword arguments will be added as further columns to the\
            net["mass_storage"] table
    :return: index - The unique ID of the created element
    :rtype: int

    :Example:
        >>> create_mass_storage(net, junction=2, mdot_kg_per_s=0.1)

    """
    if any([(init_m_stored_kg < 0), (min_m_stored_kg < 0), (max_m_stored_kg < 0)]):
        raise ValueError("init_/min_/max_stored_kg have to be >= 0!")
    if init_m_stored_kg > max_m_stored_kg:
        logger.warning("init_m_stored is automatically limited to max_m_stored_kg!")
        init_m_stored_kg = max_m_stored_kg
    if init_m_stored_kg < min_m_stored_kg:
        logger.warning("init_m_stored is automatically limited to min_m_stored_kg!")
        init_m_stored_kg = min_m_stored_kg

    add_new_component(net, MassStorage)

    _check_junction_element(net, junction)

    index = _get_index_with_check(net, "mass_storage", index)

    cols = ["name", "junction", "mdot_kg_per_s", "scaling", "init_m_stored_kg", "min_m_stored_kg",
            "max_m_stored_kg", "in_service", "type"]
    vals = [name, junction, mdot_kg_per_s, scaling, init_m_stored_kg, min_m_stored_kg,
            max_m_stored_kg, bool(in_service), type]
    _set_entries(net, "mass_storage", index, **dict(zip(cols, vals)), **kwargs)

    return index


def create_ext_grid(net, junction, p_bar=None, t_k=None, type="auto", name=None, in_service=True,
                    index=None, **kwargs):
    """
    Creates an external grid and adds it to the table net["ext_grid"]. It transfers the junction
    that it is connected to into a node with fixed value for either pressure, temperature or both
    (depending on the type). Usually external grids represent connections to other grids feeding
    the given pandapipesNet.

    :param net: The net that the external grid should be connected to
    :type net: pandapipesNet
    :param junction: The junction to which the external grid is connected
    :type junction: int
    :param p_bar: The pressure of the external grid
    :type p_bar: float, default None
    :param t_k: The fixed temperature at the external grid
    :type t_k: float, default None
    :param type: The external grid type denotes the values that are fixed at the respective node:\n
            - "auto": Will automatically assign one of the following types based on the input for \
                      p_bar and t_k \n
            - "p": The pressure is fixed, the node acts as a slack node for the mass flow. \n
            - "t": The temperature is fixed and will not be solved for, but is assumed as the \
                   node's mix temperature. Please note that pandapipes cannot check for \
                   inconsistencies in the formulation of heat transfer equations yet. \n
            - "pt": The external grid shows both "p" and "t" behavior.
    :type type: str, default "auto"
    :param name: A name tag for this ext_grid
    :type name: str, default None
    :param in_service: True for in service, False for out of service
    :type in_service: bool, default True
    :param index: Force a specified ID if it is available. If None, the index one higher than the\
            highest already existing index is selected.
    :type index: int, default None
    :param kwargs: Additional keyword arguments will be added as further columns to the\
                    net["ext_grid"] table
    :return: index - The unique ID of the created element
    :rtype: int

    :Example:
        >>> create_ext_grid(net, junction=2, p_bar=100, t_k=293.15)

    """
    add_new_component(net, ExtGrid)

    _check_junction_element(net, junction)
    index = _get_index_with_check(net, "ext_grid", index, name="external grid")

    type = _auto_ext_grid_type(p_bar, t_k, type, ExtGrid)

    cols = ["name", "junction", "p_bar", "t_k", "in_service", "type"]
    vals = [name, junction, p_bar, t_k, bool(in_service), type]
    _set_entries(net, "ext_grid", index, **dict(zip(cols, vals)), **kwargs)

    return index


def create_heat_exchanger(net, from_junction, to_junction, diameter_m, qext_w, loss_coefficient=0,
                          name=None, index=None, in_service=True, type="heat_exchanger", **kwargs):
    """
    Creates a heat exchanger element in net["heat_exchanger"] from heat exchanger parameters.

    :param net: The net for which this heat exchanger should be created
    :type net: pandapipesNet
    :param from_junction: ID of the junction on one side which the heat exchanger will be\
            connected with
    :type from_junction: int
    :param to_junction: ID of the junction on the other side which the heat exchanger will be\
            connected with
    :type to_junction: int
    :param diameter_m: The heat exchanger inner diameter in [m]
    :type diameter_m: float
    :param qext_w: External heat flux in [W]. If positive, heat is derived from the network. If
            negative, heat is being fed into the network from a heat source.
    :type qext_w: float
    :param loss_coefficient: An additional pressure loss coefficient, introduced by e.g. bends
    :type loss_coefficient: float
    :param name: The name of the heat exchanger
    :type name: str, default None
    :param index: Force a specified ID if it is available. If None, the index one higher than the\
            highest already existing index is selected.
    :type index: int, default None
    :param in_service: True if the heat exchanger is in service or False if it is out of service
    :type in_service: bool, default True
    :param type: Not used yet
    :type type: str
    :param kwargs: Additional keyword arguments will be added as further columns to the\
                    net["heat_exchanger"] table
    :return: index - The unique ID of the created heat exchanger
    :rtype: int

    :Example:
        >>> create_heat_exchanger(net, from_junction=0, to_junction=1,
        >>>                       diameter_m=40e-3, qext_w=2000)
    """
    add_new_component(net, HeatExchanger)

    index = _get_index_with_check(net, "heat_exchanger", index, "heat exchanger")
    _check_branch(net, "Heat exchanger", index, from_junction, to_junction)

    v = {"name": name, "from_junction": from_junction, "to_junction": to_junction,
         "diameter_m": diameter_m, "qext_w": qext_w, "loss_coefficient": loss_coefficient,
         "in_service": bool(in_service), "type": type}
    _set_entries(net, "heat_exchanger", index, **v, **kwargs)

    return index


def create_pipe(net, from_junction, to_junction, std_type, length_km, k_mm=0.2, loss_coefficient=0,
                sections=1, alpha_w_per_m2k=0., text_k=293, qext_w=0., name=None, index=None,
                geodata=None, in_service=True, type="pipe", **kwargs):
    """
    Creates a pipe element in net["pipe"] from pipe parameters.

    :param net: The net for which this pipe should be created
    :type net: pandapipesNet
    :param from_junction: ID of the junction on one side which the pipe will be connected to
    :type from_junction: int
    :param to_junction: ID of the junction on the other side to which the pipe will be connected to
    :type to_junction: int
    :param std_type: Name of standard type
    :type std_type: str
    :param length_km: Length of the pipe in [km]
    :type length_km: float
    :param k_mm: Pipe roughness in [mm]. 0.2 mm is quite rough, usually betweeen 0.0015 (new
            pipes) and 0.3 (old steel pipelines)
    :type k_mm: float, default 0.2
    :param loss_coefficient: An additional pressure loss coefficient, introduced by e.g. bends
    :type loss_coefficient: float, default 0
    :param sections: The number of internal pipe sections. Important for gas and temperature\
            calculations, where variables are dependent on pipe length.
    :type sections: int, default 1
    :param alpha_w_per_m2k: Heat transfer coefficient in [W/(m^2*K)]
    :type alpha_w_per_m2k: float, default 0
    :param text_k: Ambient temperature of pipe in [K]
    :type text_k: float, default 293
    :param qext_w: External heat feed-in to the pipe in [W]
    :type qext_w: float, default 0
    :param name: A name tag for this pipe
    :type name: str, default None
    :param index: Force a specified ID if it is available. If None, the index one higher than the\
            highest already existing index is selected.
    :type index: int, default None
    :param geodata: The coordinates of the pipe. The first row should be the coordinates of\
            junction a and the last should be the coordinates of junction b. The points in the\
            middle represent the bending points of the pipe.
    :type geodata: array, shape=(,2L), default None
    :param in_service: True for in service, False for out of service
    :type in_service: bool, default True
    :param type: An identifier for special types of pipes (e.g. below or above ground)
    :type type: str, default "pipe"
    :param kwargs: Additional keyword arguments will be added as further columns to the\
            net["pipe"] table
    :return: index - The unique ID of the created element
    :rtype: int

    :Example:
        >>> create_pipe(net, from_junction=0, to_junction=1,
        >>>             std_type='315_PE_80_SDR_17', length_km=1)

    """
    add_new_component(net, Pipe)

    index = _get_index_with_check(net, "pipe", index)
    _check_branch(net, "Pipe", index, from_junction, to_junction)
    _check_std_type(net, std_type, "pipe", "create_pipe")

    pipe_parameter = load_std_type(net, std_type, "pipe")
    v = {"name": name, "from_junction": from_junction, "to_junction": to_junction,
         "std_type": std_type, "length_km": length_km,
         "diameter_m": pipe_parameter["inner_diameter_mm"] / 1000, "k_mm": k_mm,
         "loss_coefficient": loss_coefficient, "alpha_w_per_m2k": alpha_w_per_m2k,
         "sections": sections, "in_service": bool(in_service), "type": type, "qext_w": qext_w,
         "text_k": text_k}
    _set_entries(net, "pipe", index, **v, **kwargs)

    if geodata is not None:
        net["pipe_geodata"].at[index, "coords"] = geodata

    return index


def create_pipe_from_parameters(net, from_junction, to_junction, length_km, diameter_m, k_mm=0.2,
                                loss_coefficient=0, sections=1, alpha_w_per_m2k=0., text_k=293,
                                qext_w=0., name=None, index=None, geodata=None, in_service=True,
                                type="pipe", **kwargs):
    """
    Creates a pipe element in net["pipe"] from pipe parameters.

    :param net: The net for which this pipe should be created
    :type net: pandapipesNet
    :param from_junction: ID of the junction on one side which the pipe will be connected with
    :type from_junction: int
    :param to_junction: ID of the junction on the other side to which the pipe will be connected to
    :type to_junction: int
    :param length_km: Length of the pipe in [km]
    :type length_km: float
    :param diameter_m: The pipe diameter in [m]
    :type diameter_m: float
    :param k_mm: Pipe roughness in [mm]. 0.2 mm is quite rough, usually betweeen 0.0015 (new
            pipes) and 0.3 (old steel pipelines)
    :type k_mm: float, default 0.2
    :param loss_coefficient: An additional pressure loss coefficient, introduced by e.g. bends
    :type loss_coefficient: float, default 0
    :param sections: The number of internal pipe sections. Important for gas and temperature\
            calculations, where variables are dependent on pipe length.
    :type sections: int, default 1
    :param alpha_w_per_m2k: Heat transfer coefficient in [W/(m^2*K)]
    :type alpha_w_per_m2k: float, default 0
    :param qext_w: external heat feed-in to the pipe in [W]
    :type qext_w: float, default 0
    :param text_k: Ambient temperature of pipe in [K]
    :type text_k: float, default 293
    :param name: A name tag for this pipe
    :type name: str, default None
    :param index: Force a specified ID if it is available. If None, the index one higher than the\
            highest already existing index is selected.
    :type index: int, default None
    :param geodata: The coordinates of the pipe. The first row should be the coordinates of\
            junction a and the last should be the coordinates of junction b. The points in the\
            middle represent the bending points of the pipe
    :type geodata: array, shape= (,2L), default None
    :param in_service: True for in service, false for out of service
    :type in_service: bool, default True
    :param type: An identifier for special types of pipes (e.g. below or above ground)
    :type type: str, default "pipe"
    :param kwargs: Additional keyword arguments will be added as further columns to the\
            net["pipe"] table
    :return: index - The unique ID of the created element
    :rtype: int

    :Example:
        >>> create_pipe_from_parameters(net, from_junction=0, to_junction=1,
        >>>                             length_km=1, diameter_m=40e-3)

    """
    add_new_component(net, Pipe)

    index = _get_index_with_check(net, "pipe", index)
    _check_branch(net, "Pipe", index, from_junction, to_junction)

    v = {"name": name, "from_junction": from_junction, "to_junction": to_junction,
         "std_type": None, "length_km": length_km, "diameter_m": diameter_m, "k_mm": k_mm,
         "loss_coefficient": loss_coefficient, "alpha_w_per_m2k": alpha_w_per_m2k,
         "sections": sections, "in_service": bool(in_service),
         "type": type, "qext_w": qext_w, "text_k": text_k}
    if 'std_type' in kwargs:
        raise UserWarning('you have defined a std_type, however, using this function you can only '
                          'create a pipe setting specific, individual parameters. If you want to '
                          'create a pipe from net.std_types, please use `create_pipe`')
    _set_entries(net, "pipe", index, **v, **kwargs)

    if geodata is not None:
        net["pipe_geodata"].at[index, "coords"] = geodata

    return index


def create_valve(net, from_junction, to_junction, diameter_m, opened=True, loss_coefficient=0,
                 name=None, index=None, type='valve', **kwargs):
    """
    Creates a valve element in net["valve"] from valve parameters.

    :param net: The net for which this valve should be created
    :type net: pandapipesNet
    :param from_junction: ID of the junction on one side which the valve will be connected with
    :type from_junction: int
    :param to_junction: ID of the junction on the other side which the valve will be connected with
    :type to_junction: int
    :param diameter_m: The valve diameter in [m]
    :type diameter_m: float
    :param opened: Flag to show if the valve is opened and allows for fluid flow or if it is closed\
            to block the fluid flow.
    :type opened: bool, default True
    :param loss_coefficient: The pressure loss coefficient introduced by the valve shape
    :type loss_coefficient: float, default 0
    :param name: A name tag for this valve
    :type name: str, default None
    :param index: Force a specified ID if it is available. If None, the index one higher than the\
            highest already existing index is selected.
    :type index: int, default None
    :param type: An identifier for special types of valves
    :type type: str, default None
    :param kwargs: Additional keyword arguments will be added as further columns to the\
            net["valve"] table
    :return: index - The unique ID of the created element
    :rtype: int

    :Example:
        >>> create_valve(net, 0, 1, diameter_m=4e-3, name="valve1")

    """
    add_new_component(net, Valve)

    index = _get_index_with_check(net, "valve", index)
    _check_branch(net, "Valve", index, from_junction, to_junction)

    v = {"name": name, "from_junction": from_junction, "to_junction": to_junction,
         "diameter_m": diameter_m, "opened": opened, "loss_coefficient": loss_coefficient,
         "type": type}
    _set_entries(net, "valve", index, **v, **kwargs)

    return index


def create_pump(net, from_junction, to_junction, std_type, name=None, index=None, in_service=True,
                type="pump", **kwargs):
    """
    Adds one pump in table net["pump"].

    :param net: The net for which this pump should be created
    :type net: pandapipesNet
    :param from_junction: ID of the junction on one side which the pump will be connected with
    :type from_junction: int
    :param to_junction: ID of the junction on the other side which the pump will be connected with
    :type to_junction: int
    :param std_type: There are currently three different std_types. This std_types are P1, P2, P3.\
            Each of them describes a specific pump behaviour setting volume flow and pressure in\
            context.
    :type std_type: string, default None
    :param name: A name tag for this pump
    :type name: str, default None
    :param index: Force a specified ID if it is available. If None, the index one higher than the\
            highest already existing index is selected.
    :type index: int, default None
    :param in_service: True if the pump is in service or False if it is out of service
    :type in_service: bool, default True
    :param type:  Type variable to classify the pump
    :type type: str, default "pump"
    :param kwargs: Additional keyword arguments will be added as further columns to the\
            net["pump"] table
    :type kwargs: dict
    :return: index - The unique ID of the created element
    :rtype: int

    EXAMPLE:
        >>> create_pump(net, 0, 1, std_type="P1")

    """
    add_new_component(net, Pump)

    index = _get_index_with_check(net, "pump", index)
    _check_branch(net, "Pump", index, from_junction, to_junction)

    _check_std_type(net, std_type, "pump", "create_pump")
    v = {"name": name, "from_junction": from_junction, "to_junction": to_junction,
         "std_type": std_type, "in_service": bool(in_service), "type": type}
    _set_entries(net, "pump", index, **v, **kwargs)

    return index


def create_pump_from_parameters(net, from_junction, to_junction, new_std_type_name,
                                pressure_list=None, flowrate_list=None, reg_polynomial_degree=None,
                                poly_coefficents=None, name=None, index=None, in_service=True,
                                type="pump", **kwargs):
    """
    Adds one pump in table net["pump"].

    :param net: The net for which this pump should be created
    :type net: pandapipesNet
    :param from_junction: ID of the junction on one side which the pump will be connected with
    :type from_junction: int
    :param to_junction: ID of the junction on the other side which the pump will be connected with
    :type to_junction: int
    :param new_std_type_name: Set a name for your pump. You will find your definied pump under
            std_type in your net. The name will be given under std_type in net.pump.
    :type new_std_type_name: string
    :param pressure_list: This list contains measured pressure supporting points required\
            to define and determine the dependencies of the pump between pressure and volume flow.\
            The pressure must be given in [bar]. Needs to be defined only if no pump of standard\
            type is selected.
    :type pressure_list: list, default None
    :param flowrate_list: This list contains the corresponding flowrate values to the given\
            pressure values. Thus, the length must be equal to the pressure list. Needs to be\
            defined only if no pump of standard type is selected. ATTENTION: The flowrate values\
            are given in :math:`[\\frac{m^3}{h}]`.
    :type flowrate_list: list, default None
    :param reg_polynomial_degree: The degree of the polynomial fit must be defined if pressure\
            and flowrate list are given. The fit describes the behaviour of the pump (delta P /\
            volumen flow curve).
    :type reg_polynomial_degree: int, default None
    :param poly_coefficents: Alternatviely to taking measurement values and degree of polynomial
            fit, previously calculated regression parameters can also be given directly. It
            describes the dependency between pressure and flowrate.\
            ATTENTION: The determined parameteres must be retrieved by setting flowrate given\
            in :math:`[\\frac{m^3}{h}]` and pressure given in bar in context. The first entry in\
            the list (c[0]) is for the polynom of the highest degree (c[0]*x**n), the last one for\
            c*x**0.
    :type poly_coefficents: list, default None
    :param name: A name tag for this pump
    :type name: str, default None
    :param index: Force a specified ID if it is available. If None, the index one higher than the\
            highest already existing index is selected.
    :type index: int, default None
    :param in_service: True if the pump is in service or False if it is out of service
    :type in_service: bool, default True
    :param type:  type variable to classify the pump
    :type type: str, default "pump"
    :param kwargs: Additional keyword arguments will be added as further columns to the\
            net["pump"] table
    :type kwargs: dict
    :return: index - The unique ID of the created element
    :rtype: int

    EXAMPLE:
        >>> create_pump_from_parameters(net, 0, 1, 'pump1', pressure_list=[0,1,2,3],
        >>>                             flowrate_list=[0,1,2,3], reg_polynomial_degree=1)
        >>> create_pump_from_parameters(net, 0, 1, 'pump2', poly_coefficents=[1,0])

    """
    add_new_component(net, Pump)

    index = _get_index_with_check(net, "pump", index)
    _check_branch(net, "Pump", index, from_junction, to_junction)

    if pressure_list is not None and flowrate_list is not None \
            and reg_polynomial_degree is not None:
        reg_par = regression_function(flowrate_list, pressure_list, reg_polynomial_degree)
        pump = PumpStdType(new_std_type_name, reg_par)
        create_pump_std_type(net, new_std_type_name, pump)
    elif poly_coefficents is not None:
        pump = PumpStdType(new_std_type_name, poly_coefficents)
        create_pump_std_type(net, new_std_type_name, pump)

    v = {"name": name, "from_junction": from_junction, "to_junction": to_junction,
         "std_type": new_std_type_name, "in_service": bool(in_service), "type": type}
    _set_entries(net, "pump", index, **v, **kwargs)

    return index


def create_circ_pump_const_pressure(net, return_junction, flow_junction, p_flow_bar, plift_bar,
                                    t_flow_k=None, type="auto", name=None, index=None,
                                    in_service=True, **kwargs):
    """
    Adds one circulation pump with a constant pressure lift in table net["circ_pump_pressure"]. \n
    A circulation pump is a component that sets the pressure at its outlet (flow junction) and
    asserts that the correct mass flow is extracted at its inlet (return junction). \n
    In this particular case, the pressure lift is fixed, i.e. the pressure on both sides are set
    (with the pressure lift as difference). The mass flow through the component is just a result
    of the balance of the network. An equal representation is adding external grids at each of the
    connected nodes.

    :param net: The net for which this pump should be created
    :type net: pandapipesNet
    :param return_junction: ID of the junction on one side which the pump will be connected with
    :type return_junction: int
    :param flow_junction: ID of the junction on the other side which the pump will be connected with
    :type flow_junction: int
    :param p_flow_bar: Pressure set point at the flow junction
    :type p_flow_bar: float
    :param plift_bar: Pressure lift induced by the pump
    :type plift_bar: float
    :param t_flow_k: Temperature set point at the flow junction
    :type t_flow_k: float, default None
    :param type: The pump type denotes the values that are fixed:\n
            - "auto": Will automatically assign one of the following types based on the input for \
                  p_bar and t_k \n
            - "p": The pressure at the flow junction is fixed. \n
            - "t": The temperature at the flow junction is fixed and will not be solved. Please \
                note that pandapipes cannot check for inconsistencies in the formulation of heat \
                transfer equations yet.
            - "pt": The circulation pump shows both "p" and "t" behavior.
    :type type: str, default "auto"
    :param name: Name of the pump
    :type name: str
    :param index: Force a specified ID if it is available. If None, the index one higher than the\
            highest already existing index is selected.
    :type index: int, default None
    :param in_service: True if the circulation pump is in service or False if it is out of service
    :type in_service: bool, default True
    :param kwargs: Additional keyword arguments will be added as further columns to the\
            net["circ_pump_pressure"] table
    :type kwargs: dict
    :return: index - The unique ID of the created element
    :rtype: int

    :Example:
        >>> create_circ_pump_const_pressure(net, 0, 1, p_flow_bar=5, plift_bar=2,
        >>>                                 t_flow_k=350, type="p")

    """

    add_new_component(net, CirculationPumpPressure)

    index = _get_index_with_check(net, "circ_pump_pressure", index,
                                  name="circulation pump with constant pressure")
    _check_branch(net, "circulation pump with constant pressure", index, return_junction,
                  flow_junction)

    type = _auto_ext_grid_type(p_flow_bar, t_flow_k, type, CirculationPumpPressure)

    v = {"name": name, "return_junction": return_junction, "flow_junction": flow_junction,
         "p_flow_bar": p_flow_bar, "t_flow_k": t_flow_k, "plift_bar": plift_bar, "type": type,
         "in_service": bool(in_service)}
    _set_entries(net, "circ_pump_pressure", index, **v, **kwargs)

    return index


def create_circ_pump_const_mass_flow(net, return_junction, flow_junction, p_flow_bar,
                                     mdot_flow_kg_per_s, t_flow_k=None, type="auto", name=None,
                                     index=None, in_service=True, **kwargs):
    """
    Adds one circulation pump with a constant mass flow in table net["circ_pump_mass"].\n
    A circulation pump is a component that sets the pressure at its outlet (flow junction) and
    asserts that the correct mass flow is extracted at its inlet (return junction). \n
    In this particular case, the mass flow and the pressure on the flow side are fixed, i.e. the
    pressure on the return side is just a result of the friction losses in the network. An equal
    representation is adding an external grid at the flow junction and a sink with the given mass
    flow at the return junction.

    :param net: The net for which this pump should be created
    :type net: pandapipesNet
    :param return_junction: ID of the junction on one side which the pump will be connected with
    :type return_junction: int
    :param flow_junction: ID of the junction on the other side which the pump will be connected with
    :type flow_junction: int
    :param p_flow_bar: Pressure set point at the flow junction
    :type p_flow_bar: float
    :param mdot_flow_kg_per_s: Constant mass flow, which is transported through the pump
    :type mdot_flow_kg_per_s: float
    :param t_flow_k: Temperature set point at the flow junction
    :type t_flow_k: float, default None
    :param name: Name of the pump
    :type name: str
    :param index: Force a specified ID if it is available. If None, the index one higher than the\
            highest already existing index is selected.
    :type index: int, default None
    :param in_service: True if the circulation pump is in service or False if it is out of service
    :type in_service: bool, default True
    :param type: The pump type denotes the values that are fixed:\n
            - "auto": Will automatically assign one of the following types based on the input for \
                  p_bar and t_k \n
            - "p": The pressure is fixed. \n
            - "t": The temperature is fixed and will not be solved. Please note that pandapipes\
             cannot check for inconsistencies in the formulation of heat transfer equations yet.
            - "pt": The pump shows both "p" and "t" behavior.
    :type type: str, default "auto"
    :param kwargs: Additional keyword arguments will be added as further columns to the\
            net["circ_pump_mass"] table
    :type kwargs: dict
    :return: index - The unique ID of the created element
    :rtype: int

    :Example:
        >>> create_circ_pump_const_mass_flow(net, 0, 1, p_flow_bar=5, mdot_flow_kg_per_s=2,
        >>>                                  t_flow_k=350, type="pt")

    """

    add_new_component(net, CirculationPumpMass)

    index = _get_index_with_check(net, "circ_pump_mass", index,
                                  name="circulation pump with constant mass flow")
    _check_branch(net, "circulation pump with constant mass flow", index, return_junction,
                  flow_junction)

    type = _auto_ext_grid_type(p_flow_bar, t_flow_k, type, CirculationPumpMass)

    v = {"name": name, "return_junction": return_junction, "flow_junction": flow_junction,
         "p_flow_bar": p_flow_bar, "t_flow_k": t_flow_k, "mdot_flow_kg_per_s": mdot_flow_kg_per_s,
         "type": type, "in_service": bool(in_service)}
    _set_entries(net, "circ_pump_mass", index, **v, **kwargs)

    return index


def create_compressor(net, from_junction, to_junction, pressure_ratio, name=None, index=None,
                      in_service=True, **kwargs):
    """Adds a compressor with relative pressure lift to net["compressor"].

    The outlet (absolute) pressure is calculated by (p_in + p_ambient) * pressure_ratio. For
    reverse flow, bypassing is assumed (no pressure lift).

    :param net: The net within this compressor should be created
    :type net: pandapipesNet
    :param from_junction: ID of the junction on one side which the compressor will be connected with
    :type from_junction: int
    :param to_junction: ID of the junction on the other side which the compressor will be connected\
                        with
    :type to_junction: int
    :param pressure_ratio: enforced ratio of outlet to inlet absolute pressures
    :type pressure_ratio: float
    :param name: A name tag for this compressor
    :type name: str, default None
    :param index: Force a specified ID if it is available. If None, the index one higher than the\
            highest already existing index is selected.
    :type index: int, default None
    :param in_service: True for in_service or False for out of service
    :type in_service: bool, default True
    :param kwargs: Additional keyword arguments will be added as further columns to the\
            net["compressor"] table
    :type kwargs: dict
    :return: index - The unique ID of the created element
    :rtype: int

    EXAMPLE:
        >>> create_compressor(net, 0, 1, pressure_ratio=1.3)

    """
    add_new_component(net, Compressor)

    index = _get_index_with_check(net, "compressor", index)
    _check_branch(net, "Compressor", index, from_junction, to_junction)

    v = {"name": name, "from_junction": from_junction, "to_junction": to_junction,
         "pressure_ratio": pressure_ratio, "in_service": bool(in_service)}
    _set_entries(net, "compressor", index, **v, **kwargs)

    return index


def create_pressure_control(net, from_junction, to_junction, controlled_junction, controlled_p_bar,
                            control_active=True, loss_coefficient=0., name=None, index=None,
                            in_service=True, type="pressure_control", **kwargs):
    """Adds one pressure control that enforces a pressure at a specific junction.

    The pressure control unit creates a pressure drop / lift between the 'from' and the 'to'
    junction so that the pressure set point at the controlled junction is met.
    It is required that the controlled junction is hydraulically properly connected to the from
    and to junction and no other pessure control unit is inbetween.

    :param net: The net for which this pressure control should be created
    :type net: pandapipesNet
    :param from_junction: ID of the junction on one side which the pressure control will be \
            connected with
    :type from_junction: int
    :param to_junction: ID of the junction on the other side which the pressure control will be \
            connected with
    :type to_junction: int
    :param controlled_junction: ID of the junction at which the pressure is controlled
    :type controlled_junction: int
    :param controlled_p_bar: Pressure set point
    :type controlled_p_bar: float
    :param control_active: Variable to state whether the pressure control is active (otherwise \
        behaviour similar to open valve)
    :type control_active: bool, default True
    :param loss_coefficient: The pressure loss coefficient introduced by the component's shape \
        (used only if control is not active).
    :type loss_coefficient: float, default 0
    :param name: Name of the pressure control element
    :type name: str
    :param index: Force a specified ID if it is available. If None, the index one higher than the\
            highest already existing index is selected.
    :type index: int, default None
    :param in_service: True if the pressure control is in service or False if it is out of service
    :type in_service: bool, default True
    :param type: Currently not used - possibility to specify a certain type of pressure control
    :type type: str, default "pressure_control"
    :param kwargs: Additional keyword arguments will be added as further columns to the \
            net["press_control"] table
    :type kwargs: dict
    :return: index - The unique ID of the created element
    :rtype: int

    :Example:
        Connect junction 0 and 1 and set the pressure at junction 1 to 5 bar.

        >>> create_pressure_control(net, 0, 1, 1, controlled_p_bar=5)

    """
    from pandapipes.toolbox import check_pressure_controllability
    if not check_pressure_controllability(net, to_junction, controlled_junction):
        return logger.error('The controlled junction of the created pressure control '
                            'is not controllable, as it is either not reachable or '
                            'another pressure controllable component is in between')

    logger.info('Using a default pressure controller in pandapipes assumes, that the temperature '
                'settings at the junctions are kept. Therefore, energy is induced to meet these '
                'constraints.')

    add_new_component(net, PressureControlComponent)

    index = _get_index_with_check(net, "press_control", index)

    # check if junctions exist to attach the pump to
    _check_branch(net, "PressureControl", index, from_junction, to_junction)

    _set_entries(net, "press_control", index, name=name, from_junction=from_junction,
                 to_junction=to_junction, controlled_junction=controlled_junction,
                 control_active=bool(control_active), loss_coefficient=loss_coefficient,
                 controlled_p_bar=controlled_p_bar, in_service=bool(in_service), type=type,
                 **kwargs)

    if controlled_junction != from_junction and controlled_junction != to_junction:
        logger.warning("The pressure controller %d controls the pressure at a junction that it is "
                       "not connected to. Please note that this can lead to errors in the pipeflow "
                       "calculation that will not be displayed properly. Make sure that your grid "
                       "configuration is valid." % index)

    return index


def create_flow_control(net, from_junction, to_junction, controlled_mdot_kg_per_s, diameter_m,
                        control_active=True, name=None, index=None, in_service=True, type="fc",
                        **kwargs):
    """
    Adds one flow control with a constant mass flow in table net["flow_control"].

    :param net: The net for which this flow control should be created
    :type net: pandapipesNet
    :param from_junction: ID of the junction on one side which the flow control will be connected \
        with
    :type from_junction: int
    :param to_junction: ID of the junction on the other side which the flow control will be \
        connected with
    :type to_junction: int
    :param controlled_mdot_kg_per_s: Mass flow set point
    :type controlled_mdot_kg_per_s: float
    :param diameter_m: Measure of the diameter to derive the cross-sectional area (important for \
            the velocity calculation)
    :type diameter_m: float
    :param control_active: Variable to state whether the flow control is active (otherwise \
            similar to open valve)
    :type control_active: bool, default True
    :param name: Name of the flow control element
    :type name: str
    :param index: Force a specified ID if it is available. If None, the index one higher than the\
            highest already existing index is selected.
    :type index: int, default None
    :param in_service: True if flow controller is in service or False if it is out of service
    :type in_service: bool, default True
    :param type: Currently not used - possibility to specify a certain type of flow control
    :type type: str, default "fc"
    :param kwargs: Additional keyword arguments will be added as further columns to the \
            net["flow_control"] table
    :return: index - The unique ID of the created elements
    :rtype: int

    :Example:
        >>> create_flow_control(net, 0, 1, 0.5, 0.8)

    """

    add_new_component(net, FlowControlComponent)

    index = _get_index_with_check(net, "flow_control", index)

    # check if junctions exist to attach the pump to
    _check_branch(net, "FlowControl", index, from_junction, to_junction)

    _set_entries(net, "flow_control", index, name=name, from_junction=from_junction,
                 to_junction=to_junction, controlled_mdot_kg_per_s=controlled_mdot_kg_per_s,
                 diameter_m=diameter_m, control_active=bool(control_active),
                 in_service=bool(in_service), type=type, **kwargs)

    return index


def create_junctions(net, nr_junctions, pn_bar, tfluid_k, height_m=0, name=None, index=None,
                     in_service=True, type="junction", geodata=None, **kwargs):
    """
    Convenience function for creating many junctions at once. Parameter 'nr_junctions' specifies \
    the number of junctions created. Other parameters may be either arrays of length 'nr_junctions'\
    or single values.

    :param net: The pandapipes network in which the element is created
    :type net: pandapipesNet
    :param nr_junctions: Number of junctions to be created.
    :type nr_junctions: int
    :param pn_bar: The nominal pressure in [bar]. Used as an initial value for pressure calculation.
    :type pn_bar: Iterable or float
    :param tfluid_k: The fluid temperature in [K]. Used as parameter for gas calculations and as\
            initial value for temperature calculations.
    :type tfluid_k: Iterable or float
    :param height_m: Heights of nodes above sea level in [m]
    :type height_m: Iterable or float, default 0
    :param name: The names for these junctions
    :type name: Iterable or string, default None
    :param index: Force specified IDs if they are available. If None, the index one higher than the\
            highest already existing index is selected and counted onwards.
    :type index: Iterable(int), default None
    :param in_service: True if the junctions are in service or False if they are out of service
    :type in_service: Iterable or boolean, default True
    :param type: not used yet - Designed for type differentiation on pandas lookups (e.g. \
            household connection vs. crossing)
    :type type: Iterable or string, default "junction"
    :param geodata: Coordinates used for plotting
    :type geodata: Iterable of (x,y)-tuples, default None
    :param kwargs: Additional keyword arguments will be added as further columns to the\
            net["junction"] table
    :return: index - The unique IDs of the created elements
    :rtype: array(int)

    :Example:
        >>> create_junctions(net, 200, pn_bar=5, tfluid_k=320,
        >>>                  height_m=np.arange(200))
    """
    add_new_component(net, Junction)

    index = _get_multiple_index_with_check(net, "junction", index, nr_junctions)
    entries = {"pn_bar": pn_bar, "type": type, "tfluid_k": tfluid_k, "height_m": height_m,
               "in_service": in_service, "name": name}
    _set_multiple_entries(net, "junction", index, **entries, **kwargs)

    if geodata is not None:
        # works with a 2-tuple or a matching array
        net.junction_geodata = pd.concat([net.junction_geodata, pd.DataFrame(
            np.zeros((len(index), len(net.junction_geodata.columns)), dtype=int), index=index,
            columns=net.junction_geodata.columns)])
        net.junction_geodata.loc[index, :] = np.nan
        net.junction_geodata.loc[index, ["x", "y"]] = geodata

    return index


def create_sinks(net, junctions, mdot_kg_per_s, scaling=1., name=None, index=None, in_service=True,
                 type='sink', **kwargs):
    """
    Convenience function for creating many sinks at once. Parameter 'junctions' must be an array \
    of the desired length. Other parameters may be either arrays of the same length or single \
    values.

    :param net: The net for which this sink should be created
    :type net: pandapipesNet
    :param junctions: The index of the junctions to which the sinks are connected
    :type junctions: Iterable(int)
    :param mdot_kg_per_s: The required mass flow
    :type mdot_kg_per_s: Iterable or float, default None
    :param scaling: An optional scaling factor to be set customly
    :type scaling: Iterable or float, default 1
    :param name: Name tags for the sinks
    :type name: Iterable or str, default None
    :param index: Force specified IDs if they are available. If None, the index one higher than the\
            highest already existing index is selected and counted onwards.
    :type index: Iterable(int), default None
    :param in_service: True for in service, False for out of service
    :type in_service: Iterable or bool, default True
    :param type: Type variables to classify the sinks
    :type type: Iterable or str, default None
    :param kwargs: Additional keyword arguments will be added as further columns to the\
            net["sink"] table
    :return: index - The unique IDs of the created elements
    :rtype: array(int)

    :Example:
        >>> new_sink_ids = create_sinks(net, junctions=[1, 5, 10],
        >>>                             mdot_kg_per_s=[0.1, 0.05, 0.2])
    """
    add_new_component(net, Sink)

    _check_multiple_junction_elements(net, junctions)
    index = _get_multiple_index_with_check(net, "sink", index, len(junctions))

    entries = {"junction": junctions, "mdot_kg_per_s": mdot_kg_per_s, "scaling": scaling,
               "in_service": in_service, "name": name, "type": type}
    _set_multiple_entries(net, "sink", index, **entries, **kwargs)

    return index


def create_sources(net, junctions, mdot_kg_per_s, scaling=1., name=None, index=None,
                   in_service=True, type='source', **kwargs):
    """
    Convenience function for creating many sources at once. Parameter 'junctions' must be an array \
    of the desired length. Other parameters may be either arrays of the same length or single \
    values.

    :param net: The net for which this source should be created
    :type net: pandapipesNet
    :param junctions: The index of the junctions to which the sources are connected
    :type junctions: Iterabl(int)
    :param mdot_kg_per_s: The required mass flow
    :type mdot_kg_per_s: Iterable or float, default None
    :param scaling: An optional scaling factor to be set customly
    :type scaling: Iterable or float, default 1
    :param name: Name tags for the sources
    :type name: Iterable or str, default None
    :param index: Force specified IDs if they are available. If None, the index one higher than the\
            highest already existing index is selected and counted onwards.
    :type index: Iterable(int), default None
    :param in_service: True for in service, False for out of service
    :type in_service: Iterable or bool, default True
    :param type: Type variable to classify the sources
    :type type: Iterable or str, default None
    :param kwargs: Additional keyword arguments will be added as further columns to the\
            net["source"] table
    :return: index - The unique IDs of the created elements
    :rtype: array(int)

    :Example:
        >>> new_source_ids = create_sources(net, junctions=[1, 5, 10],
        >>>                                 mdot_kg_per_s=[0.1, 0.05, 0.2])
    """
    add_new_component(net, Source)

    _check_multiple_junction_elements(net, junctions)
    index = _get_multiple_index_with_check(net, "source", index, len(junctions))

    entries = {"junction": junctions, "mdot_kg_per_s": mdot_kg_per_s, "scaling": scaling,
               "in_service": in_service, "name": name, "type": type}
    _set_multiple_entries(net, "source", index, **entries, **kwargs)

    return index


def create_ext_grids(net, junctions, p_bar, t_k, name=None, in_service=True, index=None,
                     type="auto", **kwargs):
    """
    Convenience function for creating many external grids at once. Parameter 'junctions' must be an\
    array of the desired length. Other parameters may be either arrays of the same length or single\
    values.\n
    External grids transfer the junction that it is connected into a node with fixed value for \
    either pressure, temperature or both (depending on the type). Usually external grids represent \
    connections to other grids feeding the given pandapipesNet.

    :param net: The net that the external grid should be connected to
    :type net: pandapipesNet
    :param junctions: The junctions to which the external grids are connected
    :type junctions: Iterable(int)
    :param p_bar: The pressure of the external grids
    :type p_bar: Iterable(float) or float
    :param t_k: The fixed temperature at the external grids
    :type t_k: Iterable(float) or float
    :param name: A name tag for the ext_grids
    :type name: Iterable(str) or str, default None
    :param in_service: True for in service, False for out of service
    :type in_service: Iterable(bool) or bool, default True
    :param index: Force specified IDs if they are available. If None, the index one higher than the\
            highest already existing index is selected and counted onwards.
    :type index: Iterable(int), default None
    :param type: The external grid type denotes the values that are fixed at the respective node:\n
            - "auto": Will automatically assign one of the following types based on the input for \
                      p_bar and t_k \n
            - "p": The pressure is fixed, the node acts as a slack node for the mass flow. \n
            - "t": The temperature is fixed and will not be solved for, but is assumed as the \
                   node's mix temperature. Please note that pandapipes cannot check for \
                   inconsistencies in the formulation of heat transfer equations yet. \n
            - "pt": The external grid shows both "p" and "t" behavior.
    :type type: Iterable(str) or str, default "auto"
    :param kwargs: Additional keyword arguments will be added as further columns to the\
            net["ext_grid"] table
    :return: index - The unique IDs of the created elements
    :rtype: Iterable(int)

    :Example:
        >>> create_ext_grids(net, junctions=[2, 4], p_bar=[100, 200], t_k=293.15)

    """
    add_new_component(net, ExtGrid)

    _check_multiple_junction_elements(net, junctions)
    index = _get_multiple_index_with_check(net, "ext_grid", index, len(junctions))

    type = _auto_ext_grid_types(p_bar, t_k, type, ExtGrid)

    entries = {"junction": junctions, "p_bar": p_bar, "t_k": t_k,
               "in_service": in_service, "name": name, "type": type}
    _set_multiple_entries(net, "ext_grid", index, **entries, **kwargs)

    return index


def create_pipes(net, from_junctions, to_junctions, std_type, length_km, k_mm=0.2,
                 loss_coefficient=0, sections=1, alpha_w_per_m2k=0., text_k=293, qext_w=0.,
                 name=None, index=None, geodata=None, in_service=True, type="pipe", **kwargs):
    """
    Convenience function for creating many pipes at once. Parameters 'from_junctions' and \
    'to_junctions' must be arrays of equal length. Other parameters may be either arrays of the \
    same length or single values. In any case the line parameters are defined through a single \
    standard type, so all pipes have the same standard type.

    :param net: The net for which this pipe should be created
    :type net: pandapipesNet
    :param from_junctions: IDs of the junctions on one side which the pipes will be connected to
    :type from_junctions: Iterable(int)
    :param to_junctions: IDs of the junctions on the other side to which the pipes will be \
            connected to
    :type to_junctions: Iterable(int)
    :param std_type: Name of standard type
    :type std_type: str
    :param length_km: Lengths of the pipes in [km]
    :type length_km: Iterable or float
    :param k_mm: Pipe roughness in [mm]. 0.2 mm is quite rough, usually betweeen 0.0015 (new
            pipes) and 0.3 (old steel pipelines)
    :type k_mm: Iterable or float, default 0.2
    :param loss_coefficient: Additional pressure loss coefficients, introduced by e.g. bends
    :type loss_coefficient: Iterable or float, default 0
    :param sections: The number of internal pipe sections. Important for gas and temperature\
            calculations, where variables are dependent on pipe length.
    :type sections: Iterable or int, default 1
    :param alpha_w_per_m2k: Heat transfer coefficients in [W/(m^2*K)]
    :type alpha_w_per_m2k: Iterable or float, default 0
    :param text_k: Ambient temperatures of pipes in [K]
    :type text_k: Iterable or float, default 293
    :param qext_w: External heat feed-in to the pipes in [W]
    :type qext_w: Iterable or float, default 0
    :param name: Name tags for these pipes
    :type name: Iterable or str, default None
    :param index: Force specified IDs if they are available. If None, the index one higher than the\
            highest already existing index is selected and counted onwards.
    :type index: Iterable(int), default None
    :param geodata: The coordinates of the pipes. The first row should be the coordinates of\
            junction a and the last should be the coordinates of junction b. The points in the\
            middle represent the bending points of the pipe.
    :type geodata: array, shape=(no_pipes,2L) or (,2L), default None
    :param in_service: True for in service, False for out of service
    :type in_service: Iterable or bool, default True
    :param type: Identifiers for special types of pipes (e.g. below or above ground)
    :type type: Iterable or str, default "pipe"
    :param kwargs: Additional keyword arguments will be added as further columns to the\
            net["pipe"] table
    :return: index - The unique IDs of the created elements
    :rtype: array(int)

    :Example:
        >>> pipe_indices = create_pipes(net,
        >>>                             from_junctions=[0, 2, 6],
        >>>                             to_junctions=[1, 3, 7],
        >>>                             std_type='315_PE_80_SDR_17',
        >>>                             length_km=[0.2, 1, 0.3])

    """
    add_new_component(net, Pipe)

    nr_pipes = len(from_junctions)
    index = _get_multiple_index_with_check(net, "pipe", index, nr_pipes)
    _check_branches(net, from_junctions, to_junctions, "pipe")
    _check_std_type(net, std_type, "pipe", "create_pipes")

    pipe_parameters = load_std_type(net, std_type, "pipe")
    entries = {"name": name, "from_junction": from_junctions, "to_junction": to_junctions,
               "std_type": std_type, "length_km": length_km,
               "diameter_m": pipe_parameters["inner_diameter_mm"] / 1000, "k_mm": k_mm,
               "loss_coefficient": loss_coefficient, "alpha_w_per_m2k": alpha_w_per_m2k,
               "sections": sections, "in_service": in_service, "type": type, "qext_w": qext_w,
               "text_k": text_k}
    _set_multiple_entries(net, "pipe", index, **entries, **kwargs)

    if geodata is not None:
        _add_multiple_branch_geodata(net, "pipe", geodata, index)
    return index


def create_pipes_from_parameters(net, from_junctions, to_junctions, length_km, diameter_m, k_mm=0.2,
                                 loss_coefficient=0, sections=1, alpha_w_per_m2k=0., text_k=293,
                                 qext_w=0., name=None, index=None, geodata=None, in_service=True,
                                 type="pipe", **kwargs):
    """
    Convenience function for creating many pipes at once. Parameters 'from_junctions' and \
    'to_junctions' must be arrays of equal length. Other parameters may be either arrays of the \
    same length or single values.

    :param net: The net for which this pipe should be created
    :type net: pandapipesNet
    :param from_junctions: IDs of the junctions on one side which the pipes will be connected to
    :type from_junctions: Iterable(int)
    :param to_junctions: IDs of the junctions on the other side to which the pipes will be \
            connected to
    :type to_junctions: Iterable(int)
    :param length_km: Lengths of the pipes in [km]
    :type length_km: Iterable or float
    :param diameter_m: The pipe diameters in [m]
    :type diameter_m: Iterable or float
    :param k_mm: Pipe roughness in [mm]. 0.2 mm is quite rough, usually betweeen 0.0015 (new
            pipes) and 0.3 (old steel pipelines)
    :type k_mm: Iterable or float, default 0.2 mm
    :param loss_coefficient: Additional pressure loss coefficients, introduced by e.g. bends
    :type loss_coefficient: Iterable or float, default 0
    :param sections: The number of internal pipe sections. Important for gas and temperature\
            calculations, where variables are dependent on pipe length.
    :type sections: Iterable or int, default 1
    :param alpha_w_per_m2k: Heat transfer coefficients in [W/(m^2*K)]
    :type alpha_w_per_m2k: Iterable or float, default 0
    :param text_k: Ambient temperatures of pipes in [K]
    :type text_k: Iterable or float, default 293
    :param qext_w: External heat feed-in to the pipes in [W]
    :type qext_w: Iterable or float, default 0
    :param name: Name tags for these pipes
    :type name: Iterable or str, default None
    :param index: Force specified IDs if they are available. If None, the index one higher than the\
            highest already existing index is selected and counted onwards.
    :type index: Iterable(int), default None
    :param geodata: The coordinates of the pipes. The first row should be the coordinates of\
            junction a and the last should be the coordinates of junction b. The points in the\
            middle represent the bending points of the pipe.
    :type geodata: array, shape=(no_pipes,2L) or (,2L), default None
    :param in_service: True for in service, False for out of service
    :type in_service: Iterable or bool, default True
    :param type: Identifiers for special types of pipes (e.g. below or above ground)
    :type type: Iterable or str, default "pipe"
    :param kwargs: Additional keyword arguments will be added as further columns to the\
            net["pipe"] table
    :return: index - The unique IDs of the created elements
    :rtype: array(int)

    :Example:
        >>> pipe_indices = create_pipes_from_parameters(net,
        >>>                                             from_junctions=[0, 2, 6],
        >>>                                             to_junctions=[1, 3, 7],
        >>>                                             length_km=[0.2, 1, 0.3],
        >>>                                             diameter_m=40e-3)

    """
    add_new_component(net, Pipe)

    index = _get_multiple_index_with_check(net, "pipe", index, len(from_junctions))
    _check_branches(net, from_junctions, to_junctions, "pipe")

    entries = {"name": name, "from_junction": from_junctions, "to_junction": to_junctions,
               "std_type": None, "length_km": length_km, "diameter_m": diameter_m, "k_mm": k_mm,
               "loss_coefficient": loss_coefficient, "alpha_w_per_m2k": alpha_w_per_m2k,
               "sections": sections, "in_service": in_service, "type": type, "qext_w": qext_w,
               "text_k": text_k}
    _set_multiple_entries(net, "pipe", index, **entries, **kwargs)

    if geodata is not None:
        _add_multiple_branch_geodata(net, "pipe", geodata, index)
    return index


def create_valves(net, from_junctions, to_junctions, diameter_m, opened=True, loss_coefficient=0,
                  name=None, index=None, type='valve', **kwargs):
    """
    Convenience function for creating many valves at once. Parameters 'from_junctions' and \
    'to_junctions' must be arrays of equal length. Other parameters may be either arrays of the \
    same length or single values.

    :param net: The net for which these valves should be created
    :type net: pandapipesNet
    :param from_junctions: IDs of the junctions on one side which the valves will be connected to
    :type from_junctions: Iterable(int)
    :param to_junctions: IDs of the junctions on the other side to which the valves will be \
            connected to
    :type to_junctions: Iterable(int)
    :param diameter_m: The valve diameters in [m]
    :type diameter_m: Iterable or float
    :param opened: Flag to show if the valves are opened and allow for fluid flow or if they are\
            closed to block the fluid flow.
    :type opened: Iterable or bool, default True
    :param loss_coefficient: The pressure loss coefficients introduced by the valve shapes
    :type loss_coefficient: Iterable or float, default 0
    :param name: Name tags for the valves
    :type name: Iterable or str, default None
    :param index: Force specified IDs if they are available. If None, the index one higher than the\
            highest already existing index is selected and counted onwards.
    :type index: Iterable(int), default None
    :param type: Identifiers for special types of valves (e.g. below or above ground)
    :type type: Iterable or str, default "valve"
    :param kwargs: Additional keyword arguments will be added as further columns to the\
            net["valve"] table
    :return: index - The unique IDs of the created elements
    :rtype: array(int)

    :Example:
        >>> create_valves(net, from_junctions=[0, 1, 4], to_junctions=[1, 5, 6],
        >>>               opened=[True, False, True], diameter_m=4e-3,
        >>>               name=["valve_%d" for d in range(3)])

    """
    add_new_component(net, Valve)

    index = _get_multiple_index_with_check(net, "valve", index, len(from_junctions))
    _check_branches(net, from_junctions, to_junctions, "valve")

    entries = {"name": name, "from_junction": from_junctions, "to_junction": to_junctions,
               "diameter_m": diameter_m, "opened": opened, "loss_coefficient": loss_coefficient,
               "type": type}
    _set_multiple_entries(net, "valve", index, **entries, **kwargs)

    return index


def create_pressure_controls(net, from_junctions, to_junctions, controlled_junctions,
                             controlled_p_bar, control_active=True, loss_coefficient=0., name=None,
                             index=None, in_service=True, type="pressure_control", **kwargs):
    """
    Convenience function for creating many pressure controls at once. Parameters 'from_junctions'\
    and 'to_junctions' must be arrays of equal length. Other parameters may be either arrays of the\
    same length or single values.

    Pressure control units enforce a pressure at a specific junction by creating a pressure drop /
    lift between the 'from' and the 'to' junction so that the pressure set point at the
    controlled junction is met.
    It is required that the controlled junction is hydraulically properly connected to the from
    and to junction and no other pessure control unit is inbetween.

    :param net: The net for which these pressure controls should be created
    :type net: pandapipesNet
    :param from_junctions: IDs of the junctions on one side which the pressure controls will be\
            connected to
    :type from_junctions: Iterable(int)
    :param to_junctions: IDs of the junctions on the other side to which the pressure controls will\
            be connected to
    :type to_junctions: Iterable(int)
    :param controlled_junctions: IDs of the junctions at which the pressure is controlled
    :type controlled_junctions: Iterable or int
    :param controlled_p_bar: Pressure set points
    :type controlled_p_bar: Iterable or float
    :param control_active: Variable to state whether the pressure control is active (otherwise \
        behaviour similar to open valve)
    :type control_active: bool, default True
    :param loss_coefficient: The pressure loss coefficient introduced by the component's shape \
        (used only if control is not active).
    :type loss_coefficient: float, default 0
    :param name: Name of the pressure control elements
    :type name: Iterable or str
    :param index: Force specified IDs if they are available. If None, the index one higher than the\
        highest already existing index is selected and counted onwards.
    :type index: Iterable(int), default None
    :param in_service: True if the pressure controls are in service or False if they are out of\
        service
    :type in_service: Iterable or bool, default True
    :param type: Currently not used - possibility to specify a certain type of pressure control
    :type type: Iterable or str, default "pressure_control"
    :param kwargs: Additional keyword arguments will be added as further columns to the \
            net["press_control"] table
    :return: index - The unique IDs of the created elements
    :rtype: array(int)

    :Example:
        Create one unit to connect junction 0 and 1,
        set the pressure at junction 1 to 5 bar.

        And create a 2nd unit to connect junction 2 and 4,
        set the pressure at junction 3 to 4.9 bar.

        >>> create_pressure_controls(net, [0, 2], [1, 4], [1, 3],
        >>>                          controlled_p_bar=[5, 4.9])

    """
    add_new_component(net, PressureControlComponent)

    index = _get_multiple_index_with_check(net, "press_control", index, len(from_junctions))
    _check_branches(net, from_junctions, to_junctions, "press_control")

    entries = {"name": name, "from_junction": from_junctions, "to_junction": to_junctions,
               "controlled_junction": controlled_junctions, "controlled_p_bar": controlled_p_bar,
               "control_active": control_active, "loss_coefficient": loss_coefficient,
               "in_service": in_service, "type": type}
    _set_multiple_entries(net, "press_control", index, **entries, **kwargs)

    controlled_elsewhere = (controlled_junctions != from_junctions) \
        & (controlled_junctions != to_junctions)
    if np.any(controlled_elsewhere):
        controllers_warn = index[controlled_elsewhere]
        logger.warning("The pressure controllers %s control the pressure at junctions that they are"
                       " not connected to. Please note that this can lead to errors in the pipeflow"
                       " calculation that will not be displayed properly. Make sure that your grid "
                       "configuration is valid." % controllers_warn)

    return index


def create_flow_controls(net, from_junctions, to_junctions, controlled_mdot_kg_per_s, diameter_m,
                         control_active=True, name=None, index=None, in_service=True, type="fc",
                         **kwargs):
    """
    Convenience function for creating many flow controls at once. Parameters 'from_junctions'\
    and 'to_junctions' must be arrays of equal length. Other parameters may be either arrays of the\
    same length or single values.

    :param net: The net for which these flow controls should be created
    :type net: pandapipesNet
    :param from_junctions: IDs of the junctions on one side which the flow controls will be\
            connected to
    :type from_junctions: Iterable(int)
    :param to_junctions: IDs of the junctions on the other side to which the pressure controls will\
            be connected to
    :type to_junctions: Iterable(int)
    :param controlled_mdot_kg_per_s: Mass flow set points
    :type controlled_mdot_kg_per_s: Iterable or float
    :param diameter_m: Measure of the diameter to derive the cross-sectional area (important for \
            the velocity calculation)
    :type diameter_m: Iterable or float
    :param control_active: Variable to state whether the flow control is active (otherwise \
            similar to open valve)
    :type control_active: bool, default True
    :param name: Name of the flow control elements
    :type name: Iterable or str
    :param index: Force specified IDs if they are available. If None, the index one higher than the\
            highest already existing index is selected and counted onwards.
    :type index: Iterable(int), default None
    :param in_service: True if the flow controls are in service or False if they are out of service
    :type in_service: Iterable or bool, default True
    :param type: Currently not used - possibility to specify a certain type of flow control
    :type type: Iterable or str, default "fc"
    :param kwargs: Additional keyword arguments will be added as further columns to the \
            net["flow_control"] table
    :return: index - The unique IDs of the created elements
    :rtype: array(int)

    :Example:
        Create two flow controllers with 0.8 m diameter between junction 0 and 1 with 0.5 kg/s and
        junction 2 and 4 with 0.9 kg/s, respectively.

        >>> create_flow_controls(net, [0, 2], [1, 4], [0.5, 0.9], [0.8, 0.8])

    """

    add_new_component(net, FlowControlComponent)

    index = _get_multiple_index_with_check(net, "flow_control", index, len(from_junctions))
    _check_branches(net, from_junctions, to_junctions, "flow_control")

    entries = {"name": name, "from_junction": from_junctions, "to_junction": to_junctions,
               "controlled_mdot_kg_per_s": controlled_mdot_kg_per_s, "diameter_m": diameter_m,
               "control_active": control_active, "in_service": in_service, "type": type}
    _set_multiple_entries(net, "flow_control", index, **entries, **kwargs)

    return index


def create_fluid_from_lib(net, name, overwrite=True):
    """
    Creates a fluid from library (if there is an entry) and sets net["fluid"] to this value.
    Currently, existing fluids in the library are: "hgas", "lgas", "hydrogen", "methane", "water",
    "air".

    :param net: The net for which this fluid should be created
    :type net: pandapipesNet
    :param name: The name of the fluid that shall be extracted from the fluid lib
    :type name: str
    :param overwrite: Flag if a possibly existing fluid in the net shall be overwritten
    :type overwrite: bool, default True
    :return: No output

    :Example:
        >>> create_fluid_from_lib(net, name="water")

    """
    _add_fluid_to_net(net, call_lib(name), overwrite=overwrite)


def _check_multiple_junction_elements(net, junctions):
    return _check_multiple_node_elements(net, junctions, node_table="junction", name="junctions")


def _check_junction_element(net, junction):
    return _check_node_element(net, junction, node_table="junction")


def _check_branch(net, element_name, index, from_junction, to_junction):
    return _check_branch_element(net, element_name, index, from_junction, to_junction,
                                 node_name="junction", plural="s")


def _check_branches(net, from_junctions, to_junctions, table):
    return _check_multiple_branch_elements(net, from_junctions, to_junctions, table,
                                           node_name="junction", plural="s")


def _check_std_type(net, std_type, table, function_name):
    if 'std_types' not in net:
        raise UserWarning('%s is defined as std_type in %s but there are no std_types '
                          'defined in your net. You need to define a std_type first or set '
                          'add_stdtypes=True in create_empty_network.' % (std_type, function_name))
    if std_type not in net['std_types'][table]:
        raise UserWarning('%s is not given in std_types (%s). Either change std_type or define new '
                          'one' % (std_type, table))


ALLOWED_EG_TYPES = ["auto", "t", "p", "pt", "tp"]


def _auto_ext_grid_type(p_bar, t_k, typ, comp):
    """
    Determine the type of node that an "ext_grid" would imply (fixed pressure and / or temperature).
    Also perform some validity checks.

    :param p_bar: fixed pressure
    :type p_bar: float
    :param t_k: fixed temperature
    :type t_k: float
    :param typ: given type of ext_grid
    :type typ: str
    :param comp: component (just needed for output)
    :type comp: pandapipes component
    :return: adapted_type
    :rtype: str
    """
    p_null, t_null = p_bar is None or np.isnan(p_bar), t_k is None or np.isnan(t_k)

    if p_null and t_null:
        raise UserWarning("For component %s, either pressure or temperature must be defined!"
                          % comp.__name__)

    if typ not in ALLOWED_EG_TYPES:
        logger.warning("The type for component %s was %s, but must be one of the following for "
                       "correct model implementation: %s." % (comp.__name__, typ, ALLOWED_EG_TYPES))

    if typ not in ["t", "auto"] and p_null:
        raise UserWarning("The type %s for component %s requires a pressure as input!"
                          % (typ, comp.__name__))

    if typ not in ["p", "auto"] and t_null:
        raise UserWarning("The type %s for component %s requires a temperature as input!"
                          % (typ, comp.__name__))

    if typ != "auto" and "p" not in typ and not p_null:
        logger.warning("For component %s you gave a value for p, although the component is not of "
                       "'p'-type (type is %s), i.e. this value is probably neglected internally."
                       % (comp, typ))

    if typ != "auto" and "t" not in typ and not t_null:
        logger.warning("For component %s you gave a value for t, although the component is not of "
                       "'t'-type (type is %s), i.e. this value is probably neglected internally."
                       % (comp, typ))

    if typ == "tp":
        logger.warning("The type 'tp' is replaced by type 'pt' (might be mistaken internally).")
        return "pt"

    if typ == "auto":
        real_type = ""
        if not p_null:
            real_type += "p"
        if not t_null:
            real_type += "t"
        return real_type

    return typ


def _auto_ext_grid_types(p_bar, t_k, typ, comp):
    """
    Determine the type of node that an "ext_grid" would imply (fixed pressure and / or temperature).
    Also perform some validity checks. --> Same as `_auto_ext_grid_type`, but vectorized.

    .. note:
        This function is currently not used yet. Can be used, if bulk functions for ext_grid-like \
        components are added.

    :param p_bar: fixed pressure values
    :type p_bar: iterable(float)
    :param t_k: fixed temperature values
    :type t_k: iterable(float)
    :param typ: given types of ext_grid
    :type typ: iterable(str)
    :param comp: component (just needed for output)
    :type comp: pandapipes component
    :return: adapted_types
    :rtype: iterable(str)
    """
    p_arr = hasattr(p_bar, "__iter__")
    t_arr = hasattr(t_k, "__iter__")
    typ_arr = hasattr(typ, "__iter__") and not isinstance(typ, str)

    if not any([p_arr, t_arr, typ_arr]):
        return _auto_ext_grid_type(p_bar, t_k, typ, comp)
    else:
        lengths = np.array([len(arr) for arr, it in zip([p_bar, t_k, typ], [p_arr, t_arr, typ_arr])
                            if it])
        if not np.array_equal(lengths[1:], lengths[:-1]):
            raise UserWarning("The arrays for p_bar, t_k and type must have equal length!")
        length = lengths[0]

    if not p_arr:
        p_bar = np.array([p_bar] * length)

    if not t_arr:
        t_k = np.array([t_k] * length)

    if not typ_arr:
        typ = np.array([typ] * length)

    p_null = np.equal(p_bar, None) | np.isnan(p_bar)
    t_null = np.equal(t_k, None) | np.isnan(t_k)

    ptn = p_null & t_null
    if np.any(ptn):
        invalid_ind = np.where(ptn)
        raise UserWarning("For component %s in positions %s, either pressure or temperature must be"
                          " defined!" % (comp.__name__, invalid_ind))

    not_allowed_types = ~np.isin(typ, ALLOWED_EG_TYPES)
    if np.any(not_allowed_types):
        typ_not_allowed_types = typ[not_allowed_types]
        pos_not_allowed_types = np.where(not_allowed_types)
        overview = pd.DataFrame({"Position": pos_not_allowed_types, "Type": typ_not_allowed_types})
        logger.warning("Please check the following types for component %s, as they must be one of "
                       "the following for correct model implementation: %s. \n%s"
                       % (comp.__name__, ALLOWED_EG_TYPES, overview))

    p_types = np.isin(typ, [tp for tp in ALLOWED_EG_TYPES if tp not in ["t", "auto"]])
    t_types = np.isin(typ, [tp for tp in ALLOWED_EG_TYPES if tp not in ["p", "auto"]])

    invalid_p = p_types & p_null
    if np.any(invalid_p):
        invalid_types = typ[invalid_p]
        invalid_ind = np.where(invalid_p)
        raise UserWarning("The types %s for component %s (positions %s) require a pressure as "
                          "input!" % (invalid_types, invalid_ind, comp.__name__))

    invalid_t = t_types & t_null
    if np.any(invalid_t):
        invalid_types = typ[invalid_t]
        invalid_ind = np.where(invalid_t)
        raise UserWarning("The types %s for component %s (positions %s) require a temperature as "
                          "input!" % (invalid_types, invalid_ind, comp.__name__))

    auto_types = np.isin(typ, ["auto"])

    ununsed_p = ~auto_types & ~p_null & ~p_types
    if np.any(ununsed_p):
        overview = pd.DataFrame({"Positions": np.where(ununsed_p)[0],
                                 "Type": np.array(typ)[ununsed_p],
                                 "p_value": np.array(p_bar)[ununsed_p]})
        logger.warning("For component %s you gave a value for p in some cases, although the "
                       "respective components are not of 'p'-type, i.e. the given values are "
                       "probably neglected internally. \n%s" % (comp, overview))

    ununsed_t = ~auto_types & ~t_null & ~t_types
    if np.any(ununsed_t):
        overview = pd.DataFrame({"Positions": np.where(ununsed_t)[0],
                                 "Type": np.array(typ)[ununsed_t],
                                 "t_value": np.array(p_bar)[ununsed_t]})
        logger.warning("For component %s you gave a value for t in some cases, although the "
                       "respective components are not of 't'-type, i.e. the given values are "
                       "probably neglected internally. \n%s" % (comp, overview))

    real_types = np.array(typ).copy()
    tp_type = np.isin(real_types, ["tp"])
    if np.any(tp_type):
        tp_ind = np.where(tp_type)
        logger.warning("The type 'tp' (positions %s) is replaced by type 'pt' (otherwise might be "
                       "mistaken internally)." % tp_ind)
        real_types[tp_type] = "pt"

    if np.any(auto_types):
        real_types[auto_types & ~p_null & t_null] = "p"
        real_types[auto_types & p_null & ~t_null] = "t"
        real_types[auto_types & ~p_null & ~t_null] = "pt"

    return real_types
