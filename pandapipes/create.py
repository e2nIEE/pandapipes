# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
import pandas as pd
from pandapipes.component_models.auxiliaries.component_toolbox import add_new_component
from pandapipes.pandapipes_net import pandapipesNet, get_default_pandapipes_structure
from pandapipes.properties import call_lib, add_fluid_to_net
from pandapower.auxiliary import get_free_id, _preserve_dtypes
from pandapipes.properties.fluids import Fluid
from pandapipes.std_types.std_type import PumpStdType, add_basic_std_types, add_pump_std_type, \
    load_std_type
from pandapipes.std_types.std_type_toolbox import regression_function
from pandapipes.component_models import Junction, Sink, Source, Pump, Pipe, ExtGrid, \
    HeatExchanger, Valve, CirculationPumpPressure, CirculationPumpMass

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def create_empty_network(name="", fluid=None, add_stdtypes=True):
    """
    This function initializes the pandapipes datastructure.

    :param name: name for the network
    :type name: string, default None
    :param fluid: A fluid that can be added to the net from the start. Should be either of type\
            Fluid (c.f. pandapipes.properties.fluids.Fluid) or a string which refers to a standard\
            fluid type used to call `create_fluid_from_lib`. A fluid is required for pipeflow\
            calculations, but can also be added later.
    :type fluid: Fluid or str, default None
    :param add_stdtypes: flag whether to add a dictionary of typical pump and pipe std types
    :type add_stdtypes: bool, default True
    :return: net - pandapipesNet with empty tables
    :rtype: pandapipesNet

    :Example:
        >>> net1 = create_empty_network("my_first_pandapipesNet", "lgas")
        >>> net2 = create_empty_network()

    """
    net = pandapipesNet(get_default_pandapipes_structure())
    add_new_component(net, Junction, True)
    add_new_component(net, Pipe, True)
    add_new_component(net, ExtGrid, True)
    net['controller'] = pd.DataFrame(np.zeros(0, dtype=net['controller']), index=[])
    net['name'] = name
    if add_stdtypes:
        add_basic_std_types(net)

    net["name"] = name
    if fluid is not None:
        if isinstance(fluid, Fluid):
            net["fluid"] = fluid
        elif isinstance(fluid, str):
            create_fluid_from_lib(net, fluid)
        else:
            logger.warning("The fluid %s cannot be added to the net Only fluids of type Fluid or "
                           "strings can be used." % fluid)
    return net


def create_junction(net, pn_bar, tfluid_k, height_m=0, name=None, index=None, in_service=True,
                    type="junction", geodata=None, **kwargs):
    """
    Adds one junction in table net["junction"].

    Junctions are the nodes of the network that all other elements connect to.

    :param net: The pandapipes network in which the element is created
    :type net: pandapipesNet
    :param pn_bar: The nominal pressure in [bar]. Used as an initial value for pressure calculation.
    :type pn_bar: float
    :param tfluid_k: The fluid temperature in [K]. Used as parameter for gas calculations and as\
            initial value for temperature calculations
    :type tfluid_k: float
    :param height_m: Height of node above sea level in [m]
    :type height_m: float, default 0
    :param name: the name for this junction
    :type name: string, default None
    :param index: Force a specified ID if it is available. If None, the index one higher than the \
            highest already existing index is selected.
    :type index: int, default None
    :param in_service: True for in_service or False for out of service
    :type in_service: boolean, default True
    :param type: not used yet - designed for type differentiation on pandas lookups (e.g. household\
            connection vs. crossing)
    :type type: string, default "junction"
    :param geodata: coordinates used for plotting
    :type geodata: (x,y)-tuple, default None
    :param kwargs: Additional keyword arguments will be added as further columns to the\
            net["junction"] table
    :return: index - The unique ID of the created element
    :rtype: int

    :Example:
        >>> create_junction(net, pn_bar=5, tfluid_k=320)
    """
    add_new_component(net, Junction)

    if index and index in net["junction"].index:
        raise UserWarning("A junction with index %s already exists" % index)

    if index is None:
        index = get_free_id(net["junction"])

    # store dtypes
    dtypes = net.junction.dtypes
    cols = ["name", "pn_bar", "tfluid_k", "height_m", "in_service", "type"]
    vals = [name, pn_bar, tfluid_k, height_m, bool(in_service), type]

    all_values = {col: val for col, val in zip(cols, vals)}
    all_values.update(kwargs)
    for col, val in all_values.items():
        net.junction.at[index, col] = val

    # and preserve dtypes
    _preserve_dtypes(net.junction, dtypes)

    if geodata is not None:
        if len(geodata) != 2:
            raise UserWarning("geodata must be given as (x, y) tupel")
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
    :param index: Force a specified ID if it is available. If None, the index one higher than the \
            highest already existing index is selected.
    :type index: int, default None
    :param in_service: True for in service, false for out of service
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

    if junction not in net["junction"].index.values:
        raise UserWarning("Cannot attach to junction %s, junction does not exist" % junction)

    if index is None:
        index = get_free_id(net["sink"])

    if index in net["sink"].index:
        raise UserWarning("A sink with the id %s already exists" % index)

    # store dtypes
    dtypes = net.sink.dtypes

    cols = ["name", "junction", "mdot_kg_per_s", "scaling", "in_service", "type"]
    vals = [name, junction, mdot_kg_per_s, scaling, bool(in_service), type]
    all_values = {col: val for col, val in zip(cols, vals)}
    all_values.update(kwargs)
    for col, val in all_values.items():
        net.sink.at[index, col] = val

    # and preserve dtypes
    _preserve_dtypes(net.sink, dtypes)

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
    :param index: Force a specified ID if it is available. If None, the index one higher than the \
            highest already existing index is selected.
    :type index: int, default None
    :param in_service: True for in service, false for out of service
    :type in_service: bool, default True
    :param type: Type variable to classify the sink
    :type type: str, default None
    :param kwargs: Additional keyword arguments will be added as further columns to the\
            net["source"] table
    :return: index - The unique ID of the created element
    :rtype: int

    :Example:
        >>> create_source(net,junction=2,mdot_kg_per_s=0.1)

    """
    add_new_component(net, Source)

    if junction not in net["junction"].index.values:
        raise UserWarning("Cannot attach to junction %s, junction does not exist" % junction)

    if index is None:
        index = get_free_id(net["source"])

    if index in net["source"].index:
        raise UserWarning("A sink with the id %s already exists" % index)

    # store dtypes
    dtypes = net.source.dtypes

    cols = ["name", "junction", "mdot_kg_per_s", "scaling", "in_service", "type"]
    vals = [name, junction, mdot_kg_per_s, scaling, bool(in_service), type]
    all_values = {col: val for col, val in zip(cols, vals)}
    all_values.update(kwargs)
    for col, val in all_values.items():
        net.source.at[index, col] = val

    # and preserve dtypes
    _preserve_dtypes(net.source, dtypes)

    return index


def create_ext_grid(net, junction, p_bar, t_k, name=None, in_service=True, index=None, type="pt"):
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
    :type p_bar: float
    :param t_k: The fixed temperature at the external grid
    :type t_k: float, default 285.15
    :param name: A name tag for this ext_grid
    :type name: str, default None
    :param in_service: True for in service, false for out of service
    :type in_service: bool, default True
    :param index: Force a specified ID if it is available. If None, the index one higher than the \
            highest already existing index is selected.
    :param type: The external grid type denotes the values that are fixed at the respective node:\
            - "p": The pressure is fixed, the node acts as a slack node for the mass flow.\
            - "t": The temperature is fixed and will not be solved for, but is assumed as the \
                   node's mix temperature. Please note that pandapipes cannot check for \
                   inconsistencies in the formulation of heat transfer equations yet.
            - "pt": The external grid shows both "p" and "t" behasior.
    :type type: str, default "pt"
    :type index: int, default None
    :return: index - The unique ID of the created element
    :rtype: int

    :Example:
        >>> create_ext_grid(net, junction=2, p_bar=100, t_k=293.15)

    """
    add_new_component(net, ExtGrid)

    if not type in ["p", "t", "pt"]:
        logger.warning("no proper type was chosen.")

    if junction not in net["junction"].index.values:
        raise UserWarning("Cannot attach to bus %s, bus does not exist" % junction)

    if index is not None and index in net["ext_grid"].index:
        raise UserWarning("An external grid with with index %s already exists" % index)

    if index is None:
        index = get_free_id(net["ext_grid"])

    # store dtypes
    dtypes = net.ext_grid.dtypes

    net.ext_grid.loc[index, ["name", "junction", "p_bar", "t_k", "in_service", "type"]] = \
        [name, junction, p_bar, t_k, bool(in_service), type]

    # and preserve dtypes
    _preserve_dtypes(net.ext_grid, dtypes)
    return index


def create_heat_exchanger(net, from_junction, to_junction, diameter_m, qext_w, loss_coefficient=0,
                          name=None, index=None, in_service=True, type="heat_exchanger", **kwargs):
    """
    Creates a heat exchanger element in net["heat_exchanger"] from heat exchanger parameters.

    :param net: The net within this heat exchanger should be created
    :type net: pandapipesNet
    :param from_junction: ID of the junction on one side which the heat exchanger  will be \
            connected with
    :type from_junction: int
    :param to_junction: ID of the junction on the other side which the heat exchanger  will be \
            connected with
    :type to_junction: int
    :param diameter_m: The heat exchanger inner diameter im [m]
    :type diameter_m: float
    :param qext_w: external heat feed-in through the heat exchanger in [W]
    :type qext_w: float, default 0.0
    :param loss_coefficient: An additional pressure loss coefficient, introduced by e.g. bends
    :type loss_coefficient: float
    :param name: The name of the heat exchanger
    :type name: str, default None
    :param index: Force a specified ID if it is available. If None, the index one higher than the \
            highest already existing index is selected.
    :type index: int, default None
    :param in_service: True for in_service or False for out of service
    :type in_service: bool, default True
    :param type: Not used yet
    :type type: str
    :param kwargs: Additional keyword arguments will be added as further columns to the\
                    net["heat_exchanger"] table
    :return: index - The unique ID of the created heat exchanger
    :rtype: int

    EXAMPLE:
        create_heat_exchanger(net, from_junction=0, to_junction=1, diameter_m=40e-3, qext_w=2000)
    """
    add_new_component(net, HeatExchanger)

    # check if junction exist to attach the heat exchanger to
    for b in [from_junction, to_junction]:
        if b not in net["junction"].index.values:
            raise UserWarning("Heat exchanger %s tries to attach to non-existing junction %s"
                              % (name, b))

    if index is None:
        index = get_free_id(net["heat_exchanger"])

    if index in net["heat_exchanger"].index:
        raise UserWarning("A heat exchanger with index %s already exists" % index)

    v = {"name": name, "from_junction": from_junction, "to_junction": to_junction,
         "diameter_m": diameter_m, "qext_w": qext_w, "loss_coefficient": loss_coefficient,
         "in_service": bool(in_service), "type": type}
    v.update(kwargs)

    # store dtypes
    dtypes = net.heat_exchanger.dtypes

    for col, val in v.items():
        net.heat_exchanger.at[index, col] = val

    # and preserve dtypes
    _preserve_dtypes(net.heat_exchanger, dtypes)

    return index


def create_pipe(net, from_junction, to_junction, std_type, length_km, k_mm=1, loss_coefficient=0,
                sections=1, alpha_w_per_m2k=0., text_k=293, qext_w=0., name=None, index=None,
                geodata=None, in_service=True, type="pipe", **kwargs):
    """
    Creates a pipe element in net["pipe"] from pipe parameters.

    :param net: The net for which this pipe should be created
    :type net: pandapipesNet
    :param from_junction: ID of the junction on one side which the pipe will be connected to
    :type from_junction: int
    :param to_junction: ID of the junction on the other side which the pipe will be connected to
    :type to_junction: int
    :param std_type: name of standard type
    :type std_type: str
    :param length_km: Length of the pipe in [km]
    :type length_km: float
    :param k_mm: Pipe roughness in [mm]
    :type k_mm: float, default 1
    :param loss_coefficient: An additional pressure loss coefficient, introduced by e.g. bends
    :type loss_coefficient: float, default 0
    :param sections: The number of internal pipe sections. Important for gas and temperature\
            calculations, where variables are dependent on pipe length.
    :type sections: int, default 1
    :param alpha_w_per_m2k: Heat transfer coefficient in [W/(m^2*K)]
    :type alpha_w_per_m2k: float, default 0
    :param name: A name tag for this pipe
    :type name: str, default None
    :param index: Force a specified ID if it is available. If None, the index one higher than the \
            highest already existing index is selected.
    :type index: int, default None
    :param geodata: The coordinates of the pipe. The first row should be the coordinates of \
            junction a and the last should be the coordinates of junction b. The points in the\
            middle represent the bending points of the pipe
    :type geodata: array, shape= (,2L), default None
    :param in_service: True for in service, false for out of service
    :type in_service: bool, default True
    :param type: An identifier for special types of pipes (e.g. below or above ground)
    :type type: str, default "pipe"
    :param qext_w: external heat feed-in to the pipe in [W]
    :type qext_w: float, default 0
    :param text_k: Ambient temperature of pipe in [K]
    :type text_k: float, default 293
    :param kwargs: Additional keyword arguments will be added as further columns to the\
            net["pipe"] table
    :return: index - The unique ID of the created element
    :rtype: int

    :Example:
        >>> create_pipe(net,from_junction=0,to_junction=1,std_type='315_PE_80_SDR_17',length_km=1)

    """
    add_new_component(net, Pipe)

    # check if junction exist to attach the pipe to
    for b in [from_junction, to_junction]:
        if b not in net["junction"].index.values:
            raise UserWarning("Pipe %s tries to attach to non-existing junction %s"
                              % (name, b))

    if index is None:
        index = get_free_id(net["pipe"])

    if index in net["pipe"].index:
        raise UserWarning("A pipe with index %s already exists" % index)

    if 'std_type' not in net:
        raise UserWarning('%s is defined as std_type in create_pipe_from_std_type but there are no'
                          ' std_types defined in your net. You need to define a std_type first or '
                          'set add_stdtypes=True in create_empty_network.' % std_type)
    if std_type not in net['std_type']['pipe']:
        raise UserWarning('%s is not given in std_type. Either change std_type or define new '
                          'one' % std_type)
    pipe_parameter = load_std_type(net, std_type, "pipe")
    v = {"name": name, "from_junction": from_junction, "to_junction": to_junction,
         "std_type": std_type, "length_km": length_km, "diameter_m":
             pipe_parameter["inner_diameter_mm"] / 1000, "k_mm": k_mm,
         "loss_coefficient": loss_coefficient, "alpha_w_per_m2k": alpha_w_per_m2k,
         "sections": sections, "in_service": bool(in_service), "type": type, "qext_w": qext_w,
         "text_k": text_k}
    v.update(kwargs)

    # store dtypes
    dtypes = net.pipe.dtypes

    for col, val in v.items():
        net.pipe.at[index, col] = val

    # and preserve dtypes
    _preserve_dtypes(net.pipe, dtypes)

    if geodata is not None:
        net["pipe_geodata"].at[index, "coords"] = geodata

    return index


def create_pipe_from_parameters(net, from_junction, to_junction, length_km, diameter_m, k_mm=1,
                                loss_coefficient=0, sections=1, alpha_w_per_m2k=0., text_k=293,
                                qext_w=0., name=None, index=None, geodata=None, in_service=True,
                                type="pipe", **kwargs):
    """
    Creates a pipe element in net["pipe"] from pipe parameters.

    :param net: The net for which this pipe should be created
    :type net: pandapipesNet
    :param from_junction: ID of the junction on one side which the pipe will be connected with
    :type from_junction: int
    :param to_junction: ID of the junction on the other side which the pipe will be connected with
    :type to_junction: int
    :param length_km: Length of the pipe in [km]
    :type length_km: float
    :param diameter_m: The pipe diameter im [m]
    :type diameter_m: float
    :param k_mm: Pipe roughness in [mm]
    :type k_mm: float, default 1
    :param loss_coefficient: An additional pressure loss coefficient, introduced by e.g. bends
    :type loss_coefficient: float, default 0
    :param alpha_w_per_m2k: Heat transfer coefficient in [W/(m^2*K)]
    :type alpha_w_per_m2k: float, default 0
    :param sections: The number of internal pipe sections. Important for gas and temperature\
            calculations, where variables are dependent on pipe length.
    :type sections: int, default 1
    :param name: A name tag for this pipe
    :type name: str, default None
    :param index: Force a specified ID if it is available. If None, the index one higher than the \
            highest already existing index is selected.
    :type index: int, default None
    :param geodata: The coordinates of the pipe. The first row should be the coordinates of \
            junction a and the last should be the coordinates of junction b. The points in the\
            middle represent the bending points of the pipe
    :type geodata: array, shape= (,2L), default None
    :param in_service: True for in service, false for out of service
    :type in_service: bool, default True
    :param type: An identifier for special types of pipes (e.g. below or above ground)
    :type type: str, default "pipe"
    :param qext_w: external heat feed-in to the pipe in [W]
    :type qext_w: float, default 0
    :param text_k: Ambient temperature of pipe in [K]
    :type text_k: float, default 293
    :param kwargs: Additional keyword arguments will be added as further columns to the\
            net["pipe"] table
    :return: index - The unique ID of the created element
    :rtype: int

    :Example:
        >>> create_pipe_from_parameters(net,from_junction=0,to_junction=1,length_km=1,diameter_m=40e-3)

    """
    add_new_component(net, Pipe)

    # check if junction exist to attach the pipe to
    for b in [from_junction, to_junction]:
        if b not in net["junction"].index.values:
            raise UserWarning("Pipe %s tries to attach to non-existing junction %s"
                              % (name, b))

    if index is None:
        index = get_free_id(net["pipe"])

    if index in net["pipe"].index:
        raise UserWarning("A pipe with index %s already exists" % index)

    v = {"name": name, "from_junction": from_junction, "to_junction": to_junction,
         "std_type": None, "length_km": length_km, "diameter_m": diameter_m, "k_mm": k_mm,
         "loss_coefficient": loss_coefficient, "alpha_w_per_m2k": alpha_w_per_m2k,
         "sections": sections, "in_service": bool(in_service),
         "type": type, "qext_w": qext_w, "text_k": text_k}
    if 'std_type' in kwargs:
        raise UserWarning('you have defined a std_type, however, using this function you can only'
                          'create a pipe setting specific, individual parameters. If you want to '
                          'create a pipe from net.std_type please use create_pipe')
    v.update(kwargs)

    # store dtypes
    dtypes = net.pipe.dtypes

    for col, val in v.items():
        net.pipe.at[index, col] = val

    # and preserve dtypes
    _preserve_dtypes(net.pipe, dtypes)

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
    :param index: Force a specified ID if it is available. If None, the index one higher than the \
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

    # check if junction exist to attach the pipe to
    for b in [from_junction, to_junction]:
        if b not in net["junction"].index.values:
            raise UserWarning("Valve %s tries to attach to non-existing junction %s"
                              % (name, b))

    if index is None:
        index = get_free_id(net["valve"])

    if index in net["valve"].index:
        raise UserWarning("A valve with index %s already exists" % index)

    v = {"name": name, "from_junction": from_junction, "to_junction": to_junction,
         "diameter_m": diameter_m,
         "opened": opened, "loss_coefficient": loss_coefficient, "type": type}
    v.update(kwargs)
    # store dtypes
    dtypes = net.valve.dtypes

    for col, val in v.items():
        net.valve.at[index, col] = val

    # and preserve dtypes
    _preserve_dtypes(net.valve, dtypes)

    return index


def create_pump(net, from_junction, to_junction, std_type, name=None, index=None, in_service=True,
                type="pump", **kwargs):
    """
    Adds one pump in table net["pump"].

    :param net: The net within this pump should be created
    :type net: pandapipesNet
    :param from_junction: ID of the junction on one side which the pump will be connected with
    :type from_junction: int
    :param to_junction: ID of the junction on the other side which the pump will be connected with
    :type to_junction: int
    :param std_type: There are currently three different std_types. This std_types are P1, P2, P3. \
            Each of them describes a specific pump behaviour setting volume flow and pressure in \
            context.
    :type std_type: string, default None
    :param name: A name tag for this pump
    :type name: str, default None
    :param index: Force a specified ID if it is available. If None, the index one higher than the\
            highest already existing index is selected.
    :type index: int, default None
    :param in_service: True for in_service or False for out of service
    :type in_service: bool, default True
    :param type:  type variable to classify the pump
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

    for b in [from_junction, to_junction]:
        if b not in net["junction"].index.values:
            raise UserWarning("Pump %s tries to attach to non-existing junction %s" % (name, b))

    if index is None:
        index = get_free_id(net["pump"])
    if index in net["pump"].index:
        raise UserWarning("A pump with the id %s already exists" % id)

    # store dtypes
    dtypes = net.pump.dtypes

    if 'std_type' not in net:
        raise UserWarning('%s is defined as std_type in create_pump but there are no std_types'
                          ' defined in your net. You need to define a std_type first.'
                          % std_type)
    if std_type not in net['std_type']['pump']:
        raise UserWarning('%s is not given in std_type. Either change std_type or define new '
                          'one' % std_type)
    v = {"name": name, "from_junction": from_junction, "to_junction": to_junction,
         "std_type": std_type, "in_service": bool(in_service), "type": type}
    v.update(kwargs)
    # and preserve dtypes
    for col, val in v.items():
        net.pump.at[index, col] = val
    _preserve_dtypes(net.pump, dtypes)

    return index


def create_pump_from_parameters(net, from_junction, to_junction, pump_name, pressure_list=None,
                                flowrate_list=None, regression_degree=None,
                                regression_parameters=None, name=None, index=None, in_service=True,
                                type="pump", **kwargs):
    """
    Adds one pump in table net["pump"].

    :param net: The net within this pump should be created
    :type net: pandapipesNet
    :param from_junction: ID of the junction on one side which the pump will be connected with
    :type from_junction: int
    :param to_junction: ID of the junction on the other side which the pump will be connected with
    :type to_junction: int
    :param pump_name: Set a name for your pump. You will find your definied pump under std_type in\
            your net. The name will be given under std_type in net.pump.
    :type pump_name: string
    :param pressure_list: This list contains measured pressure supporting points required \
            to define and determine the dependencies of the pump between pressure and volume flow. \
            The pressure must be given in [bar]. Needs to be defined only if no pump of standard \
            type is selected.
    :type pressure_list: list, default None
    :param flowrate_list: This list contains the corresponding flowrate values to the given \
            pressure values. Thus the length must be equal to the pressure list. Needs to be \
            defined only if no pump of standard type is selected. ATTENTION: The flowrate values \
            are given in :math:`[\\frac{m^3}{h}]`.
    :type flowrate_list: list, default None
    :param regression_degree: The regression degree must be defined if pressure and flowrate list \
            are given. It describes the degree of the regression function polynomial describing \
            the behaviour of the pump.
    :type regression_degree: int, default None
    :param regression_parameters: Alternatviely to taking measurement values \
            also the already calculated regression parameters can be given. It describes the \
            dependency between pressure and flowrate. ATTENTION: The determined parameteres must \
            be retrieved by setting flowrate given in :math:`[\\frac{m^3}{h}]` and pressure given \
            in bar in context.
    :type regression_parameters: list, default None
    :param name: A name tag for this pump
    :type name: str, default None
    :param index: Force a specified ID if it is available. If None, the index one higher than the\
            highest already existing index is selected.
    :type index: int, default None
    :param in_service: True for in_service or False for out of service
    :type in_service: bool, default True
    :param type:  type variable to classify the pump
    :type type: str, default "pump"
    :param kwargs: Additional keyword arguments will be added as further columns to the\
            net["pump"] table
    :type kwargs: dict
    :return: index - The unique ID of the created element
    :rtype: int

    EXAMPLE:
        >>> create_pump_from_parameters(net, 0, 1, 'pump1', pressure_list=[0,1,2,3], \
                                        flowrate_list=[0,1,2,3], regression_degree=1)
        >>> create_pump_from_parameters(net, 0, 1, 'pump2', regression_parameters=[1,0])

    """
    add_new_component(net, Pump)

    for b in [from_junction, to_junction]:
        if b not in net["junction"].index.values:
            raise UserWarning("Pump %s tries to attach to non-existing junction %s" % (name, b))

    if index is None:
        index = get_free_id(net["pump"])
    if index in net["pump"].index:
        raise UserWarning("A pump with the id %s already exists" % id)

    # store dtypes
    dtypes = net.pump.dtypes

    if pressure_list is not None and flowrate_list is not None and regression_degree is not None:
        reg_par = regression_function(pressure_list, flowrate_list, regression_degree)
        pump = PumpStdType(pump_name, reg_par)
        add_pump_std_type(net, pump_name, pump)
    elif regression_parameters is not None:
        pump = PumpStdType(pump_name, regression_parameters)
        add_pump_std_type(net, pump_name, pump)

    v = {"name": name, "from_junction": from_junction, "to_junction": to_junction,
         "std_type": pump_name, "in_service": bool(in_service), "type": type}
    v.update(kwargs)
    # and preserve dtypes
    for col, val in v.items():
        net.pump.at[index, col] = val
    _preserve_dtypes(net.pump, dtypes)

    return index


def create_circ_pump_const_pressure(net, from_junction, to_junction, p_bar, plift_bar,
                                    t_k=None, name=None, index=None, in_service=True, type="pt",
                                    **kwargs):
    """
    Adds one circulation pump with a constant pressure lift in table net["circ_pump_pressure"].

    :param net: The net within this pump should be created
    :type net: pandapipesNet
    :param from_junction:
    :type from_junction:
    :param to_junction:
    :type to_junction:
    :param p_bar:
    :type p_bar:
    :param plift_bar:
    :type plift_bar:
    :param t_k:
    :type t_k:
    :param name:
    :type name:
    :param index:
    :type index:
    :param in_service:
    :type in_service:
    :param type:
    :type type:
    :param kwargs:
    :type kwargs:
    :return:
    :rtype:

    EXAMPLE:
        >>> create_circ_pump_const_pressure(net, 0, 1, p_bar=5, plift_bar=2, t_k=350, type="p")

    """

    add_new_component(net, CirculationPumpPressure)

    for b in [from_junction, to_junction]:
        if b not in net["junction"].index.values:
            raise UserWarning(
                    "CirculationPumpPressure %s tries to attach to non-existing junction %s"
                    % (name, b))

    if index is None:
        index = get_free_id(net["circ_pump_pressure"])
    if index in net["circ_pump_pressure"].index:
        raise UserWarning("A CirculationPumpPressure with the id %s already exists" % id)

    # store dtypes
    dtypes = net.circ_pump_pressure.dtypes

    v = {"name": name, "from_junction": from_junction, "to_junction": to_junction,
         "p_bar": p_bar, "t_k": t_k, "plift_bar": plift_bar,
         "in_service": bool(in_service), "type": type}
    v.update(kwargs)
    # and preserve dtypes
    for col, val in v.items():
        net.circ_pump_pressure.at[index, col] = val
    _preserve_dtypes(net.circ_pump_pressure, dtypes)

    return index


def create_circ_pump_const_mass_flow(net, from_junction, to_junction, p_bar, mdot_kg_per_s,
                                     t_k=None, name=None, index=None, in_service=True,
                                     type="pt", **kwargs):
    """
    Adds one circulation pump with a constant mass flow in table net["circ_pump_mass"].

    :param net: The net within this pump should be created
    :type net: pandapipesNet
    :param from_junction:
    :type from_junction:
    :param to_junction:
    :type to_junction:
    :param p_bar:
    :type p_bar:
    :param mdot_kg_per_s:
    :type mdot_kg_per_s:
    :param t_k:
    :type t_k:
    :param name:
    :type name:
    :param index:
    :type index:
    :param in_service:
    :type in_service:
    :param type:
    :type type:
    :param kwargs:
    :type kwargs:
    :return:
    :rtype:

    EXAMPLE:
        >>> create_circ_pump_const_pressure(net, 0, 1, p_bar=5, mdot_kg_per_s=2, t_k=350, type="p")

    """

    add_new_component(net, CirculationPumpMass)

    for b in [from_junction, to_junction]:
        if b not in net["junction"].index.values:
            raise UserWarning("CirculationPumpMass %s tries to attach to non-existing junction %s"
                              % (name, b))

    if index is None:
        index = get_free_id(net["circ_pump_mass"])
    if index in net["circ_pump_mass"].index:
        raise UserWarning("A CirculationPumpMass with the id %s already exists" % id)

    # store dtypes
    dtypes = net.circ_pump_mass.dtypes

    v = {"name": name, "from_junction": from_junction, "to_junction": to_junction,
         "p_bar": p_bar, "t_k": t_k, "mdot_kg_per_s": mdot_kg_per_s,
         "in_service": bool(in_service), "type": type}
    v.update(kwargs)
    # and preserve dtypes
    for col, val in v.items():
        net.circ_pump_mass.at[index, col] = val
    _preserve_dtypes(net.circ_pump_mass, dtypes)

    return index


def create_fluid_from_lib(net, name, overwrite=True):
    """
    Creates a fluid from library (if there is an entry) and sets net["fluid"] to this value.
    Currently existing fluids in the library are: "water", "natural_gas", "air".

    :param net: The net for which this fluid should be created
    :type net: pandapipesNet
    :param name: The name of the fluid that shall be extracted from the fluid lib
    :type name: str
    :param overwrite: Flag if a possibly existing fluid in the net shall be overwritten.
    :type overwrite: bool, default True
    :return: No output.

    EXAMPLE:
        pp.create_fluid_from_lib(net, name="water")

    """
    add_fluid_to_net(net, call_lib(name), overwrite=overwrite)
