# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from pandapower.create import _get_index_with_check, load_std_type, _set_entries

from pandapipes.component_models.component_toolbox import add_new_component
from pandapipes.converter.stanet.valve_pipe_component.valve_pipe_component import ValvePipe
from pandapipes.create import _check_branch, _check_std_type
from pandapipes.pandapipes_net import pandapipesNet


def create_valve_pipe(net, from_junction, to_junction, std_type, length_km, k_mm=0.15e-3,
                      opened=True, loss_coefficient=0, sections=1, alpha_w_per_m2k=0., text_k=293, qext_w=0.,
                      name=None, index=None, geodata=None, in_service=True, type="valve_pipe", **kwargs):
    """
    Creates a valve pipe element in net["valve_pipe"] from valve pipe parameters. In any case the
    line parameters are defined through a single standard type. This component is
    an equivalent to STANET's valve element, as it represents a valve with a length, unlike the
    normal pandapipes valve which is assumed to be of length 0. This component is not supposed to
    be added to the standard pandapipes components, as it doesn't add value to the model itself, but
    is just an alternative to how valves can be modeled. Its equivalent of one valve connected to
    one or two pipes is a cleaner model from our point of view.

    :param net: The net within this pipe should be created
    :type net: pandapipesNet
    :param from_junction: ID of the junction on one side which the pipe will be connected with
    :type from_junction: int
    :param to_junction: ID of the junction on the other side which the pipe will be connected with
    :type to_junction: int
    :param std_type: Name of standard type
    :type std_type: str
    :param length_km: The pipe length in km
    :type length_km: float
    :param k_mm: Pipe roughness in [mm]
    :type k_mm: float, default 0.15e-3
    :param opened: Flag to show if the valve is opened and allows for fluid flow or if it is closed\
            to block the fluid flow.
    :type opened: bool, default True
    :param loss_coefficient: The pressure loss coefficient introduced by the valve shape
    :type loss_coefficient: float, default 0
    :param sections: The number of internal pipe sections. Important for gas and temperature\
            calculations, where variables are dependent on pipe length.
    :type sections: int, default 1
    :param alpha_w_per_m2k: Heat transfer coefficient in [W/(m^2*K)]
    :type alpha_w_per_m2k: float, default 0
    :param text_k: Ambient temperatures of pipes in [K]
    :type text_k: Iterable or float, default 293
    :param qext_w: external heat feed-in to the pipe in [W]
    :type qext_w: float, default 0.0
    :param name: A name tag for this valve pipe
    :type name: str, default None
    :param index: Force a specified ID if it is available. If None, the index one higher than the
                    highest already existing index is selected.
    :type index: int, default None
    :param geodata: The coordinates of the pipe. The first row should be the coordinates of\
        junction a and the last should be the coordinates of junction b. The points in the\
        middle represent the bending points of the pipe
    :type geodata: array, shape= (,2L), default None
    :param in_service: True for in_service or False for out of service
    :type in_service: bool, default True
    :param type: An identifier for special types of valves
    :type type: str, default None
    :param kwargs: Additional keyword arguments will be added as further columns to the\
                    net["source"] table
    :return: index - The unique ID of the created valve pipe
    :rtype: int

    EXAMPLE:
        create_valve_pipe(net, "valve_pipe1", from_junction=0, to_junction=1, std_type='315_PE_80_SDR_17',
                          length_km=1)
    """

    # check if junction exist to attach the pipe to

    add_new_component(net, ValvePipe)

    index = _get_index_with_check(net, "valve_pipe", index)
    _check_branch(net, "ValvePipe", index, from_junction, to_junction)
    _check_std_type(net, std_type, "pipe", "create_pipe")

    pipe_parameter = load_std_type(net, std_type, "pipe")
    v = {"name": name, "from_junction": from_junction, "to_junction": to_junction,
         "std_type": std_type, "length_km": length_km, "diameter_m": pipe_parameter["inner_diameter_mm"] / 1000,
         "k_mm": k_mm, "opened": opened, "loss_coefficient": loss_coefficient,
         "alpha_w_per_m2k": alpha_w_per_m2k, "sections": sections, "in_service": bool(in_service), "type": type,
         "qext_w": qext_w, "text_k": text_k}

    _set_entries(net, "valve_pipe", index, **v, **kwargs)

    if geodata is not None:
        net["pipe_geodata"].at[index, "coords"] = geodata

    return index


def create_valve_pipe_from_parameters(net, from_junction, to_junction, length_km, diameter_m, k_mm=0.15e-3,
                                      opened=True, loss_coefficient=0, sections=1, alpha_w_per_m2k=0., text_k=293,
                                      qext_w=0., name=None, index=None, geodata=None, in_service=True,
                                      type="valve_pipe", **kwargs):
    """
    Creates a valve pipe element in net["valve_pipe"] from valve pipe parameters. This component is
    an equivalent to STANET's valve element, as it represents a valve with a length, unlike the
    normal pandapipes valve which is assumed to be of length 0. This component is not supposed to
    be added to the standard pandapipes components, as it doesn't add value to the model itself, but
    is just an alternative to how valves can be modeled. Its equivalent of one valve connected to
    one or two pipes is a cleaner model from our point of view.

    :param net: The net within this pipe should be created
    :type net: pandapipesNet
    :param from_junction: ID of the junction on one side which the pipe will be connected with
    :type from_junction: int
    :param to_junction: ID of the junction on the other side which the pipe will be connected with
    :type to_junction: int
    :param length_km: The pipe length in km
    :type length_km: float
    :param diameter_m: The pipe diameter in m, for a start only circular pipes are supported
    :type diameter_m: float
    :param k_mm: Pipe roughness in [mm]
    :type k_mm: float, default 0.15e-3
    :param opened: Flag to show if the valve is opened and allows for fluid flow or if it is closed\
            to block the fluid flow.
    :type opened: bool, default True
    :param loss_coefficient: The pressure loss coefficient introduced by the valve shape
    :type loss_coefficient: float, default 0
    :param sections: The number of internal pipe sections. Important for gas and temperature\
            calculations, where variables are dependent on pipe length.
    :type sections: int, default 1
    :param alpha_w_per_m2k: Heat transfer coefficient in [W/(m^2*K)]
    :type alpha_w_per_m2k: float, default 0
    :param text_k: Ambient temperatures of pipes in [K]
    :type text_k: Iterable or float, default 293
    :param qext_w: external heat feed-in to the pipe in [W]
    :type qext_w: float, default 0.0
    :param name: A name tag for this valve pipe
    :type name: str, default None
    :param index: Force a specified ID if it is available. If None, the index one higher than the
                    highest already existing index is selected.
    :type index: int, default None
    :param geodata: The coordinates of the pipe. The first row should be the coordinates of\
        junction a and the last should be the coordinates of junction b. The points in the\
        middle represent the bending points of the pipe
    :type geodata: array, shape= (,2L), default None
    :param in_service: True for in_service or False for out of service
    :type in_service: bool, default True
    :param type: An identifier for special types of valves
    :type type: str, default None
    :param kwargs: Additional keyword arguments will be added as further columns to the\
                    net["source"] table
    :return: index - The unique ID of the created valve pipe
    :rtype: int

    EXAMPLE:
        create_valve_pipe_from_parameters(net, "valve_pipe1", from_junction=0, to_junction=1,
                                          length_km=1, d=4e-3)
    """

    # check if junction exist to attach the pipe to

    add_new_component(net, ValvePipe)

    index = _get_index_with_check(net, "valve_pipe", index)
    _check_branch(net, "ValvePipe", index, from_junction, to_junction)

    v = {"name": name, "from_junction": from_junction, "to_junction": to_junction,
         "length_km": length_km, "diameter_m": diameter_m, "k_mm": k_mm,
         "opened": opened, "loss_coefficient": loss_coefficient,
         "alpha_w_per_m2k": alpha_w_per_m2k, "sections": sections, "in_service": bool(in_service), "type": type,
         "qext_w": qext_w, "text_k": text_k}

    _set_entries(net, "valve_pipe", index, **v, **kwargs)

    if geodata is not None:
        net["valve_pipe_geodata"].at[index, "coords"] = geodata

    return index
