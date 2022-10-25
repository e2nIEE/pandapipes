from pandapipes.component_models.component_toolbox import add_new_component
from pandapipes.pandapipes_net import pandapipesNet
from pandapower.auxiliary import get_free_id, _preserve_dtypes
from pandapipes.converter.stanet.valve_pipe_component.valve_pipe_component import ValvePipe


def create_valve_pipe(net, from_junction, to_junction, length_km, diameter_m, opened=True,
                      loss_coefficient=0, sections=1, std_type=None, k_mm=0.15e-3,
                      max_vdot_m3_per_s=100, max_v_m_per_s=None, alpha_w_per_m2k=0., qext_w=0.,
                      name=None, index=None, in_service=True, type="valve_pipe", **kwargs):
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
    :param opened: Flag to show if the valve is opened and allows for fluid flow or if it is closed\
            to block the fluid flow.
    :type opened: bool, default True
    :param loss_coefficient: The pressure loss coefficient introduced by the valve shape
    :type loss_coefficient: float, default 0
    :param sections: The number of internal pipe sections. Important for gas and temperature\
            calculations, where variables are dependent on pipe length.
    :type sections: int, default 1
    :param std_type:
    :type std_type: str, default None
    :param k_mm:
    :type k_mm: float, default 0.15e-3
    :param max_vdot_m3_per_s:
    :type max_vdot_m3_per_s: float, default 100
    :param max_v_m_per_s:
    :type max_v_m_per_s: ?, default None
    :param alpha_w_per_m2k: Heat transfer coefficient in [W/(m^2*K)]
    :type alpha_w_per_m2k: float, default 0
    :param qext_w: external heat feed-in to the pipe in [W]
    :type qext_w: float, default 0.0
    :param name: A name tag for this valve pipe
    :type name: str, default None
    :param index: Force a specified ID if it is available. If None, the index one higher than the
                    highest already existing index is selected.
    :type index: int, default None
    :param in_service: True for in_service or False for out of service
    :type in_service: bool, default True
    :param type: An identifier for special types of valves
    :type type: str, default None
    :param kwargs: Additional keyword arguments will be added as further columns to the\
                    net["source"] table
    :return: index - The unique ID of the created valve pipe
    :rtype: int

    EXAMPLE:
        create_valve_pipe(net, "valve_pipe1", from_junction = 0, to_junction = 1, length_m=1,
                            d = 4e-3)
    """

    # check if junction exist to attach the pipe to

    add_new_component(net, ValvePipe)

    for b in [from_junction, to_junction]:
        if b not in net["junction"].index.values:
            raise UserWarning("Pipe %s tries to attach to non-existing junction %s"
                              % (name, b))

    if index is None:
        index = get_free_id(net["valve_pipe"])

    if index in net["valve_pipe"].index:
        raise UserWarning("A pipe with index %s already exists" % index)

    v = {"name": name, "from_junction": from_junction, "to_junction": to_junction,
         "length_km": length_km, "diameter_m": diameter_m, "opened": opened,
         "loss_coefficient": loss_coefficient, "sections": sections,
         "std_type": std_type, "k_mm": k_mm, "max_vdot_m3_per_s": max_vdot_m3_per_s,
         "max_v_m_per_s": max_v_m_per_s, "in_service": bool(in_service), "type": type,
         "alpha_w_per_m2k": alpha_w_per_m2k, "qext_w": qext_w}
    v.update(kwargs)
    # store dtypes
    dtypes = net.valve_pipe.dtypes

    for col, val in v.items():
        net.valve_pipe.at[index, col] = val

    # and preserve dtypes
    _preserve_dtypes(net.valve_pipe, dtypes)

    return index
