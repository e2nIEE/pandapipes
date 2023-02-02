import pandapipes as ps
import pandapower as pp
import numpy as np
from matplotlib import pyplot as plt
from pandapower.plotting import draw_collections
try:
    from workshop.pp.plotting import create_power_net_collections
    from workshop.pps.plotting import create_gas_net_collections2
except ModuleNotFoundError:
    from pp.plotting import create_power_net_collections
    from pps.plotting import create_gas_net_collections2


from pandapipes.multinet.control.controller.multinet_control import P2GControlMultiEnergy
from pandapipes.multinet.create_multinet import create_empty_multinet, add_net_to_multinet
from pandapipes.multinet.control.run_control_multinet import run_control

from pandapower.plotting import create_annotation_collection


def couple_two_nets(enet, gnet, offset=0):
    mn = create_empty_multinet('coupled_nets')
    add_net_to_multinet(mn, enet, 'power')
    add_net_to_multinet(mn, gnet, 'gas')

    p2g_bus = 30
    p2g_junction = 30

    p2g_load = pp.create_load(enet, bus=p2g_bus, p_mw=0.2, name='P2G unit')
    p2g_source = ps.create_source(gnet, junction=p2g_junction, mdot_kg_per_s=0.001, name='P2G unit')

    P2GControlMultiEnergy(mn, p2g_load, p2g_source, efficiency=0.7,
                          name_power_net='power', name_gas_net='gas')

    pp.runpp(enet)
    ps.pipeflow(gnet, iter=100)
    run_control(mn, iter=100)

    # %% combined plots:
    gnet.junction_geodata.x -= offset
    ps_coll = create_gas_net_collections2(gnet)
    pp_coll = create_power_net_collections(enet)
    annotations = create_annotations(enet, gnet)
    draw_collections(ps_coll + pp_coll + annotations)

    plt.show()
    return mn


def create_annotations(enet, gnet):
    coords_bus = enet.bus_geodata.loc[30, ['x', 'y']].values
    coords_junct = gnet.junction_geodata.loc[30, ['x', 'y']].values
    coords = np.concatenate([[coords_bus], [coords_junct]])

    jic = create_annotation_collection(size=30,
                                       texts=['electrolyser'] * len(coords),
                                       coords=coords, zorder=150, color='k')

    # tuples of all junction coords
    return [jic]


if __name__ == '__main__':
    enet = pp.from_json('workshop/pp/net.json')
    gnet = ps.from_json('workshop/pps/net.json')
    mn = couple_two_nets(enet, gnet, offset=1000)
    ps.to_json(mn, 'workshop/multinet.json')
