# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import pandapipes.topology as top
import pandas as pd

def pressure_profile_to_junction_geodata(net, pressures=None):
    if pressures is None:
        if not net.converged:
            raise ValueError("no results in this pandapower network")
        pressures = net.res_junction.p_bar

    dist = top.calc_minimum_distance_to_junctions(net, net.ext_grid.junction.values)

    bgd = pd.DataFrame({"x": dist.loc[net.junction.index.values].values,
                        "y": pressures.loc[net.junction.index.values].values},
                       index=net.junction.index)
    return bgd


if __name__ == '__main__':
    import pandapipes.networks as nw
    import pandapipes as pp

    net = nw.schutterwald()    
    pp.pipeflow(net)

    dist = top.calc_minimum_distance_to_junctions(net, net.ext_grid.junction.values)
    bgd = pressure_profile_to_junction_geodata(net)