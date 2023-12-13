# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.


import pandapipes.topology as top
import pandas as pd


def pressure_profile_to_junction_geodata(net):
    """
    Calculates pressure profile for a pandapipes network.

     INPUT:
        **net** (pandapipesNet) - Variable that contains a pandapipes network.

     OUTPUT:
        **bgd** - Returns a pandas DataFrame containing distance to the closest ext_grid as x \
                  coordinate and pressure level as y coordinate for each junction.

     EXAMPLE:
        import pandapipes.networks as nw
        import pandapipes.plotting as plotting
        import pandapipes as pp

        net = nw.schutterwald()
        pp.pipeflow(net)
        bgd = plotting.pressure_profile_to_junction_geodata(net)

    """
    if "res_junction" not in net:
        raise ValueError("no results in this pandapipes network")

    dist = top.calc_minimum_distance_to_junctions(net, net.ext_grid.junction.values)
    junctions = net.junction.index.values
    return pd.DataFrame({"x": dist.loc[junctions].values,
                         "y": net.res_junction.p_bar.loc[junctions].values},
                        index=junctions)
