# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import pandapipes as pp
from pandapipes.constants import GRAVITATION_CONSTANT, P_CONVERSION


def transform_water_tower_into_ext_grid(net, junction, height_m, t_k=293.15, name='water_tower', in_service=True,
                                       index=None, type='pt'):
    density = net.fluid.get_density(t_k).item()

    p_bar = density * height_m * GRAVITATION_CONSTANT / P_CONVERSION

    return pp.create_ext_grid(net, junction, p_bar, t_k, name, in_service, index, type)
