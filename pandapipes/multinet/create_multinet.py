# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
import pandas as pd
from pandapipes.multinet.multinet import MultiNet, get_default_multinet_structure

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.WARNING)


def create_empty_multinet(name=""):
    """
    This function initializes the multinet datastructure.

    :param name: Name for the multi net
    :type name: string, default None
    :return: MultiNet with empty tables
    :rtype: MultiNet

    :Example:
        >>> mn = create_empty_multinet("my_first_multinet")

    """
    multinet = MultiNet(get_default_multinet_structure())
    multinet['controller'] = pd.DataFrame(np.zeros(0, dtype=multinet['controller']), index=[])
    multinet['name'] = name
    return multinet


def add_net_to_multinet(multinet, net, net_name='power', overwrite=False):
    """
    Add a pandapipes or pandapower net to the multinet structure.

    :param multinet: multinet to which a pandapipes/pandapower net will be added
    :type multinet: pandapipes.MultiNet
    :param net: pandapipes or pandapower net that will be added to the multinet
    :type net: pandapowerNet or pandapipesNet
    :param net_name: unique name for the added net, e.g. 'power', 'gas', or 'power_net1'
    :type net_name: str
    :param overwrite: whether a net should be overwritten if it has the same net_name
    :type overwrite: bool
    :return: net reference is added inplace to the multinet (in multinet['nets'])
    :rtype: None
    """
    if net_name in multinet['nets'] and not overwrite:
        logger.warning("A net with the name %s exists already in the multinet. If you want to "
                       "overwrite it, set 'overwrite' to True." % net_name)
    else:
        multinet['nets'][net_name] = net


def add_nets_to_multinet(multinet, overwrite=False, **networks):
    """
    Add multiple nets to a multinet. 'networks' is one or more keyword arguments with nets.

    :param multinet: multinet to which several pandapipes/pandapower nets are added
    :type multinet: pandapipes.MultiNet
    :param overwrite: whether a net should be overwritten if it has the same net_name
    :type overwrite: bool
    :param networks: one or more keyword arguments with pandapipes/pandapower nets as values. 
                     The keyword of each net will be set in multinet.nets as the name for the
                     network in the respective argument.
    :type networks: kwarg (name=net)
    :return: nets are added to multinet
    :rtype: None
    """
    for name, net in networks.items():
        add_net_to_multinet(multinet, net, name, overwrite)
