# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import copy

import pandas as pd
from numpy import dtype
from pandapower import pandapowerNet
from pandapower.auxiliary import ADict

from pandapipes import __version__
from pandapipes import pandapipesNet

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


class MultiNet(ADict):
    """
    A 'MultiNet' is a frame for different pandapipes & pandapower nets and coupling controllers.

    Usually, a multinet is a multi energy net which one net per energy carrier. 
    The coupled simulation can be run with
    pandapipes.multinet.control.run_control_multinet.run_control()
    The nets are stored with a unique key in a dictionary in multinet['nets'].
    Controllers that connect to nets are stored in multinet['controller'].
    """

    def __init__(self, *args, **kwargs):
        """

        :param args: item of the ADict
        :type args: variable
        :param kwargs: item of the ADict with corresponding name
        :type kwargs: dict
        """
        super().__init__(*args, **kwargs)
        if isinstance(args[0], self.__class__):
            net = args[0]
            self.clear()
            self.update(**net.deepcopy())

    def deepcopy(self):
        return copy.deepcopy(self)

    def __repr__(self):  # pragma: no cover
        """
        defines the representation of the multinet in the console

        :return: representation
        :rtype: str
        """

        r = "This multi net includes following nets:"
        for cat in self.nets:
            if isinstance(self['nets'][cat], pandapowerNet):
                r += "\n   - %s (%s pandapowerNet)" % (cat, 1)
            elif isinstance(self['nets'][cat], pandapipesNet):
                r += "\n   - %s (%s pandapipesNet)" % (cat, 1)
            else:
                r += "\n   - %s (%s nets)" % (cat, len(self['nets'][cat]))

        par = []
        for tb in list(self.keys()):
            if isinstance(self[tb], pd.DataFrame) and len(self[tb]) > 0:
                par.append(tb)
            elif tb == 'std_type':
                par.append(tb)
        if par:
            r += "\nand the following parameter tables:"
            for tb in par:
                r += "\n   - %s (%s elements)" % (tb, len(self[tb]))
        return r


def get_default_multinet_structure():
    """
    Return the default structure of an empty multinet with categories and data types.

    :return: default structure of an empty multinet
    :rtype: dict
    """
    default_multinet_structure = {
        # structure data
        # f8, u4 etc. are probably referencing numba or numpy data types
        "name": "",
        "nets": dict(),
        "version": __version__,
        "controller": [('object', dtype(object)),
                       ('in_service', "bool"),
                       ('order', "float64"),
                       ('level', dtype(object)),
                       ('initial_run', "bool"),
                       ("recycle", "bool")]}
    return default_multinet_structure
