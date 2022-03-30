# Copyright (c) 2020-2021 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import copy

import numpy as np
import pandas as pd
from numpy import dtype
from pandapipes import __version__
from pandapipes.component_models import Junction, Pipe, ExtGrid
from pandapipes.component_models.auxiliaries.create_toolbox import add_new_component
from pandapower.auxiliary import ADict

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


class pandapipesNet(ADict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(args[0], self.__class__):
            net = args[0]
            self.clear()
            self.update(**net.deepcopy())

    def deepcopy(self):
        return copy.deepcopy(self)

    def __repr__(self):  # pragma: no cover
        r = "This pandapipes network includes the following parameter tables:"
        par = []
        res = []
        for tb in list(self.keys()):
            if isinstance(self[tb], pd.DataFrame) and len(self[tb]) > 0:
                if 'res_' in tb:
                    res.append(tb)
                else:
                    par.append(tb)
            elif tb == 'std_type':
                par.append(tb)
        for tb in par:
            r += "\n   - %s (%s elements)" % (tb, len(self[tb]))
        if res:
            r += "\nand the following results tables:"
            for tb in res:
                r += "\n   - %s (%s elements)" % (tb, len(self[tb]))
        r += "."
        if "fluid" in self and self["fluid"] and bool(self["fluid"]):
            r += "\nIt contains the following fluids: "
            try:
                for key in self["fluid"].keys():
                     r += "\n%s" % self["fluid"][key]
            except:
                r += "\n%s" % self["fluid"]
        else:
            r += "\nIt does not contain any defined fluid"
        if "component_list" in self:
            r += "\nand uses the following component models:"
            for component in self.component_list:
                r += "\n   - %s" %component.__name__
        elif "node_list" in self:
            r += "\nand uses the following component models:"
            for component in np.concatenate([self.node_list, self.branch_list, self.node_element_list]):
                r += "\n   - %s" %component.__name__
        return r


def get_basic_net_entries():
    return {
        "fluid": {},
        "converged": False,
        "OPF_converged": False,
        "name": "",
        "version": __version__,
        "node_list": [],
        "node_element_list": [],
        "branch_list": []}


def get_basic_components():
    return Junction, ExtGrid, Pipe


def add_default_components(net, overwrite=False):
    for comp in get_basic_components():
        add_new_component(net, comp, overwrite)
    if "controller" not in net or overwrite:
        ctrl_dtypes = [('object', dtype(object)),
                       ('in_service', "bool"),
                       ('order', "float64"),
                       ('level', dtype(object)),
                       ('initial_run', "bool"),
                       ("recycle", "bool")]
        net['controller'] = pd.DataFrame(np.zeros(0, dtype=ctrl_dtypes), index=pd.Index([], dtype=np.int64))
