# Copyright (c) 2020-2021 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from pandapipes.component_models.abstract_models import ConstFlow
from numpy import dtype


class StorageFluid(ConstFlow):
    """

    """

    @classmethod
    def table_name(cls):
        return "storage_fluid"

    @classmethod
    def sign(cls):
        return 1

    @classmethod
    def get_component_input(cls):
        """

        :return:
        :rtype:
        """
        return [("name", dtype(object)),
                ("junction", "u4"),
                ("mdot_kg_per_s", "f8"),
                ("scaling", "f8"),
                ("min_kg", "f8"),
                ("max_kg", "f8"),
                ("in_service", "bool"),
                ("type", dtype(object))]

    @classmethod
    def get_result_table(cls, net):
        """Get results.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :return: (columns, all_float) - the column names and whether they are all float type. Only
                if False, returns columns as tuples also specifying the dtypes
        :rtype: (list, bool)
        """
        return ["mdot_kg_per_s"], True
