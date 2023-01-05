# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
from pandapipes.component_models.junction_component import Junction
from pandapipes.component_models.abstract_models import ConstFlow
from numpy import dtype


class MassStorage(ConstFlow):
    """

    """

    @classmethod
    def table_name(cls):
        return "mass_storage"

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
                ("init_m_stored_kg", "f8"),
                ("min_m_stored_kg", "f8"),
                ("max_m_stored_kg", "f8"),
              # ("m_stored_kg", "f8"),  # not in the DF by default, should be created by controller
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

    @classmethod
    def get_connected_node_type(cls):
        return Junction
