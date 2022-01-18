# Copyright (c) 2020-2021 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from pandapipes.component_models.abstract_models import ConstFlow
from numpy import dtype


class Source(ConstFlow):
    """

    """

    @classmethod
    def table_name(cls):
        return "source"

    @classmethod
    def sign(cls):
        return -1

    @classmethod
    def get_component_input(cls):
        """

        :return:
        :rtype:
        """
        return [("name", dtype(object)),
                ("junction", "u4"),
                ("mdot_kg_per_s", "f8"),
                ("fluid", dtype(object)),
                ("scaling", "f8"),
                ("in_service", "bool"),
                ("type", dtype(object))]