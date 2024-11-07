# Copyright (c) 2020-2024 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from pandapipes.component_models.abstract_models import ConstFlow
from numpy import dtype


class MassStorage(ConstFlow):
    """

    """
    @property
    def table_name(self):
        return "mass_storage"

    @property
    def sign(self):
        return 1

    @classmethod
    def get_component_input(self):
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
