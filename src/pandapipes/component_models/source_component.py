# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
from pandapipes.component_models.junction_component import Junction
from pandapipes.component_models.abstract_models.const_flow_models import ConstFlow


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
    def get_connected_node_type(cls):
        return Junction

    @classmethod
    def active_identifier(cls):
        return "in_service"
