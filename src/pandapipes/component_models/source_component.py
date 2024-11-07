# Copyright (c) 2020-2024 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from pandapipes.component_models.abstract_models.const_flow_models import ConstFlow


class Source(ConstFlow):
    """

    """
    @property
    def table_name(self):
        return "source"

    @property
    def sign(self):
        return -1
