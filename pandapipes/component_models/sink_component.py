# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from pandapipes.component_models.abstract_models import ConstFlow


class Sink(ConstFlow):
    """

    """

    @classmethod
    def table_name(cls):
        return "sink"

    @classmethod
    def sign(cls):
        return 1
