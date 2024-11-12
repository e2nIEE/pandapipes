# Copyright (c) 2020-2024 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import os


class ComponentRegistry(object):

    registry = dict()

    def __init__(self, comp_list):
        for comp in comp_list:
            comp.__iscustom__ = False
            comp_instance = comp()
            self.registry[comp_instance.table_name] = comp_instance
            self.registry[comp] = comp_instance

    @classmethod
    def register(cls):
        def wrapper(wrapped_class):
            wrapped_class.__iscustom__ = True
            wrapped_class.__path_from_home__ = os.path.relpath(__file__, os.path.expanduser("~"))
            wrapped_instance = wrapped_class()
            cls.registry[wrapped_instance.table_name] = wrapped_instance
            cls.registry[wrapped_class] = wrapped_instance
            return wrapped_class
        return wrapper

    @classmethod
    def get(cls, comp):
        return cls.registry.get(comp, None)

def register_component(cls):
    return ComponentRegistry.register()(cls)
