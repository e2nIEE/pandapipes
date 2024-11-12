# Copyright (c) 2020-2024 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import os


class ComponentRegistry(object):
    """
    Container for all pandapipes and custom components. Internal pandapipes components are added
    during class initialization, while custom components are added during class creation via the
    @register_component method.
    The registry holds all instances of components in a singleton style and is accessible with
    either the table_name (e.g. 'pipe') or the class name (e.g. Pipe, Pipe needs to be imported).
    If other components methods are needed in a component, the 'get' method enables lazy-loading
    of the component instance.
    """

    registry = dict()

    def __init__(self, comp_list):
        for comp in comp_list:
            comp.__iscustom__ = False
            comp_instance = comp()
            self.registry[comp_instance.table_name] = comp_instance
            self.registry[comp] = comp_instance

    @classmethod
    def register(cls):
        """
        Adds custom component to the component registry.
        """
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
        """
        Useful for lazy-loading component instances.
        """
        return cls.registry.get(comp, None)

def register_component(cls):
    """
    Adds a custom component to the component registry.
    """
    return ComponentRegistry.register()(cls)
