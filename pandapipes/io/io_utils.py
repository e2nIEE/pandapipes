# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import importlib
from inspect import isclass

from pandapipes.component_models.abstract_models import Component
from pandapipes.create import create_empty_network
from pandapipes.pandapipes_net import pandapipesNet
from pandapower.io_utils import get_obj_idx_addr_iterator, get_weakref_idx_addr_iterator, \
    address_string
from pandapower.io_utils import with_signature, to_serializable, \
    isinstance_partial as ppow_isinstance, FromSerializableRegistry

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)

try:
    import fiona
    import geopandas

    GEOPANDAS_INSTALLED = True
except ImportError:
    GEOPANDAS_INSTALLED = False

try:
    import shapely.geometry

    SHAPELY_INSTALLED = True
except (ImportError, OSError):
    SHAPELY_INSTALLED = False


def isinstance_partial(obj, cls):
    if isinstance(obj, pandapipesNet):
        return False
    return ppow_isinstance(obj, cls)


class FromSerializableRegistryPpipe(FromSerializableRegistry):
    from_serializable = FromSerializableRegistry.from_serializable
    class_name = ''
    module_name = ''

    def __init__(self, obj, d, obj_hook, memo_pp, addresses_to_fill, weakrefs_to_fill):
        """
        Class containing all registered function for deserialization.

        :param obj: object to be deserialized (usually string)
        :type obj: str
        :param d: dictionary with additional information
        :type d: dict
        :param obj_hook: the object hook to be handed to underlying decoders
        :type obj_hook: function
        :param memo_pp: object memory dict to be filled with the object if necessary
        :type memo_pp: dict
        :param addresses_to_fill: dict mapping addresses to setter functions where to insert the\
                                  objects restored for these addresses (saved in memo_pp)
        :type addresses_to_fill: defaultdict
        :param weakrefs_to_fill: dict mapping addresses to setter functions where to insert the\
                                 weakrefs of objects restored for these addresses (saved in memo_pp)
        :type weakrefs_to_fill: defaultdict
        """
        super().__init__(obj, d, obj_hook, memo_pp, addresses_to_fill, weakrefs_to_fill)

    @from_serializable.register(class_name="method")
    def method(self):
        module = importlib.import_module(self.module_name)
        logger.warning('Deserializing of method not tested. This might fail...')
        func = getattr(module, self.obj)
        # class_ = getattr(module, obj) # doesn't work
        return func

    @from_serializable.register(class_name='pandapipesNet', module_name='pandapipes.pandapipes_net')
    def pandapipes_net(self):
        if isinstance(self.obj, str):  # backwards compatibility
            from pandapipes import from_json_string
            net = from_json_string(self.obj)
        else:
            net = create_empty_network()
            net.update(self.obj)
        self.underlying_objects = get_obj_idx_addr_iterator(net.items())
        self.weakrefs = get_weakref_idx_addr_iterator(net.items())
        return net

    @from_serializable.register()
    def rest(self):
        module = importlib.import_module(self.module_name)
        class_ = getattr(module, self.class_name)
        if isclass(class_) and issubclass(class_, Component):
            return class_
        return super(FromSerializableRegistryPpipe, self).extract_object(class_)


@to_serializable.register(pandapipesNet)
def json_pandapipes_net(obj, memo=None):
    if memo is not None:
        if obj in memo:
            return address_string(obj)
        memo.append(obj)
    net_dict = {k: item for k, item in obj.items() if not k.startswith("_")}
    d = with_signature(obj, net_dict,with_address=True)
    return d


@to_serializable.register(type)
def json_component(class_, memo=None):
    if issubclass(class_, Component):
        d = with_signature(class_(), str(class_().__dict__))
        return d
    else:
        raise (UserWarning('with_signature needs to be defined for '
                           'class %s in @to_serializable.register(type)!' % class_))


if __name__ == '__main__':
    ntw = create_fluid_network()
    import pandapipes as pp

    pp.to_json(ntw, 'test.json')
    ntw = pp.from_json('test.json')
