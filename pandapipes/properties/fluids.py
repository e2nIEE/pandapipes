# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import os
import pandas as pd
import numpy as np
from pandapipes import pp_dir
from pandapower.io_utils import JSONSerializableClass
from scipy.interpolate import interp1d

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


class Fluid(JSONSerializableClass):
    """

    """

    def __init__(self, name, fluid_type, **kwargs):
        """

        :param name:
        :type name:
        :param fluid_type:
        :type fluid_type:
        :param kwargs:
        :type kwargs:
        """
        super(Fluid, self).__init__()
        self.name = name
        if not isinstance(fluid_type, str) or fluid_type.lower() not in ["gas", "liquid"]:
            logger.warning("The fluid %s has the fluid type %s which might cause problems in the "
                           "pipeflow calculation, as it expects either 'gas' or 'liquid'."
                           % (name, fluid_type))
        self.fluid_type = fluid_type.lower()
        self.is_gas = self.fluid_type == "gas"
        self.all_properties = kwargs
        for prop_name, prop in self.all_properties.items():
            if not isinstance(prop, FluidProperty):
                logger.warning("The property %s was not defined as a fluid property. This might "
                               "cause problems when trying to ask for values." % prop_name)

    def __repr__(self):
        r = "Fluid %s (%s) with properties:" % (self.name, self.fluid_type)
        for key in self.all_properties.keys():
            r += "\n   - %s (%s)" %(key, self.all_properties[key].__class__.__name__[13:])
        return r

    def add_property(self, property_name, prop, overwrite=True, warn_on_duplicates=True):
        """
        This function adds a new property.

        :param property_name: Name of the new property
        :type property_name: str
        :param prop: Values for the property, for example a curve or just a constant value
        :type prop: pandapipes.FluidProperty
        :param overwrite: True if existing property with the same name shall be overwritten
        :type overwrite: bool
        :param warn_on_duplicates: True, if a warning of properties with the same name should be
                                    returned
        :type warn_on_duplicates: bool

        :Example:
            >>> fluid.add_property('water_density', pandapipes.FluidPropertyConstant(998.2061), overwrite=True, warn_on_duplicates=False)

        """
        if property_name in self.all_properties:
            if warn_on_duplicates:
                ow_string = "It will be overwritten." if overwrite else "It will not be replaced."
                logger.warning("The property %s already exists. %s" % (property_name, ow_string))
            if not overwrite:
                return
        self.all_properties[property_name] = prop

    def get_property(self, property_name, *at_values):
        """
        This function returns the value of the requested property.

        :param property_name: Name of the searched property
        :type property_name: str
        :param at_values: Value for which the property should be returned
        :type at_values:
        :return: Returns property at the certain value
        :rtype: pandapipes.FluidProperty
        """

        if property_name not in self.all_properties:
            raise UserWarning("The property %s was not defined for the fluid %s"
                              % (property_name, self.name))
        return self.all_properties[property_name].get_property(*at_values)

    def get_density(self, temperature):
        """
        This function returns the density at a certain temperature.

        :param temperature: Temperature at which the density is queried
        :type temperature: float
        :return: Density at the required temperature

        """

        return self.get_property("density", temperature)

    def get_viscosity(self, temperature):
        """
        This function returns the viscosity at a certain temperature.

        :param temperature: Temperature at which the viscosity is queried
        :type temperature: float
        :return: Viscosity at the required temperature

        """

        return self.get_property("viscosity", temperature)

    def get_heat_capacity(self, temperature):
        """
        This function returns the heat capacity at a certain temperature.

        :param temperature: Temperature at which the heat capacity is queried
        :type temperature: float
        :return: Heat capacity at the required temperature

        """

        return self.get_property("heat_capacity", temperature)

    def get_molar_mass(self):
        """
        This function returns the molar mass.

        :return: molar mass

        """

        return self.get_property("molar_mass")

    def get_compressibility(self, temperature):
        """
        This function returns the compressibility at a certain temperature.

        :param temperature: Temperature at which the compressibility is queried
        :type temperature: float
        :return: compressibility at the required temperature

        """

        return self.get_property("compressibility", temperature)

    def get_der_compressibility(self):
        """
        This function returns the derivative of the compressibility.

        :return: derivative of the compressibility

        """

        return self.get_property("der_compressibility")


class FluidProperty(JSONSerializableClass):
    """
    Property Base Class
    """

    def __init__(self):
        super().__init__()

    def get_property(self, *args):
        """

        :param arg:
        :type arg:
        :return:
        :rtype:
        """
        raise NotImplementedError("Please implement a proper fluid property!")


class FluidPropertyInterExtra(FluidProperty):
    """
    Creates Property with interpolated or extrapolated values.
    """
    json_excludes = JSONSerializableClass.json_excludes + ["prop_getter"]
    prop_getter_entries = {"x": "x", "y": "y", "_fill_value_orig": "fill_value"}

    def __init__(self, x_values, y_values, method="interpolate_extrapolate"):
        """

        :param x_values:
        :type x_values:
        :param y_values:
        :type y_values:
        :param method:
        :type method:
        """
        super(FluidPropertyInterExtra, self).__init__()
        if method.lower() == "interpolate_extrapolate":
            self.prop_getter = interp1d(x_values, y_values, fill_value="extrapolate")
        else:
            self.prop_getter = interp1d(x_values, y_values)

    def get_property(self, arg):
        """

        :param arg: Name of the property and one or more values (x-values) for which the y-values of the property are to be displayed
        :type arg: str, float or array
        :return: y-value/s
        :rtype: float, array
        """
        return self.prop_getter(arg)

    @classmethod
    def from_path(cls, path, method="interpolate_extrapolate"):
        """
        Reads a text file with temperature values in the first column and property values in
        second column.

        :param path: Target path of the txt file
        :type path: str
        :param method: Method with which the values are to be interpolated
        :type method: str
        :return: interpolated values
        :rtype: pandapipes.FluidProperty
        """
        values = np.loadtxt(path)
        return cls(values[:, 0], values[:, 1], method=method)

    def to_dict(self):
        d = super(FluidPropertyInterExtra, self).to_dict()
        d.update({k: self.prop_getter.__dict__[k] for k in self.prop_getter_entries.keys()})
        # d.update({"x_values": self.prop_getter.x, "y_values": self.prop_getter.y,
        #           "method": "interpolate_extrapolate"
        #           if self.prop_getter.fill_value == "extrapolate" else None})
        return d

    @classmethod
    def from_dict(cls, d, net):
        obj = JSONSerializableClass.__new__(cls)
        d2 = {cls.prop_getter_entries[k]: v for k, v in d.items()
              if k in cls.prop_getter_entries.keys()}
        d3 = {k: v for k, v in d.items() if k not in cls.prop_getter_entries.keys()}
        d3["prop_getter"] = interp1d(**d2)
        obj.__dict__.update(d3)
        return obj


class FluidPropertyConstant(FluidProperty):
    """
    Creates Property with a constant value.
    """

    def __init__(self, value):
        """

        :param value:
        :type value:
        """
        super(FluidPropertyConstant, self).__init__()
        self.value = value

    def get_property(self, *args):
        """

        :param arg: Name of the property
        :type arg: str
        :return: Value of the property
        :rtype: float

        :Example:
            >>> heat_capacity = get_fluid(net).get_property("heat_capacity")
        """
        if len(args) > 1:
            raise(UserWarning('Please define either none or an array-like argument'))
        elif len(args) == 1:
            logger.warning('One constant property has several input variables even though it is '
                           'independent of these')
            output = np.array([self.value]) * np.ones(len(args[0]))
        else:
            output = np.array([self.value])
        return  output

    @classmethod
    def from_path(cls, path):
        """
        Reads a text file with temperature values in the first column and property values in
        second column.

        :param path:
        :type path:
        :param method:
        :type method:
        :return:
        :rtype:
        """
        value = np.loadtxt(path).item()
        return cls(value)


class FluidPropertyLinear(FluidProperty):
    """
    Creates Property with a linear course.
    """

    def __init__(self, slope, offset):
        """

        :param slope:
        :type slope:
        :param offset:
        :type offset:

        """
        super(FluidPropertyLinear, self).__init__()
        self.slope = slope
        self.offset = offset

    def get_property(self, arg):
        """

        :param arg: Name of the property and one or more values (x-values) for which the function of the property should be calculated
        :type arg: str, float or array
        :return: y-value or function values
        :rtype: float, array

        :Example:
            >>> comp_fact = get_fluid(net).get_property("compressibility", p_bar)

        """
        if type(arg) == pd.Series:
            return self.offset + self.slope * arg.values
        else:
            return self.offset + self.slope * np.array(arg)

    @classmethod
    def from_path(cls, path):
        """
        Reads a text file with temperature values in the first column and property values in
        second column.

        :param path:
        :type path:
        :param method:
        :type method:
        :return:
        :rtype:
        """
        slope, offset = np.loadtxt(path)
        return cls(slope, offset)


def create_constant_property(net, property_name, value, overwrite=True, warn_on_duplicates=True):
    """
    Creates a property with a constant value.

    :param net: Name of the network to which the property is added
    :type net: pandapipesNet
    :param property_name: Name of the new property
    :type property_name: str
    :param value: Constant value of the property
    :type value: float
    :param overwrite:  True if existing property with the same name shall be overwritten
    :type overwrite: basestring
    :param warn_on_duplicates: True, if a warning of properties with the same name should be
                                returned
    :type warn_on_duplicates: basestring
    """
    prop = FluidPropertyConstant(value)
    get_fluid(net).add_property(property_name, prop, overwrite=overwrite,
                                warn_on_duplicates=warn_on_duplicates)
    return prop


def create_linear_property(net, property_name, slope, offset, overwrite=True,
                           warn_on_duplicates=True):
    """
    Creates a property with a linear correlation.

    :param net: Name of the network to which the property is added
    :type net: pandapipesNet
    :param property_name: Name of the new property
    :type property_name: str
    :param slope: Slope of the linear correlation
    :type slope: float
    :param offset: Offset of the linear function
    :type offset: float
    :param overwrite:  True if existing property with the same name shall be overwritten
    :type overwrite: basestring
    :param warn_on_duplicates: True, if a warning of properties with the same name should be
                                returned
    :type warn_on_duplicates: basestring
    """
    prop = FluidPropertyLinear(slope, offset)
    get_fluid(net).add_property(property_name, prop, overwrite=overwrite,
                                warn_on_duplicates=warn_on_duplicates)
    return prop


def create_constant_fluid(name=None, fluid_type=None, **kwargs):
    """
    Creates a constant fluid.

    :param name: Name of the fluid
    :type name: str
    :param fluid_type: Type of the fluid
    :type fluid_type: str
    :param kwargs: Additional information
    :return: Fluid
    :rtype: Fluid
    """
    properties = dict()
    for prop_name, prop in kwargs.items():
        properties[str(prop_name)] = FluidPropertyConstant(prop)

    return Fluid(name=name, fluid_type=fluid_type, **properties)


def call_lib(fluid):
    """
    Creates a fluid with default fluid properties.

    :param fluid: Fluid which should be used
    :type fluid: str
    :return: Fluid - Chosen fluid with default fluid properties
    :rtype: Fluid
    """

    def interextra_property(prop):
        return FluidPropertyInterExtra.from_path(
            os.path.join(pp_dir, "properties", fluid, prop + ".txt"))

    def constant_property(prop):
        return FluidPropertyConstant.from_path(
            os.path.join(pp_dir, "properties", fluid, prop + ".txt"))

    def linear_property(prop):
        return FluidPropertyLinear.from_path(
            os.path.join(pp_dir, "properties", fluid, prop + ".txt"))

    liquids = ["water"]
    gases = ["air", "lgas", "hgas", "hydrogen"]

    if fluid == "natural_gas":
        logger.error("'natural_gas' is ambigious. Please choose 'hgas' or 'lgas' "
                     "(high- or low calorific natural gas)")
    if fluid not in liquids and fluid not in gases:
        raise AttributeError("Fluid '%s' not found in the fluid library. It might not be "
                             "implemented yet." % fluid)

    density = interextra_property("density")
    viscosity = interextra_property("viscosity")
    heat_capacity = interextra_property("heat_capacity")
    molar_mass = constant_property("molar_mass")
    der_compr = constant_property("der_compressibility")
    compr = linear_property("compressibility")

    phase = "liquid" if fluid in liquids else "gas"
    return Fluid(fluid, phase, density=density, viscosity=viscosity, heat_capacity=heat_capacity,
                 molar_mass=molar_mass,
                 compressibility=compr, der_compressibility=der_compr)


def get_fluid(net):
    """
    This function shows which fluid is used in the net.

    :param net: Current network
    :type net: pandapipesNet
    :return: Fluid - Name of the fluid which is used in the current network
    :rtype: Fluid
    """
    if "fluid" not in net or net["fluid"] is None:
        raise UserWarning("There is no fluid defined for the given net!")
    fluid = net["fluid"]
    if not isinstance(fluid, Fluid):
        logger.warning("The fluid in this net is not of the pandapipes Fluid type. This could lead"
                       " to errors, as some components might depend on this structure")
    return fluid


def _add_fluid_to_net(net, fluid, overwrite=True):
    """
    Adds a fluid to a net. If overwrite is False, a warning is printed and the fluid is not set.

    :param net: The pandapipes network for which to set fluid
    :type net: pandapipesNet
    :param fluid: fluid which to insert into the network
    :type fluid: Fluid
    :param overwrite: If True, an existing fluid will just be overwritten, otherwise a warning is\
        printed out and the fluid is not reset.
    :type overwrite: bool, default True
    :return: Not output.
    """
    if "fluid" in net and net["fluid"] is not None and not overwrite:
        fluid_msg = "an existing fluid" if not hasattr(net["fluid"], "name") \
            else "the fluid %s" % net["fluid"].name
        logger.warning("The fluid %s would replace %s and thus cannot be created. Try to set "
                       "overwrite to True" % (fluid.name, fluid_msg))
        return

    if isinstance(fluid, str):
        logger.warning("Instead of a pandapipes.Fluid, a string ('%s') was passed to the fluid "
                       "argument. Internally, it will be passed to call_lib(fluid) to get the "
                       "respective pandapipes.Fluid." %fluid)
        fluid = call_lib(fluid)
    net["fluid"] = fluid
