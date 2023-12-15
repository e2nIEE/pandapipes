# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from pandapipes import pp_dir
from pandapower.io_utils import JSONSerializableClass

try:
    import pandaplan.core.pplog as logging
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
        """
        Definition of fluid representation in the console.

        :return: representation of fluid in the console
        :rtype: str
        """

        r = "Fluid %s (%s) with properties:" % (self.name, self.fluid_type)
        for key in self.all_properties.keys():
            r += "\n   - %s (%s)" % (key, self.all_properties[key].__class__.__name__[13:])
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
            >>> fluid.add_property('water_density', pandapipes.FluidPropertyConstant(998.2061),\
                                   overwrite=True, warn_on_duplicates=False)

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
        return self.all_properties[property_name].get_at_value(*at_values)

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

    def get_compressibility(self, p_bar):
        """
        This function returns the compressibility at a certain pressure.

        :param p_bar: pressure at which the compressibility is queried
        :type p_bar: float or array of floats
        :return: compressibility at the required pressure

        """

        return self.get_property("compressibility", p_bar)

    def get_der_compressibility(self):
        """
        This function returns the derivative of the compressibility with respect to pressure.

        :return: derivative of the compressibility

        """

        return self.get_property("der_compressibility")


class FluidProperty(JSONSerializableClass):
    """
    Property Base Class
    """

    def __init__(self):
        """

        """
        super().__init__()

    def get_at_value(self, *args):
        """

        :param args:
        :type args:
        :return:
        :rtype: float, np.array
        """
        raise NotImplementedError("Please implement a proper fluid property!")

    def get_at_integral_value(self, *args):
        """

        :param args:
        :type args:
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

    def get_at_value(self, arg):
        """

        :param arg: Name of the property and one or more values (x-values) for which the y-values \
            of the property are to be displayed
        :type arg: str, float or array
        :return: y-value/s
        :rtype: float, array
        """
        return self.prop_getter(arg)

    def get_at_integral_value(self, upper_limit_arg, lower_limit_arg):
        """

        :param upper_limit_arg: one or more values of upper limit values for which the function \
            of the property should calculate the integral for
        :type upper_limit_arg: float or list-like objects
        :param lower_limit_arg: one or more values of lower limit values for which the function \
            of the property should calculate the integral for
        :type lower_limit_arg: float or list-like objects
        :return: integral between the limits
        :rtype: float, array

        :Example:
            >>> comp_fact = get_fluid(net).all_properties["heat_capacity"].get_at_integral_value(\
                    t_upper_k, t_lower_k)

        """
        mean = (self.prop_getter(upper_limit_arg) + self.prop_getter(upper_limit_arg)) / 2
        return mean * (upper_limit_arg-lower_limit_arg)

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
    def from_dict(cls, d):
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

    def __init__(self, value, warn_dependent_variables=False):
        """

        :param value:
        :type value:
        """
        super(FluidPropertyConstant, self).__init__()
        self.value = value
        self.warn_dependent_variables = warn_dependent_variables

    def get_at_value(self, *args):
        """

        :param args: Name of the property
        :type args: str
        :return: Value of the property
        :rtype: float

        :Example:
            >>> heat_capacity = get_fluid(net).all_properties["heat_capacity"].get_at_value(293.15)
        """
        if len(args) > 1:
            raise UserWarning('Please define either none or an array-like argument')
        elif len(args) == 1:
            if self.warn_dependent_variables:
                logger.warning('Constant property received several input variables, although it is'
                               'independent of these')
            output = np.array([self.value]) * np.ones(len(args[0]))
        else:
            output = np.array([self.value])
        return output

    def get_at_integral_value(self, upper_limit_arg, lower_limit_arg):
        """

        :param upper_limit_arg: one or more values of upper limit values for which the function \
            of the property should calculate the integral for
        :type upper_limit_arg: float or list-like objects
        :param lower_limit_arg: one or more values of lower limit values for which the function \
            of the property should calculate the integral for
        :type lower_limit_arg: float or list-like objects
        :return: integral between the limits
        :rtype: float, array

        :Example:
            >>> comp_fact = get_fluid(net).all_properties["heat_capacity"].get_at_integral_value(\
                    t_upper_k, t_lower_k)

        """
        if isinstance(upper_limit_arg, pd.Series):
            ul = self.value * upper_limit_arg.values
        else:
            ul = self.value * np.array(upper_limit_arg)
        if isinstance(lower_limit_arg, pd.Series):
            ll = self.value * lower_limit_arg.values
        else:
            ll = self.value * np.array(lower_limit_arg)
        return ul - ll

    @classmethod
    def from_path(cls, path):
        """
        Reads a text file with temperature values in the first column and property values in
        second column.

        :param path:
        :type path:
        :return:
        :rtype:
        """
        value = np.loadtxt(path).item()
        return cls(value)

    @classmethod
    def from_dict(cls, d):
        obj = super().from_dict(d)
        if "warn_dependent_variables" not in obj.__dict__.keys():
            obj.__dict__["warn_dependent_variables"] = False
        return obj


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

    def get_at_value(self, arg):
        """

        :param arg: Name of the property and one or more values (x-values) for which the function \
            of the property should be calculated
        :type arg: str, float or array
        :return: y-value or function values
        :rtype: float, array

        :Example:
            >>> comp_fact = get_fluid(net).all_properties["compressibility"].get_at_value(p_bar)

        """
        if isinstance(arg, pd.Series):
            return self.offset + self.slope * arg.values
        else:
            return self.offset + self.slope * np.array(arg)

    def get_at_integral_value(self, upper_limit_arg, lower_limit_arg):
        """

        :param upper_limit_arg: one or more values of upper limit values for which the function \
            of the property should calculate the integral for
        :type upper_limit_arg: float or list-like objects
        :param lower_limit_arg: one or more values of lower limit values for which the function \
            of the property should calculate the integral for
        :type lower_limit_arg: float or list-like objects
        :return: integral between the limits
        :rtype: float, array

        :Example:
            >>> comp_fact = get_fluid(net).all_properties["heat_capacity"].get_at_integral_value(\
                    t_upper_k, t_lower_k)

        """
        if isinstance(upper_limit_arg, pd.Series):
            ul = self.offset * upper_limit_arg.values + 0.5 * self.slope * np.power(
                upper_limit_arg.values, 2)
        else:
            ul = self.offset * np.array(upper_limit_arg) + 0.5 * self.slope * np.array(
                np.power(upper_limit_arg.values, 2))
        if isinstance(lower_limit_arg, pd.Series):
            ll = self.offset * lower_limit_arg.values + 0.5 * self.slope * np.power(
                lower_limit_arg.values, 2)
        else:
            ll = self.offset * np.array(lower_limit_arg) + 0.5 * self.slope * np.array(
                np.power(lower_limit_arg.values, 2))
        return ul - ll

    @classmethod
    def from_path(cls, path):
        """
        Reads a text file with temperature values in the first column and property values in
        second column.

        :param path:
        :type path:
        :return:
        :rtype:
        """
        slope, offset = np.loadtxt(path)
        return cls(slope, offset)


class FluidPropertyPolynominal(FluidProperty):
    """
    Creates Property with a polynominal course.
    """

    def __init__(self, x_values, y_values, polynominal_degree):
        """

        :param x_values:
        :type x_values:
        :param y_values:
        :type y_values:
        :param polynominal_degree:
        :type polynominal_degree:
        """
        super(FluidPropertyPolynominal, self).__init__()
        const = np.polyfit(x_values, y_values, polynominal_degree)
        self.prop_getter = np.poly1d(const)
        self.prop_int_getter = np.polyint(self.prop_getter)

    def get_at_value(self, arg):
        """

        :param arg: Name of the property and one or more values (x-values) for which the function \
            of the property should be calculated
        :type arg: float or list-like objects
        :return: y-value or function values
        :rtype: float, array

        :Example:
            >>> comp_fact = get_fluid(net).all_properties["heat_capacity"].get_at_value(t_k)

        """
        return self.prop_getter(arg)

    def get_at_integral_value(self, upper_limit_arg, lower_limit_arg):
        """

        :param upper_limit_arg: one or more values of upper limit values for which the function \
            of the property should calculate the integral for
        :type upper_limit_arg: float or list-like objects
        :param lower_limit_arg: one or more values of lower limit values for which the function \
            of the property should calculate the integral for
        :type lower_limit_arg: float or list-like objects
        :return: integral between the limits
        :rtype: float, array

        :Example:
            >>> comp_fact = get_fluid(net).all_properties["heat_capacity"].get_at_integral_value(\
                    t_upper_k, t_lower_k)

        """
        return self.prop_int_getter(upper_limit_arg) - self.prop_int_getter(lower_limit_arg)

    @classmethod
    def from_path(cls, path, polynominal_degree):
        """
        Reads a text file with temperature values in the first column and property values in
        second column.

        :param path: Target path of the txt file
        :type path: str
        :param polynominal_degree: degree of the polynominal
        :type polynominal_degree: int
        :return: Fluid object
        :rtype: pandapipes.FluidProperty
        """
        values = np.loadtxt(path)
        return cls(values[:, 0], values[:, 1], polynominal_degree)


class FluidPropertySutherland(FluidProperty):
    """
    Creates Property with a Sutherland model (mainly used for viscosity).
    """

    def __init__(self, eta0, t0, t_sutherland):
        """

        :param value:
        :type value:
        """
        super().__init__()
        self.eta0 = eta0
        self.t0 = t0
        self.t_sutherland = t_sutherland

    def get_at_value(self, *args):
        """

        :param arg: Name of the property
        :type arg: str
        :return: Value of the property
        :rtype: float

        :Example:
            >>> viscosity = get_fluid(net).get_property("viscosity")
        """
        return self.eta0 * (self.t0 + self.t_sutherland) / (self.t_sutherland + args[0]) \
               * np.power(args[0] / self.t0, 1.5)

    def get_at_integral_value(self, upper_limit_arg, lower_limit_arg):
        raise UserWarning("The sutherland property integral value function has not been implemented!")
        # if isinstance(upper_limit_arg, pd.Series):
        #     ul = self.offset * upper_limit_arg.values + 0.5 * self.slope * np.power(
        #         upper_limit_arg.values, 2)
        # else:
        #     ul = self.offset * np.array(upper_limit_arg) + 0.5 * self.slope * np.array(
        #         np.power(upper_limit_arg.values, 2))
        # if isinstance(lower_limit_arg, pd.Series):
        #     ll = self.offset * lower_limit_arg.values + 0.5 * self.slope * np.power(
        #         lower_limit_arg.values, 2)
        # else:
        #     ll = self.offset * np.array(lower_limit_arg) + 0.5 * self.slope * np.array(
        #         np.power(lower_limit_arg.values, 2))


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


def call_lib(fluid_name):
    """
    Creates a fluid with default fluid properties.

    :param fluid_name: Fluid which should be used
    :type fluid_name: str
    :return: Fluid - Chosen fluid with default fluid properties
    :rtype: Fluid
    """

    def interextra_property(prop):
        return FluidPropertyInterExtra.from_path(
            os.path.join(pp_dir, "properties", fluid_name, prop + ".txt"))

    def constant_property(prop):
        return FluidPropertyConstant.from_path(
            os.path.join(pp_dir, "properties", fluid_name, prop + ".txt"))

    def linear_property(prop):
        return FluidPropertyLinear.from_path(
            os.path.join(pp_dir, "properties", fluid_name, prop + ".txt"))

    liquids = ["water"]
    gases = ["air", "lgas", "hgas", "hydrogen", "methane"]

    if fluid_name == "natural_gas":
        logger.error("'natural_gas' is ambigious. Please choose 'hgas' or 'lgas' "
                     "(high- or low calorific natural gas)")
    if fluid_name not in liquids and fluid_name not in gases:
        raise AttributeError("Fluid '%s' not found in the fluid library. It might not be "
                             "implemented yet." % fluid_name)

    phase = "liquid" if fluid_name in liquids else "gas"

    density = interextra_property("density")
    viscosity = interextra_property("viscosity")
    heat_capacity = interextra_property("heat_capacity")
    molar_mass = constant_property("molar_mass")
    der_compr = constant_property("der_compressibility")
    compr = linear_property("compressibility")

    if (phase == 'gas') & (fluid_name != 'air'):
        lhv = constant_property("lower_heating_value")
        hhv = constant_property("higher_heating_value")

        return Fluid(fluid_name, phase, density=density, viscosity=viscosity,
                     heat_capacity=heat_capacity, molar_mass=molar_mass,
                     compressibility=compr, der_compressibility=der_compr, lhv=lhv, hhv=hhv)
    else:
        return Fluid(fluid_name, phase, density=density, viscosity=viscosity,
                     heat_capacity=heat_capacity, molar_mass=molar_mass, compressibility=compr,
                     der_compressibility=der_compr)


def get_fluid(net):
    """
    This function shows which fluid is used in the net.

    :param net: Current network
    :type net: pandapipesNet
    :return: Fluid - Name of the fluid which is used in the current network
    :rtype: Fluid
    """
    if "fluid" not in net or net["fluid"] is None:
        raise AttributeError("There is no fluid defined for the given net!")
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
    :return: No output.
    :type: None
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
                       "respective pandapipes.Fluid." % fluid)
        fluid = call_lib(fluid)
    net["fluid"] = fluid
