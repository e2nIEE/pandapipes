# Copyright (c) 2020-2022 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


from pandapipes import pp_dir
from pandapipes.properties.properties_toolbox import calculate_mixture_density, calculate_mixture_viscosity, \
    calculate_mixture_molar_mass, calculate_molar_fraction_from_mass_fraction, calculate_mixture_heat_capacity, \
    calculate_mixture_compressibility, calculate_mixture_calorific_values, calculate_mass_fraction_from_molar_fraction,\
    calculate_mixture_compressibility_fact, calculate_der_mixture_compressibility_fact
from pandapower.io_utils import JSONSerializableClass

try:
    from pandaplan.core import pplog as logging
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

    def get_critical_data(self):
        """
        This function returns the critical data: critical temperature, critical pressure, ascentric factor.

        :return: critical_data

        """

        return self.get_property("critical_data")

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

    def get_lower_heating_value(self):
        return self.get_property("lhv")

    def get_higher_heating_value(self):
        return self.get_property("hhv")




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
        return mean * (upper_limit_arg - lower_limit_arg)

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
        #if query for critical data cause more than one item is read
        if "critical" in path:
            value = np.loadtxt(path)
        else:
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


def create_constant_property(net, fluid_name, property_name, value, overwrite=True, warn_on_duplicates=True):
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
    net.fluid[fluid_name].add_property(property_name, prop, overwrite=overwrite,
                                       warn_on_duplicates=warn_on_duplicates)
    return prop


def create_linear_property(net, fluid_name, property_name, slope, offset, overwrite=True,
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
    net.fluid[fluid_name].add_property(property_name, prop, overwrite=overwrite,
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
    gases = ["air", "lgas", "hgas", "hydrogen", "methane", "ethane", "butane", "propane", "carbondioxide", "nitrogen"]

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
    crit = constant_property("critical_data")

    if (phase == 'gas'):
        lhv = constant_property("lower_heating_value")
        hhv = constant_property("higher_heating_value")

        return Fluid(fluid_name, phase, density=density, viscosity=viscosity,
                     heat_capacity=heat_capacity, molar_mass=molar_mass,
                     compressibility=compr, der_compressibility=der_compr, lhv=lhv, hhv=hhv, critical_data=crit)
    else:
        return Fluid(fluid_name, phase, density=density, viscosity=viscosity,
                     heat_capacity=heat_capacity, molar_mass=molar_mass, compressibility=compr,
                     der_compressibility=der_compr)


def get_fluid(net, fluid_name):
    """
    This function shows which fluid is used in the net.

    :param net: Current network
    :type net: pandapipesNet
    :return: Fluid - Name of the fluid which is used in the current network
    :rtype: Fluid
    """
    if fluid_name not in net.fluid.keys():
        raise AttributeError("There is no fluid defined for the given net!")
    fluid = net.fluid[fluid_name]
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
    if "fluid" in net and fluid.name in net["fluid"] and not overwrite:
        logger.warning("The fluid %s would replace the exisiting fluid with the same name and thus cannot be created. "
                       "Try to set overwrite to True" % (fluid.name))
        return

    if isinstance(fluid, str):
        logger.warning("Instead of a pandapipes.Fluid, a string ('%s') was passed to the fluid "
                       "argument. Internally, it will be passed to call_lib(fluid) to get the "
                       "respective pandapipes.Fluid." % fluid)
        fluid = call_lib(fluid)
    net["fluid"][fluid.name] = fluid


def get_property(net, property_name, fluid_name=None, *at_values):
    if len(net._fluid) == 1:
        return get_fluid(net, net._fluid[0]).get_property(property_name, *at_values)
    else:
        return net.fluid[fluid_name].get_property(property_name, *at_values)


def get_mixture_molar_mass(net, mass_fraction):
    molar_mass_list = [net.fluid[fluid].get_molar_mass() for fluid in net._fluid]
    return calculate_mixture_molar_mass(molar_mass_list, component_proportions=mass_fraction)


def get_mixture_density(net, temperature, mass_fraction):
    density_list = [net.fluid[fluid].get_density(temperature) for fluid in net._fluid]
    return calculate_mixture_density(density_list, mass_fraction.T)


def get_mixture_viscosity(net, temperature, mass_fraction):
    viscosity_list, molar_mass_list = [], []
    for fluid in net._fluid:
        viscosity_list += [net.fluid[fluid].get_viscosity(temperature)]
        molar_mass_list += [net.fluid[fluid].get_molar_mass()]
    molar_fraction = calculate_molar_fraction_from_mass_fraction(mass_fraction.T, np.array(molar_mass_list))
    return calculate_mixture_viscosity(viscosity_list, molar_fraction, np.array(molar_mass_list).T)


def get_mixture_heat_capacity(net, temperature, mass_fraction):
    heat_capacity_list = [net.fluid[fluid].get_heat_capacity(temperature) for fluid in net._fluid]
    return calculate_mixture_heat_capacity(heat_capacity_list, mass_fraction.T)


def get_mixture_compressibility(net, pressure, mass_fraction, temperature):
    """
    Returns the Gas law deviation coefficient: K = Z / Z_n.

    Nomenclature :

    In the literature:
    pv = ZRT
    K = Z / Z_n

    Z: Compressibility factor <-> Realgasfaktor
    K: Gas law deviation coefficient <-> Kompressibilitätszahl

    In this function:
    Z: compressibility_fact
    K: compressibility_fact / compressibility_fact_norm

    """

    critical_data_list = [net.fluid[fluid].get_critical_data() for fluid in net._fluid]
    molar_mass_list = [net.fluid[fluid].get_molar_mass() for fluid in net._fluid]
    molar_fraction = calculate_molar_fraction_from_mass_fraction(mass_fraction.T, np.array(molar_mass_list))
    compressibility_fact, compressibility_fact_norm = calculate_mixture_compressibility_fact(molar_fraction.T,
                                                                             pressure, temperature, critical_data_list)
    return compressibility_fact / compressibility_fact_norm  # K = Z / Z_n


def get_mixture_der_cmpressibility(net, pressure, mass_fraction, temperature):
    """
    Returns the derivative relative to the pressure of the Gas law deviation coefficient: dK /dp = dZ/dp * 1/Z_n.

    Nomenclature :

    In the literature:
    pv = ZRT
    K = Z / Z_n

    Z: Compressibility factor <-> Realgasfaktor
    K: Gas law deviation coefficient <-> Kompressibilitätszahl

    In this function:
    Z: compressibility_fact
    dK / dp : der_compressibility_fact / compressibility_fact_norm

    """
    critical_data_list = [net.fluid[fluid].get_critical_data() for fluid in net._fluid]
    molar_mass_list = [net.fluid[fluid].get_molar_mass() for fluid in net._fluid]
    molar_fraction = calculate_molar_fraction_from_mass_fraction(mass_fraction.T, np.array(molar_mass_list))
    der_compressibility_fact, compressibility_fact_norm = calculate_der_mixture_compressibility_fact(molar_fraction.T,
                                                                            pressure, temperature, critical_data_list)
    return der_compressibility_fact / compressibility_fact_norm  # dK /dp = dZ/dp * 1/Z_n.


def get_mixture_higher_heating_value(net, mass_fraction):
    calorific_list = np.array([net.fluid[fluid].get_property('hhv') for fluid in net._fluid])
    return calculate_mixture_calorific_values(calorific_list, mass_fraction.T)


def get_mixture_lower_heating_value(net, mass_fraction):
    calorific_list = np.array([net.fluid[fluid].get_property('lhv') for fluid in net._fluid])
    return calculate_mixture_calorific_values(calorific_list, mass_fraction.T)


def is_fluid_gas(net):
    if len(net._fluid) == 1:
        return get_fluid(net, net._fluid[0]).is_gas
    else:
        state = [get_fluid(net, fluid).is_gas for fluid in net._fluid]
        if np.all(state):
            return True
        elif np.all(~np.array(state)):
            return False
        else:
            logger.warning('Be careful. You look at system containing both fluid and gaseous fluids.')


def create_individual_fluid(fluid_name, fluid_components,
                            temperature_list, pressure_list,
                            component_proportions, proportion_type='mass', phase='gas'):
    molar_mass = []
    density = []
    viscosity = []
    heat_capacity = []
    compressibility = []
    der_compressibility = []
    high_calorific = []
    low_calorific = []
    for fl_co in fluid_components:
        fluid = call_lib(fl_co)
        molar_mass += [fluid.get_molar_mass()]
        density += [fluid.get_density(temperature_list)]
        viscosity += [fluid.get_viscosity(temperature_list)]
        heat_capacity += [fluid.get_heat_capacity(temperature_list)]
        compressibility += [fluid.get_property('compressibility', pressure_list)]
        der_compressibility += [fluid.get_property('der_compressibility', temperature_list)]
        high_calorific += [fluid.get_property('hhv')]
        low_calorific += [fluid.get_property('lhv')]
    if proportion_type == 'mass':
        mof = calculate_molar_fraction_from_mass_fraction(component_proportions, molar_mass)
        maf = np.array(component_proportions)
    elif proportion_type == 'molar':
        mof = np.array(component_proportions)
        maf = calculate_mass_fraction_from_molar_fraction(component_proportions, molar_mass)
    else:
        raise (AttributeError('proportion type %s not defined. Select either mass or molar' %proportion_type))
    dens = calculate_mixture_density(density, maf)
    visc = calculate_mixture_viscosity(viscosity, mof, np.array(molar_mass))
    heat = calculate_mixture_heat_capacity(heat_capacity, maf)
    comp = calculate_mixture_compressibility(compressibility, maf)
    derc = calculate_mixture_compressibility(der_compressibility, maf)
    mass = calculate_mixture_molar_mass(molar_mass, maf)
    higc = calculate_mixture_calorific_values(np.array(high_calorific), maf)
    lowc = calculate_mixture_calorific_values(np.array(low_calorific), maf)

    dens = FluidPropertyInterExtra(temperature_list, dens)
    visc = FluidPropertyInterExtra(temperature_list, visc)
    heat = FluidPropertyInterExtra(temperature_list, heat)
    mass = FluidPropertyConstant(mass)
    higc = FluidPropertyConstant(higc)
    lowc = FluidPropertyConstant(lowc)
    derc = FluidPropertyInterExtra(temperature_list, derc)
    comp = FluidPropertyInterExtra(pressure_list, comp)

    fluid = Fluid(fluid_name, phase, density=dens, viscosity=visc, heat_capacity=heat, molar_mass=mass,
                  der_compressibility=derc, compressibility=comp, hhv=higc, lhv=lowc)
    return fluid
