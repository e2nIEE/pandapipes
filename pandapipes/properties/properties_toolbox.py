# Copyright (c) 2020-2022 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
from scipy.optimize import newton
from pandapipes.constants import NORMAL_TEMPERATURE, NORMAL_PRESSURE, P_CONVERSION

p_n = NORMAL_PRESSURE
T_n = NORMAL_TEMPERATURE

def calculate_mixture_viscosity(components_viscosities, components_molar_proportions,
                                components_molar_mass):
    """
    Todo: Fill out parameters.

    :param components_viscosities:
    :type components_viscosities:
    :param components_molar_proportions:
    :type components_molar_proportions:
    :param components_molar_mass:
    :type components_molar_mass:
    :return:
    :rtype:
    """
    shape = np.shape(components_viscosities)
    if len(shape) == 1:
        com_array = np.empty([shape[0], 5], dtype=np.float64)
        com_array[:, 0] = components_viscosities
        com_array[:, 1] = components_molar_proportions
        com_array[:, 2] = components_molar_mass
        com_array[:, 3] = com_array[:, 0] * com_array[:, 1] * np.sqrt(com_array[:, 2])
        com_array[:, 4] = com_array[:, 1] * np.sqrt(com_array[:, 2])
        res = com_array[:, 3].sum() / com_array[:, 4].sum()
    else:
        com_array = np.empty([shape[0], shape[1], 5], dtype=np.float64)
        com_array[:, :, 0] = components_viscosities
        if np.shape(components_viscosities) == np.shape(components_molar_proportions):
            com_array[:, :, 1] = components_molar_proportions
        else:
            com_array[:, :, 1] = np.reshape(components_molar_proportions.repeat(shape[1]), shape)
        com_array[:, :, 2] = np.reshape(components_molar_mass.repeat(shape[1]), shape)
        com_array[:, :, 3] = com_array[:, :, 0] * com_array[:, :, 1] * np.sqrt(com_array[:, :, 2])
        com_array[:, :, 4] = com_array[:, :, 1] * np.sqrt(com_array[:, :, 2])
        res = com_array[:, :, 3].sum(axis=0) / com_array[:, :, 4].sum(axis=0)
    return res


def calculate_mixture_density(components_density, components_mass_proportions):
    """
    Todo: Fill out parameters.

    :param components_density:
    :type components_density:
    :param components_mass_proportions:
    :type components_mass_proportions: ?, default None
    :return:
    :rtype:
    """
    shape = np.shape(components_density)
    if len(shape) == 1:
        com_array = np.empty([shape[0], 3], dtype=np.float64)
        com_array[:, 0] = components_mass_proportions
        com_array[:, 1] = components_density
        com_array[:, 2] = com_array[:, 0] / com_array[:, 1]
        res = 1 / com_array[:, 2].sum()
    else:
        com_array = np.empty([shape[0], shape[1], 3], dtype=np.float64)
        if np.shape(components_density) == np.shape(components_mass_proportions):
            com_array[:, :, 0] = components_mass_proportions
        else:
            com_array[:, :, 0] = np.reshape(components_mass_proportions.repeat(shape[1]), shape)
        com_array[:, :, 1] = components_density
        com_array[:, :, 2] = com_array[:, :, 0] / com_array[:, :, 1]
        res = 1 / com_array[:, :, 2].sum(axis=0)
    return res


def calculate_mixture_heat_capacity(components_capacity, components_mass_proportions):
    """
    Todo: Fill out parameters.

    :param components_capacity:
    :type components_capacity:
    :param components_mass_proportions:
    :type components_mass_proportions:
    :return:
    :rtype:
    """
    shape = np.shape(components_capacity)
    if len(shape) == 1:
        com_array = np.empty([shape[0], 3], dtype=np.float64)
        com_array[:, 0] = components_mass_proportions
        com_array[:, 1] = components_capacity
        com_array[:, 2] = com_array[:, 1] * com_array[:, 0]
        res = com_array[:, 2].sum()
    else:
        com_array = np.empty([shape[0], shape[1], 3], dtype=np.float64)
        if np.shape(components_capacity) == np.shape(components_mass_proportions):
            com_array[:, :, 0] = components_mass_proportions
        else:
            com_array[:, :, 0] = np.reshape(components_mass_proportions.repeat(shape[1]), shape)
        com_array[:, :, 1] = components_capacity
        com_array[:, :, 2] = com_array[:, :, 1] * com_array[:, :, 0]
        res = com_array[:, :, 2].sum(axis=0)
    return res


def calculate_mixture_molar_mass(components_molar_mass, component_proportions, proportion_type='mass'):
    """
    Todo: Fill out parameters.

    :param components_molar_mass:
    :type components_molar_mass:
    :param components_molar_proportions:
    :type components_molar_proportions: ?, default None
    :param components_mass_proportions:
    :type components_mass_proportions: ?, default None
    :return:
    :rtype:
    """
    if proportion_type == 'molar':
        com_array = np.empty([len(component_proportions), 3], dtype=np.float64)
        com_array[:, 0] = component_proportions
        com_array[:, 1] = components_molar_mass
        com_array[:, 2] = com_array[:, 0] * com_array[:, 1]
        res = com_array[:, 2].sum()
    elif proportion_type == 'mass':
        com_array = np.empty([len(component_proportions), 3], dtype=np.float64)
        com_array[:, 0] = component_proportions
        com_array[:, 1] = components_molar_mass
        com_array[:, 2] = com_array[:, 0] / com_array[:, 1]
        res = 1 / com_array[:, 2].sum()
    else:
        raise (AttributeError('at least one needs to be different from None: '
                              'component_molar_proportions, component_mass_proportions'))
    return res


def calculate_mass_fraction_from_molar_fraction(component_molar_proportions, component_molar_mass):
    """
    Todo: Fill out parameters.

    :param component_molar_proportions:
    :type component_molar_proportions:
    :param component_molar_mass:
    :type component_molar_mass:
    :return:
    :rtype:
    """
    shape = np.shape(component_molar_proportions)
    if len(shape) == 1:
        com_array = np.empty([len(component_molar_proportions), 4], dtype=np.float64)
        com_array[:, 0] = component_molar_proportions
        com_array[:, 1] = component_molar_mass
        com_array[:, 2] = com_array[:, 0] * com_array[:, 1]
        com_array[:, 3] = com_array[:, 2] / com_array[:, 2].sum()
        res = com_array[:, 3]
    else:
        com_array = np.empty([shape[0], shape[1], 4], dtype=np.float64)
        com_array[:, :, 0] = component_molar_proportions
        com_array[:, :, 1] = np.reshape(component_molar_mass.repeat(shape[1]), shape)
        com_array[:, :, 2] = com_array[:, :, 0] * com_array[:, :, 1]
        com_array[:, :, 3] = com_array[:, :, 2] / com_array[:, :, 2].sum()
        res = com_array[:, :, 3]
    return res


def calculate_molar_fraction_from_mass_fraction(component_mass_proportions, component_molar_mass):
    """
    Todo: Fill out parameters.

    :param component_molar_proportions:
    :type component_molar_proportions:
    :param component_molar_mass:
    :type component_molar_mass:
    :return:
    :rtype:
    """
    shape = np.shape(component_mass_proportions)

    # todo: check the one-fluid case
    if len(shape) == 1:
        com_array = np.empty([len(component_mass_proportions), 4], dtype=np.float64)
        com_array[:, 0] = component_mass_proportions
        com_array[:, 1] = component_molar_mass.T
        com_array[:, 2] = com_array[:, 0] / com_array[:, 1]
        com_array[:, 3] = com_array[:, 2] / com_array[:, 2].sum()
        res = com_array[:, 3]
    else:
        com_array = np.empty([shape[0], shape[1], 4], dtype=np.float64)
        com_array[:, :, 0] = component_mass_proportions
        com_array[:, :, 1] = np.reshape(component_molar_mass.repeat(shape[1]), shape)
        com_array[:, :, 2] = com_array[:, :, 0] / com_array[:, :, 1] # molar fraction of each component divided by its
        # respective molar mass
        com_array[:, :, 3] = com_array[:, :, 2] / com_array[:, :, 2].sum(axis=0)
        res = com_array[:, :, 3]
    return res


def calculate_mixture_compressibility(components_compressibility, components_mass_proportions):
    """
    Todo: Needs to be checked

    :param component_molar_proportions:
    :type component_molar_proportions:
    :param component_molar_mass:
    :type component_molar_mass:
    :return:
    :rtype:
    """
    shape = np.shape(components_compressibility)
    if len(shape) == 1:
        com_array = np.empty([shape[0], 3], dtype=np.float64)
        com_array[:, 0] = components_mass_proportions
        com_array[:, 1] = components_compressibility
        com_array[:, 2] = com_array[:, 1] * com_array[:, 0]
        res = com_array[:, 2].sum()
    else:
        com_array = np.empty([shape[0], shape[1], 3], dtype=np.float64)
        if np.shape(components_compressibility) == np.shape(components_mass_proportions):
            com_array[:, :, 0] = components_mass_proportions
        else:
            com_array[:, :, 0] = np.reshape(components_mass_proportions.repeat(shape[1]), shape)
        com_array[:, :, 1] = components_compressibility
        com_array[:, :, 2] = com_array[:, :, 1] * com_array[:, :, 0]
        res = com_array[:, :, 2].sum(axis=0)
    return res


def calculate_mixture_calorific_values(components_calorific_values, components_mass_proportions):
    """
    Todo: Needs to be checked

    :param component_molar_proportions:
    :type component_molar_proportions:
    :param component_molar_mass:
    :type component_molar_mass:
    :return:
    :rtype:
    """
    shape = np.shape(components_mass_proportions)
    if len(shape) == 1:
        com_array = np.empty([shape[0], 3], dtype=np.float64)
        com_array[:, 0] = components_mass_proportions
        com_array[:, 1] = components_calorific_values.T
        com_array[:, 2] = com_array[:, 1] * com_array[:, 0]
        res = com_array[:, 2].sum()
    else:
        com_array = np.empty([shape[0], shape[1], 3], dtype=np.float64)
        if np.shape(components_calorific_values) == np.shape(components_mass_proportions):
            com_array[:, :, 0] = components_calorific_values
        else:
            com_array[:, :, 0] = np.reshape(components_calorific_values.repeat(shape[1]), shape)
        com_array[:, :, 1] = components_mass_proportions
        com_array[:, :, 2] = com_array[:, :, 1] * com_array[:, :, 0]
        res = com_array[:, :, 2].sum(axis=0)
    return res


def _calculate_A_B(_p, _T, _mf, _p_crit, _T_crit, _acent_fact):
    """
    Calculates the A and B mixture coefficients of the Redlich-Kwong equation. See eq. (18) and (19) page 1200 in:

    Giorgio Soave,
    Equilibrium constants from a modified Redlich-Kwong equation of state,
    Chemical Engineering Science,
    Volume 27, Issue 6,
    1972,
    https://doi.org/10.1016/0009-2509(72)80096-4.
    (https://www.sciencedirect.com/science/article/pii/0009250972800964


    :param _p: absolute pressure in bar
    :type _p: array
    :param _T: temperature in K
    :type _T: array
    :param _mf: molar fractions of the fluids
    :type _mf: array
    :param _p_crit: critical pressure of the fluids
    :type _p_crit: array
    :param _T_crit: critical temperature of the fluids
    :type _T_crit: array
    :param _acent_fact: acentric factor of the fluids
    :type _acent_fact: array
    :return: (A-mixture, B-mixture)
    :rtype: (array, array)
    """

    summation_axis = 0

    _p_in_pa = P_CONVERSION * _p
    _p_crit_in_pa = P_CONVERSION * _p_crit

    factor_a = 0.42747 * _p_in_pa / (_T ** 2)
    factor_b = 0.08664 * _p_in_pa / _T

    if _mf.ndim == 1:
        nbr_fluid = np.shape(_mf)[0]
    else:
        nbr_node = np.shape(_mf)[0]
        nbr_fluid = np.shape(_mf)[1]

        _T = np.tile(_T, (nbr_fluid, 1)).T  # duplicate _T horizontally and then transpose it
        _p_in_pa = np.tile(_p_in_pa, (nbr_fluid, 1)).T

        summation_axis = 1

    m = 0.480 + 1.574 * _acent_fact - 0.176 * _acent_fact ** 2

    T_red = _T / _T_crit

    sqrt_alpha = (1 + m * (1 - T_red ** 0.5))

    sum_a = (_mf * _T_crit * sqrt_alpha / _p_crit_in_pa ** 0.5).sum(axis=summation_axis) ** 2
    A_mixture = factor_a * sum_a

    sum_b = (_mf * _T_crit / _p_crit_in_pa).sum(axis=summation_axis)
    B_mixture = factor_b * sum_b

    return A_mixture, B_mixture


def _func_of_Z(Z, _p, _T, _mf, critical_data):
    """
    Extracts the critical data, calls calculate_A_B and returns the Redlich-Kwong equation:
    Z³ - Z² + Z (A -B-B²) -AB

    :param Z:
    :type Z:
    :param _p: absolute pressures in bar
    :type _p: array
    :param _T: temperatures in Kelvin
    :type _T: array
    :param _mf: molar fractions of the mixtures
    :type _mf: 2d-array, rows <-> branches, columns <-> fluids
    :param critical_data: critical temperatures, pressures and acentric factors of the fluids composing the mixtures
    :type critical_data: list of arrays, each item in the list corresponds to a fluid
    :return:
    :rtype:
    """
    p_crit = np.array([item[1] for item in critical_data])
    T_crit = np.array([item[0] for item in critical_data])
    acent_fact = np.array([item[2] for item in critical_data])

    A_mixture, B_mixture = _calculate_A_B(_p, _T, _mf, p_crit, T_crit, acent_fact)
    return Z ** 3 - Z ** 2 + Z * (A_mixture - B_mixture - B_mixture ** 2) - A_mixture * B_mixture


def _der_func_of_Z(Z, _p, _T, _mf, critical_data):
    """
    Extracts the critical data, calls calculate_A_B and returns the derivative of the Redlich-Kwong equation:
    3Z² - 2Z + (A -B-B²)

    :param Z:
    :type Z:
    :param _p: absolute pressures in bar
    :type _p: array
    :param _T: temperatures in Kelvin
    :type _T: array
     :param _mf: molar fractions of the mixtures
    :type _mf: 2d-array, rows <-> branches, columns <-> fluids
    :param critical_data: critical temperatures, pressures and acentric factors of the fluids composing the mixtures
    :type critical_data: list of arrays, each item in the list corresponds to a fluid

    :return:
    :rtype:
    """
    p_crit = np.array([item[1] for item in critical_data])
    T_crit = np.array([item[0] for item in critical_data])
    acent_fact = np.array([item[2] for item in critical_data])

    A_mixture, B_mixture = _calculate_A_B(_p, _T, _mf, p_crit, T_crit, acent_fact)
    return 3 * Z ** 2 - 2 * Z + (A_mixture - B_mixture - B_mixture ** 2)


def _second_der_func_of_Z(Z, _p, _T, _mf, critical_data):
    """
    Extracts the critical data, calls calculate_A_B and returns the second derivative of the Redlich-Kwong equation:
    6Z - 2

    :param Z:
    :type Z:
    :param _p: absolute pressures in bar
    :type _p: array
    :param _T: temperatures in Kelvin
    :type _T: array
    :param _mf: molar fractions of the mixtures
    :type _mf: 2d-array, rows <-> branches, columns <-> fluids
    :param critical_data: critical temperatures, pressures and acentric factors of the fluids composing the mixtures
    :type critical_data: list of arrays, each item in the list corresponds to a fluid
    :return:
    :rtype:
    """
    p_crit = np.array([item[1] for item in critical_data])
    T_crit = np.array([item[0] for item in critical_data])
    acent_fact = np.array([item[2] for item in critical_data])

    return 6 * Z - 2


def calculate_mixture_compressibility_fact(_mf, _p, _T, critical_data):
    """

    calculates Z and Z_n (compressibility factor; in ger. Realgasfaktor) based on the SRK method*. Finds the root of
    _func_of_Z using th Newton method.

    *Giorgio Soave,
    Equilibrium constants from a modified Redlich-Kwong equation of state,
    Chemical Engineering Science,
    Volume 27, Issue 6,
    1972,
    https://doi.org/10.1016/0009-2509(72)80096-4.
    (https://www.sciencedirect.com/science/article/pii/0009250972800964)

    :param _mf: molar fractions of the mixtures
    :type _mf: 2d-array, rows <-> branches, columns <-> fluids
    :param _p: absolute pressures in bar
    :type _p: array
    :param _T: temperatures in Kelvin
    :type _T: array
    :param critical_data: critical temperatures, pressures and acentric factors of the fluids composing the mixtures
    :type critical_data: list of arrays, each item in the list corresponds to a fluid
    :return: Z and Z_n (compressibility factor and compressibility factor at normal conditions).
    :rtype: (ndarray, ndarray)
    """
    critical_data_list = [item[0] for item in critical_data]
    nbr_node = np.shape(_mf)[0]
    # Todo: Is unique node case covered? "if statement" to differentiate between unique node and multiple node cases ?
    start_value = 0.9
    start_value = np.array([list([start_value]) * nbr_node])[0]

    comp_fact = newton(_func_of_Z, start_value, fprime=_der_func_of_Z, args=(_p, _T, _mf, critical_data_list))
    comp_fact_norm = newton(_func_of_Z, start_value, fprime=_der_func_of_Z, args=(p_n, T_n, _mf, critical_data_list))
    return comp_fact, comp_fact_norm


def calculate_der_mixture_compressibility_fact(_mf, _p, _T, critical_data):
    """

    calculates dZ/dp and Z_n (Z: compressibility factor; in ger. Realgasfaktor) based on the SRK method*.

    *Giorgio Soave,
    Equilibrium constants from a modified Redlich-Kwong equation of state,
    Chemical Engineering Science,
    Volume 27, Issue 6,
    1972,
    https://doi.org/10.1016/0009-2509(72)80096-4.
    (https://www.sciencedirect.com/science/article/pii/0009250972800964)

    :param _mf: molar fractions of the mixtures at all junctions
    :type _mf: 2d-array, rows <-> branches, columns <-> fluids
    :param _p: absolute pressures in bar
    :type _p: array
    :param _T: temperatures at all junctions in Kelvin
    :type _T: array
    :param critical_data: critical temperatures, pressures and acentric factors of the fluids composing the mixtures
    :type critical_data: list of arrays, each item in the list corresponds to a fluid
    :return: dZ/dp and Z_n (compressibility factor; in ger. Realgasfaktor).
    :rtype: (ndarray, ndarray)

    """
    critical_data_list = [item[0] for item in critical_data]
    nbr_node = np.shape(_mf)[0]
    # Todo: Is unique node case covered? "if statement" to differentiate between unique node and multiple node cases ?
    start_value = 0.9
    start_value = np.array([list([start_value]) * nbr_node])[0]

    # d(Z/Z_n)/ dp = dZ/dp * 1/Z_n, no need to differentiate Z_n
    der_comp_fact = newton(_der_func_of_Z, start_value, fprime=_second_der_func_of_Z,
                           args=(_p, _T, _mf, critical_data_list))
    comp_fact_norm = newton(_func_of_Z, start_value, fprime=_der_func_of_Z, args=(p_n, T_n, _mf, critical_data_list))
    return der_comp_fact, comp_fact_norm