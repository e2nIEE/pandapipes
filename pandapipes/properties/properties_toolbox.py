# Copyright (c) 2020-2022 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np


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
