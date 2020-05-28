# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np


def calculate_mixture_viscosity(components_viscosities, components_molar_proportions,
                                components_molar_mass):
    """
    Todo: Fill out parameters.

    :param component_viscosities:
    :type component_viscosities:
    :param component_proportions:
    :type component_proportions:
    :param component_molar_mass:
    :type component_molar_mass:
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
        com_array[:, :, 1] = np.reshape(components_molar_proportions.repeat(shape[1]), shape)
        com_array[:, :, 2] = np.reshape(components_molar_mass.repeat(shape[1]), shape)
        com_array[:, :, 3] = com_array[:, :, 0] * com_array[:, :, 1] * np.sqrt(com_array[:, :, 2])
        com_array[:, :, 4] = com_array[:, :, 1] * np.sqrt(com_array[:, :, 2])
        res = com_array[:, :, 3].sum(axis=0) / com_array[:, :, 4].sum(axis=0)
    return res


def calculate_mixture_density(components_density, components_mass_proportions):
    """
    Todo: Fill out parameters.

    :param component_density:
    :type component_density:
    :param component_molar_proportions:
    :type component_molar_proportions: ?, default None
    :param component_mass_proportions:
    :type component_mass_proportions: ?, default None
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
        com_array[:, :, 0] = np.reshape(components_mass_proportions.repeat(shape[1]), shape)
        com_array[:, :, 1] = components_capacity
        com_array[:, :, 2] = com_array[:, :, 1] * com_array[:, :, 0]
        res = com_array[:, :, 2].sum(axis=0)
    return res


def calculate_mixture_molar_mass(components_molar_mass, components_molar_proportions=None,
                                 components_mass_proportions=None):
    """
    Todo: Fill out parameters.

    :param component_molar_mass:
    :type component_molar_mass:
    :param component_molar_proportions:
    :type component_molar_proportions: ?, default None
    :param component_mass_proportions:
    :type component_mass_proportions: ?, default None
    :return:
    :rtype:
    """
    if components_molar_proportions is not None:
        com_array = np.empty([len(components_molar_proportions), 3], dtype=np.float64)
        com_array[:, 0] = components_molar_proportions
        com_array[:, 1] = components_molar_mass
        com_array[:, 2] = com_array[:, 0] * com_array[:, 1]
        res = com_array[:, 2].sum()
    elif components_mass_proportions is not None:
        com_array = np.empty([len(components_mass_proportions), 3], dtype=np.float64)
        com_array[:, 0] = components_mass_proportions
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
    com_array = np.empty([len(component_molar_proportions), 4], dtype=np.float64)
    com_array[:, 0] = component_molar_proportions
    com_array[:, 1] = component_molar_mass
    com_array[:, 2] = com_array[:, 0] * com_array[:, 1]
    com_array[:, 3] = com_array[:, 2] / com_array[:, 2].sum()
    return com_array[:, 3]
