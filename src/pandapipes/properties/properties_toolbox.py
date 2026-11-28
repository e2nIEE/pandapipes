# Copyright (c) 2020-2026 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np

from pandapipes.constants import NORMAL_TEMPERATURE, NORMAL_PRESSURE
from pandapipes.idx_branch import IdxBranch
from pandapipes.idx_node import IdxNode
from pandapipes.pf.internals_toolbox import get_from_nodes_corrected, get_to_nodes_corrected

def water_saturation_temperature_k(absolute_pressure_mpa):
    """Calculate the water saturation temperature in K from absolute pressure in MPa."""
    n1, n2, n3, n4, n5, n6, n7, n8, n9, n10 = (
        0.11670521452767e4,
        -0.72421316703206e6,
        -0.17073846940092e2,
        0.12020824702470e5,
        -0.32325550322333e7,
        0.14915108613530e2,
        -0.48232657361591e4,
        0.40511340542057e6,
        -0.23855557567849,
        0.65017534844798e3,
    )

    pressure = np.asarray(absolute_pressure_mpa, dtype=float)
    valid = (pressure >= 0.000611213) & (pressure <= 22.064)
    beta = np.where(valid, pressure, np.nan) ** 0.25
    e = beta**2 + n3 * beta + n6
    f = n1 * beta**2 + n4 * beta + n7
    g = n2 * beta**2 + n5 * beta + n8
    d = 2 * g / (-f - np.sqrt(f**2 - 4 * e * g))
    return (n10 + d - np.sqrt((n10 + d)**2 - 4 * (n9 + n10 * d))) / 2


def calculate_mixture_viscosity(components_viscosities, components_molar_proportions,
                                components_molar_mass):
    """Todo: Fill out parameters.

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
        com_array[:, :, 1] = np.reshape(components_molar_proportions.repeat(shape[1]), shape)
        com_array[:, :, 2] = np.reshape(components_molar_mass.repeat(shape[1]), shape)
        com_array[:, :, 3] = com_array[:, :, 0] * com_array[:, :, 1] * np.sqrt(com_array[:, :, 2])
        com_array[:, :, 4] = com_array[:, :, 1] * np.sqrt(com_array[:, :, 2])
        res = com_array[:, :, 3].sum(axis=0) / com_array[:, :, 4].sum(axis=0)
    return res


def calculate_mixture_density(components_density, components_mass_proportions):
    """Todo: Fill out parameters.

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
        com_array[:, :, 0] = np.reshape(components_mass_proportions.repeat(shape[1]), shape)
        com_array[:, :, 1] = components_density
        com_array[:, :, 2] = com_array[:, :, 0] / com_array[:, :, 1]
        res = 1 / com_array[:, :, 2].sum(axis=0)
    return res


def calculate_mixture_heat_capacity(components_capacity, components_mass_proportions):
    """Todo: Fill out parameters.

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
    """Todo: Fill out parameters.

    :param components_molar_mass:
    :type components_molar_mass:
    :param components_molar_proportions:
    :type components_molar_proportions: ?, default None
    :param components_mass_proportions:
    :type components_mass_proportions: ?, default None
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
    """Todo: Fill out parameters.

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


def get_branch_real_density(fluid, node_pit, branch_pit):
    from_nodes = get_from_nodes_corrected(branch_pit)
    t_from = node_pit[from_nodes, IdxNode.TINIT]
    t_to = branch_pit[:, IdxBranch.TOUTINIT]
    if fluid.is_gas:
        from_p = node_pit[from_nodes, IdxNode.PINIT] + node_pit[from_nodes, IdxNode.PAMB]
        to_nodes = get_to_nodes_corrected(branch_pit)
        to_p = node_pit[to_nodes, IdxNode.PINIT] + node_pit[to_nodes, IdxNode.PAMB]
        normal_rho = fluid.get_density(NORMAL_TEMPERATURE)
        from_rho = np.divide(normal_rho * NORMAL_TEMPERATURE * from_p,
                             t_from * NORMAL_PRESSURE * fluid.get_compressibility(from_p, t_from))
        to_rho = np.divide(normal_rho * NORMAL_TEMPERATURE * to_p,
                           t_to * NORMAL_PRESSURE * fluid.get_compressibility(to_p, t_to))
    else:
        from_rho = fluid.get_density(t_from)
        to_rho = fluid.get_density(t_to)
    rho = (from_rho + to_rho) / 2
    return rho

def get_branch_real_eta(fluid, node_pit, branch_pit, pm):
    from_nodes = get_from_nodes_corrected(branch_pit)
    t_from = node_pit[from_nodes, IdxNode.TINIT]
    t_to = branch_pit[:, IdxBranch.TOUTINIT]
    tm = (t_from + t_to) / 2
    eta = fluid.get_viscosity(tm, p_bar=pm)
    return eta

def get_branch_cp(fluid, node_pit, branch_pit):
    from_nodes = get_from_nodes_corrected(branch_pit)
    t_from = node_pit[from_nodes, IdxNode.TINIT]
    t_to = branch_pit[:, IdxBranch.TOUTINIT]
    cp_from = fluid.get_heat_capacity(t_from)
    cp_to = fluid.get_heat_capacity(t_to)
    cp = (cp_from + cp_to) / 2
    return cp