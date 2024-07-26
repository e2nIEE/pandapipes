# Copyright (c) 2020-2024 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import pandas as pd
import numpy as np
import os
from pandapipes import pp_dir
from pandapipes.properties.properties_toolbox import calculate_mixture_viscosity, \
    calculate_mixture_molar_mass, calculate_mass_fraction_from_molar_fraction, \
    calculate_mixture_density, calculate_mixture_heat_capacity
from scipy.interpolate import interp1d

def test_mixture_viscosity_lgas():
    test_mix_viscos = pd.read_csv(os.path.join(pp_dir, 'properties', 'lgas', 'viscosity.txt'),
                                  header=None, sep=' ', comment='#').values[:, 1]
    viscos_ch4 = pd.read_csv(os.path.join(pp_dir, 'properties', 'methane', 'viscosity.txt'),
                             header=None, sep=' ', comment='#').values
    viscos_n2 = pd.read_csv(os.path.join(pp_dir, 'properties', 'nitrogen', 'viscosity.txt'),
                            header=None, sep=' ', comment='#').values
    viscos_co2 = pd.read_csv(os.path.join(pp_dir, 'properties', 'carbondioxide', 'viscosity.txt'),
                             header=None, sep=' ', comment='#').values
    viscos_c2h6 = pd.read_csv(os.path.join(pp_dir, 'properties', 'ethane', 'viscosity.txt'),
                             header=None, sep=' ', comment='#').values
    molar_ch4 = pd.read_csv(os.path.join(pp_dir, 'properties', 'methane', 'molar_mass.txt'),
                            header=None, sep=' ', comment='#').values[0]
    molar_n2 = pd.read_csv(os.path.join(pp_dir, 'properties', 'nitrogen', 'molar_mass.txt'),
                           header=None, sep=' ', comment='#').values[0]
    molar_co2 = pd.read_csv(os.path.join(pp_dir, 'properties', 'carbondioxide', 'molar_mass.txt'),
                            header=None, sep=' ', comment='#').values[0]
    molar_c2h6 = pd.read_csv(os.path.join(pp_dir, 'properties', 'ethane', 'molar_mass.txt'),
                             header=None, sep=' ', comment='#').values[0]
    values_to_take = [263, 273, 283, 293, 303, 313, 323, 333]
    viscos_ch4 = interp1d(viscos_ch4[:, 0], viscos_ch4[:, 1])
    viscos_ch4 = viscos_ch4((values_to_take))
    viscos_n2 = interp1d(viscos_n2[:, 0], viscos_n2[:, 1], fill_value='extrapolate')
    viscos_n2 = viscos_n2((values_to_take))
    viscos_co2 = interp1d(viscos_co2[:, 0], viscos_co2[:, 1])
    viscos_co2 = viscos_co2((values_to_take))
    viscos_c2h6 = interp1d(viscos_c2h6[:, 0], viscos_c2h6[:, 1])
    viscos_c2h6 = viscos_c2h6((values_to_take))
    components_viscosities = np.concatenate([[viscos_ch4], [viscos_n2], [viscos_co2], [viscos_c2h6]])
    components_molar_proportions = np.array([.85, .103, .013, .034])
    components_molar_mass = np.concatenate([molar_ch4, molar_n2, molar_co2, molar_c2h6])
    mix_viscos = calculate_mixture_viscosity(
        components_viscosities, components_molar_proportions, components_molar_mass)

    assert np.allclose(mix_viscos, test_mix_viscos, rtol=0, atol=1e-8)

    mix_viscos = calculate_mixture_viscosity(
        components_viscosities[:, 0], components_molar_proportions, components_molar_mass)

    assert np.allclose(mix_viscos, test_mix_viscos, rtol=0, atol=2e-5)


def test_mixture_viscosity_hgas():
    test_mix_viscos = pd.read_csv(os.path.join(pp_dir, 'properties', 'hgas', 'viscosity.txt'),
                                  header=None, sep=' ', comment='#').values[:, 1]

    viscos_ch4 = pd.read_csv(os.path.join(pp_dir, 'properties', 'methane', 'viscosity.txt'),
                             header=None, sep=' ', comment='#').values
    viscos_n2 = pd.read_csv(os.path.join(pp_dir, 'properties', 'nitrogen', 'viscosity.txt'),
                            header=None, sep=' ', comment='#').values
    viscos_co2 = pd.read_csv(os.path.join(pp_dir, 'properties', 'carbondioxide', 'viscosity.txt'),
                             header=None, sep=' ', comment='#').values
    viscos_c2h6 = pd.read_csv(os.path.join(pp_dir, 'properties', 'ethane', 'viscosity.txt'),
                             header=None, sep=' ', comment='#').values
    molar_ch4 = pd.read_csv(os.path.join(pp_dir, 'properties', 'methane', 'molar_mass.txt'),
                            header=None, sep=' ', comment='#').values[0]
    molar_n2 = pd.read_csv(os.path.join(pp_dir, 'properties', 'nitrogen', 'molar_mass.txt'),
                           header=None, sep=' ', comment='#').values[0]
    molar_co2 = pd.read_csv(os.path.join(pp_dir, 'properties', 'carbondioxide', 'molar_mass.txt'),
                            header=None, sep=' ', comment='#').values[0]
    molar_c2h6 = pd.read_csv(os.path.join(pp_dir, 'properties', 'ethane', 'molar_mass.txt'),
                             header=None, sep=' ', comment='#').values[0]
    values_to_take = [263, 273, 283, 293, 303, 313, 323, 333]
    viscos_ch4 = interp1d(viscos_ch4[:, 0], viscos_ch4[:, 1])
    viscos_ch4 = viscos_ch4((values_to_take))
    viscos_n2 = interp1d(viscos_n2[:, 0], viscos_n2[:, 1], fill_value='extrapolate')
    viscos_n2 = viscos_n2((values_to_take))
    viscos_co2 = interp1d(viscos_co2[:, 0], viscos_co2[:, 1])
    viscos_co2 = viscos_co2((values_to_take))
    viscos_c2h6 = interp1d(viscos_c2h6[:, 0], viscos_c2h6[:, 1])
    viscos_c2h6 = viscos_c2h6((values_to_take))
    components_viscosities = np.concatenate([[viscos_ch4], [viscos_n2], [viscos_co2], [viscos_c2h6]])
    components_molar_proportions = np.array([.964, .005, .005, .026])
    components_molar_mass = np.concatenate([molar_ch4, molar_n2, molar_co2, molar_c2h6])
    mix_viscos = calculate_mixture_viscosity(
        components_viscosities, components_molar_proportions, components_molar_mass)

    assert np.allclose(mix_viscos, test_mix_viscos, rtol=0, atol=1e-10)


def test_mixture_density_lgas():
    test_mix_density = pd.read_csv(os.path.join(pp_dir, 'properties', 'lgas', 'density.txt'),
                                   header=None, sep=' ', comment='#').values[:, 1]

    dens_ch4 = pd.read_csv(os.path.join(pp_dir, 'properties', 'methane', 'density.txt'),
                           header=None, sep=' ', comment='#').values
    dens_n2 = pd.read_csv(os.path.join(pp_dir, 'properties', 'nitrogen', 'density.txt'),
                            header=None, sep=' ', comment='#').values
    dens_co2 = pd.read_csv(os.path.join(pp_dir, 'properties', 'carbondioxide', 'density.txt'),
                           header=None, sep=' ', comment='#').values
    dens_c2h6 = pd.read_csv(os.path.join(pp_dir, 'properties', 'ethane', 'density.txt'),
                            header=None, sep=' ', comment='#').values
    molar_ch4 = pd.read_csv(os.path.join(pp_dir, 'properties', 'methane', 'molar_mass.txt'),
                            header=None, sep=' ', comment='#').values[0]
    molar_n2 = pd.read_csv(os.path.join(pp_dir, 'properties', 'nitrogen', 'molar_mass.txt'),
                           header=None, sep=' ', comment='#').values[0]
    molar_co2 = pd.read_csv(os.path.join(pp_dir, 'properties', 'carbondioxide', 'molar_mass.txt'),
                            header=None, sep=' ', comment='#').values[0]
    molar_c2h6 = pd.read_csv(os.path.join(pp_dir, 'properties', 'ethane', 'molar_mass.txt'),
                             header=None, sep=' ', comment='#').values[0]
    values_to_take = [263, 273, 283, 293, 303, 313, 323, 333]
    dens_ch4 = interp1d(dens_ch4[:, 0], dens_ch4[:, 1])
    dens_ch4 = dens_ch4((values_to_take))
    dens_n2 = interp1d(dens_n2[:, 0], dens_n2[:, 1], fill_value='extrapolate')
    dens_n2 = dens_n2((values_to_take))
    dens_co2 = interp1d(dens_co2[:, 0], dens_co2[:, 1])
    dens_co2 = dens_co2((values_to_take))
    dens_c2h6 = interp1d(dens_c2h6[:, 0], dens_c2h6[:, 1])
    dens_c2h6 = dens_c2h6((values_to_take))
    components_density = np.concatenate([[dens_ch4], [dens_n2], [dens_co2], [dens_c2h6]])
    components_molar_proportions = np.array([.85, .103, .013, .034])
    components_molar_mass = np.concatenate([molar_ch4, molar_n2, molar_co2, molar_c2h6])
    components_mass_proportions = calculate_mass_fraction_from_molar_fraction(
        components_molar_proportions,
        components_molar_mass)
    mix_dens = calculate_mixture_density(
        components_density, components_mass_proportions=components_mass_proportions)

    assert np.allclose(mix_dens, test_mix_density, rtol=0, atol=1e-10)

    mix_dens = calculate_mixture_density(
        components_density[:, 0], components_mass_proportions=components_mass_proportions)

    assert np.allclose(mix_dens, test_mix_density[0], rtol=0, atol=1e-10)


def test_mixture_density_hgas():
    test_mix_density = pd.read_csv(os.path.join(pp_dir, 'properties', 'hgas', 'density.txt'),
                                   header=None, sep=' ', comment='#').values[:, 1]

    dens_ch4 = pd.read_csv(os.path.join(pp_dir, 'properties', 'methane', 'density.txt'),
                           header=None, sep=' ', comment='#').values
    dens_n2 = pd.read_csv(os.path.join(pp_dir, 'properties', 'nitrogen', 'density.txt'),
                            header=None, sep=' ', comment='#').values
    dens_co2 = pd.read_csv(os.path.join(pp_dir, 'properties', 'carbondioxide', 'density.txt'),
                           header=None, sep=' ', comment='#').values
    dens_c2h6 = pd.read_csv(os.path.join(pp_dir, 'properties', 'ethane', 'density.txt'),
                            header=None, sep=' ', comment='#').values
    molar_ch4 = pd.read_csv(os.path.join(pp_dir, 'properties', 'methane', 'molar_mass.txt'),
                            header=None, sep=' ', comment='#').values[0]
    molar_n2 = pd.read_csv(os.path.join(pp_dir, 'properties', 'nitrogen', 'molar_mass.txt'),
                           header=None, sep=' ', comment='#').values[0]
    molar_co2 = pd.read_csv(os.path.join(pp_dir, 'properties', 'carbondioxide', 'molar_mass.txt'),
                            header=None, sep=' ', comment='#').values[0]
    molar_c2h6 = pd.read_csv(os.path.join(pp_dir, 'properties', 'ethane', 'molar_mass.txt'),
                             header=None, sep=' ', comment='#').values[0]
    values_to_take = [263, 273, 283, 293, 303, 313, 323, 333]
    dens_ch4 = interp1d(dens_ch4[:, 0], dens_ch4[:, 1])
    dens_ch4 = dens_ch4((values_to_take))
    dens_n2 = interp1d(dens_n2[:, 0], dens_n2[:, 1], fill_value='extrapolate')
    dens_n2 = dens_n2((values_to_take))
    dens_co2 = interp1d(dens_co2[:, 0], dens_co2[:, 1])
    dens_co2 = dens_co2((values_to_take))
    dens_c2h6 = interp1d(dens_c2h6[:, 0], dens_c2h6[:, 1])
    dens_c2h6 = dens_c2h6((values_to_take))
    components_density = np.concatenate([[dens_ch4], [dens_n2], [dens_co2], [dens_c2h6]])
    components_molar_proportions = np.array([.964, .005, .005, .026])
    components_molar_mass = np.concatenate([molar_ch4, molar_n2, molar_co2, molar_c2h6])
    components_mass_proportions = calculate_mass_fraction_from_molar_fraction(
        components_molar_proportions,
        components_molar_mass)
    mix_dens = calculate_mixture_density(
        components_density, components_mass_proportions=components_mass_proportions)

    assert np.allclose(mix_dens, test_mix_density, rtol=0, atol=1e-10)


def test_mixture_heat_capacity_lgas():
    test_mix_capacity = pd.read_csv(os.path.join(pp_dir, 'properties', 'lgas', 'heat_capacity.txt'),
                                    header=None, sep=' ', comment='#').values[:, 1]

    cap_ch4 = pd.read_csv(os.path.join(pp_dir, 'properties', 'methane', 'heat_capacity.txt'),
                          header=None, sep=' ', comment='#').values
    cap_n2 = pd.read_csv(os.path.join(pp_dir, 'properties', 'nitrogen', 'heat_capacity.txt'),
                            header=None, sep=' ', comment='#').values
    cap_co2 = pd.read_csv(os.path.join(pp_dir, 'properties', 'carbondioxide', 'heat_capacity.txt'),
                          header=None, sep=' ', comment='#').values
    cap_c2h6 = pd.read_csv(os.path.join(pp_dir, 'properties', 'ethane', 'heat_capacity.txt'),
                           header=None, sep=' ', comment='#').values
    molar_ch4 = pd.read_csv(os.path.join(pp_dir, 'properties', 'methane', 'molar_mass.txt'),
                            header=None, sep=' ', comment='#').values[0]
    molar_n2 = pd.read_csv(os.path.join(pp_dir, 'properties', 'nitrogen', 'molar_mass.txt'),
                           header=None, sep=' ', comment='#').values[0]
    molar_co2 = pd.read_csv(os.path.join(pp_dir, 'properties', 'carbondioxide', 'molar_mass.txt'),
                            header=None, sep=' ', comment='#').values[0]
    molar_c2h6 = pd.read_csv(os.path.join(pp_dir, 'properties', 'ethane', 'molar_mass.txt'),
                             header=None, sep=' ', comment='#').values[0]
    values_to_take = [263, 273, 283, 293, 303, 313, 323, 333]
    cap_ch4 = interp1d(cap_ch4[:, 0], cap_ch4[:, 1])
    cap_ch4 = cap_ch4((values_to_take))
    cap_n2 = interp1d(cap_n2[:, 0], cap_n2[:, 1], fill_value='extrapolate')
    cap_n2 = cap_n2((values_to_take))
    cap_co2 = interp1d(cap_co2[:, 0], cap_co2[:, 1])
    cap_co2 = cap_co2((values_to_take))
    cap_c2h6 = interp1d(cap_c2h6[:, 0], cap_c2h6[:, 1])
    cap_c2h6 = cap_c2h6((values_to_take))
    components_capacity = np.concatenate([[cap_ch4], [cap_n2], [cap_co2], [cap_c2h6]])
    components_molar_proportions = np.array([.85, .103, .013, .034])
    components_molar_mass = np.concatenate([molar_ch4, molar_n2, molar_co2, molar_c2h6])
    components_mass_proportions = calculate_mass_fraction_from_molar_fraction(
        components_molar_proportions, components_molar_mass)
    mix_cap = calculate_mixture_heat_capacity(components_capacity, components_mass_proportions)

    assert np.allclose(mix_cap, test_mix_capacity, rtol=0, atol=1e-10)

    mix_cap = calculate_mixture_heat_capacity(components_capacity[:, 0],
                                              components_mass_proportions)

    assert np.allclose(mix_cap, test_mix_capacity[0], rtol=0, atol=1e-10)


def test_mixture_heat_capacity_hgas():
    test_mix_capacity = pd.read_csv(os.path.join(pp_dir, 'properties', 'hgas', 'heat_capacity.txt'),
                                    header=None, sep=' ', comment='#').values[:, 1]

    cap_ch4 = pd.read_csv(os.path.join(pp_dir, 'properties', 'methane', 'heat_capacity.txt'),
                          header=None, sep=' ', comment='#').values
    cap_n2 = pd.read_csv(os.path.join(pp_dir, 'properties', 'nitrogen', 'heat_capacity.txt'),
                            header=None, sep=' ', comment='#').values
    cap_co2 = pd.read_csv(os.path.join(pp_dir, 'properties', 'carbondioxide', 'heat_capacity.txt'),
                          header=None, sep=' ', comment='#').values
    cap_c2h6 = pd.read_csv(os.path.join(pp_dir, 'properties', 'ethane', 'heat_capacity.txt'),
                           header=None, sep=' ', comment='#').values
    molar_ch4 = pd.read_csv(os.path.join(pp_dir, 'properties', 'methane', 'molar_mass.txt'),
                            header=None, sep=' ', comment='#').values[0]
    molar_n2 = pd.read_csv(os.path.join(pp_dir, 'properties', 'nitrogen', 'molar_mass.txt'),
                           header=None, sep=' ', comment='#').values[0]
    molar_co2 = pd.read_csv(os.path.join(pp_dir, 'properties', 'carbondioxide', 'molar_mass.txt'),
                            header=None, sep=' ', comment='#').values[0]
    molar_c2h6 = pd.read_csv(os.path.join(pp_dir, 'properties', 'ethane', 'molar_mass.txt'),
                             header=None, sep=' ', comment='#').values[0]
    values_to_take = [263, 273, 283, 293, 303, 313, 323, 333]
    cap_ch4 = interp1d(cap_ch4[:, 0], cap_ch4[:, 1])
    cap_ch4 = cap_ch4((values_to_take))
    cap_n2 = interp1d(cap_n2[:, 0], cap_n2[:, 1], fill_value='extrapolate')
    cap_n2 = cap_n2((values_to_take))
    cap_co2 = interp1d(cap_co2[:, 0], cap_co2[:, 1])
    cap_co2 = cap_co2((values_to_take))
    cap_c2h6 = interp1d(cap_c2h6[:, 0], cap_c2h6[:, 1])
    cap_c2h6 = cap_c2h6((values_to_take))
    components_capacity = np.concatenate([[cap_ch4], [cap_n2], [cap_co2], [cap_c2h6]])
    components_molar_proportions = np.array([.964, .005, .005, .026])
    components_molar_mass = np.concatenate([molar_ch4, molar_n2, molar_co2, molar_c2h6])
    components_mass_proportions = calculate_mass_fraction_from_molar_fraction(
        components_molar_proportions, components_molar_mass)
    mix_cap = calculate_mixture_heat_capacity(components_capacity, components_mass_proportions)

    assert np.allclose(mix_cap, test_mix_capacity, rtol=0, atol=1e-10)


def test_mixture_molar_mass_lgas():
    test_mix_molar = pd.read_csv(os.path.join(pp_dir, 'properties', 'lgas', 'molar_mass.txt'),
                                 header=None, sep=' ', comment='#').values

    molar_ch4 = pd.read_csv(os.path.join(pp_dir, 'properties', 'methane', 'molar_mass.txt'),
                            header=None, sep=' ', comment='#').values[0]
    molar_n2 = pd.read_csv(os.path.join(pp_dir, 'properties', 'nitrogen', 'molar_mass.txt'),
                           header=None, sep=' ', comment='#').values[0]
    molar_co2 = pd.read_csv(os.path.join(pp_dir, 'properties', 'carbondioxide', 'molar_mass.txt'),
                            header=None, sep=' ', comment='#').values[0]
    molar_c2h6 = pd.read_csv(os.path.join(pp_dir, 'properties', 'ethane', 'molar_mass.txt'),
                             header=None, sep=' ', comment='#').values[0]
    components_molar_mass = np.concatenate([molar_ch4, molar_n2, molar_co2, molar_c2h6])
    components_molar_proportions = np.array([.85, .103, .013, .034])
    mix_molar = calculate_mixture_molar_mass(
        components_molar_mass=components_molar_mass,
        components_molar_proportions=components_molar_proportions)
    assert (mix_molar == test_mix_molar)

    components_mass_proportions = calculate_mass_fraction_from_molar_fraction(
        components_molar_proportions, components_molar_mass)
    mix_molar = calculate_mixture_molar_mass(
        components_molar_mass=components_molar_mass,
        components_mass_proportions=components_mass_proportions)
    assert (mix_molar == test_mix_molar)


def test_mixture_molar_mass_hgas():
    test_mix_molar = pd.read_csv(os.path.join(pp_dir, 'properties', 'hgas', 'molar_mass.txt'),
                                 header=None, sep=' ', comment='#').values

    molar_ch4 = pd.read_csv(os.path.join(pp_dir, 'properties', 'methane', 'molar_mass.txt'),
                            header=None, sep=' ', comment='#').values[0]
    molar_n2 = pd.read_csv(os.path.join(pp_dir, 'properties', 'nitrogen', 'molar_mass.txt'),
                           header=None, sep=' ', comment='#').values[0]
    molar_co2 = pd.read_csv(os.path.join(pp_dir, 'properties', 'carbondioxide', 'molar_mass.txt'),
                            header=None, sep=' ', comment='#').values[0]
    molar_c2h6 = pd.read_csv(os.path.join(pp_dir, 'properties', 'ethane', 'molar_mass.txt'),
                             header=None, sep=' ', comment='#').values[0]
    components_molar_mass = np.concatenate([molar_ch4, molar_n2, molar_co2, molar_c2h6])
    components_molar_proportions = np.array([.964, .005, .005, .026])
    mix_molar = calculate_mixture_molar_mass(
        components_molar_mass=components_molar_mass,
        components_molar_proportions=components_molar_proportions)
    assert (mix_molar == test_mix_molar)


def save_results(results, path):
    temp = pd.read_csv(os.path.join(pp_dir, 'properties', 'methane', 'viscosity.txt'),
                       header=None, sep=' ', comment='#')
    results = pd.DataFrame(results, index=temp[0].astype(int))
    results.to_csv(path, header=None, sep=' ')
