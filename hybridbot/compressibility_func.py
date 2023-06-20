import numpy as np
from scipy.optimize import newton


#ToDo: 1. Make function work; 2. Put Parameters in arrays; 3. Make function work for n_nodes>1(2-dimensional array)

"""
molar_mass_list = [net.fluid[fluid].get_molar_mass() for fluid in net._fluid]
temperature_list = get_temperature()
pressure_list = get_pressure()
"""

T = np.array([283.15, 283.15, 283.15])
#T = 283.15
p = np.array([1*1e5, 25*1e5, 50*1e5])
#p = 45*1e5
p_n = 101325
T_n = 273.15
# methane
# critical temperature [K]
T_crit_meth = 190.55
# critical pressure [Pa]
p_crit_meth = 45.95e5
# acentric factor - dimensionless
acent_fact_meth = 0.01142

# h2
# critical temperature [K]
T_crit_h2 = 33.20
# critical pressure [Pa]
p_crit_h2 = 13e5
# acentric factor - dimensionless
acent_fact_h2 = -0.219

T_crit = np.array([T_crit_meth, T_crit_h2])
p_crit = np.array([p_crit_meth, p_crit_h2])
acent_fact = np.array([acent_fact_meth, acent_fact_h2])

# Zum Testen'
#molar_fraction = np.array([0.6, 0.4])

# Ziel
molar_fraction = np.array([[0.11, 0.89], [0.6, 0.4], [.9, .1]])


def _calculate_A_B(a_p, a_T, a_mf,  a_p_crit, a_T_crit, a_acent_fact):
    """
    calculate initially for one node:

    a_mf: molar fractions : (n * f)-d_array. n : number of nodes, f number of fluids.
    p_crit : 1d-array
    T_crit : 1d-array
    a_acent_fact : 1d-array
    """

    summation_axis = 0

    factor_a = 0.42747 * a_p / (a_T ** 2)
    factor_b = 0.08664 * a_p / a_T

    if a_mf.ndim == 1:
        nbr_fluid =  np.shape(molar_fraction)[0]
    else:
        nbr_node = np.shape(molar_fraction)[0]
        nbr_fluid = np.shape(molar_fraction)[1]

        a_T = np.tile(a_T, (nbr_fluid, 1)).T  # duplicate a_T horizontally and then transpose it
        a_p = np.tile(a_p, (nbr_fluid, 1)).T

        summation_axis = 1

    m = 0.480 + 1.574 * a_acent_fact - 0.176 * a_acent_fact ** 2

    T_red = a_T / T_crit
    p_red = a_p / p_crit

    sqrt_alpha = (1 + m * (1 - T_red ** 0.5))

    sum_a = (a_mf * a_T_crit * sqrt_alpha / a_p_crit ** 0.5).sum(axis=summation_axis) ** 2
    A_mixture = factor_a * sum_a

    sum_b = (a_mf * a_T_crit / a_p_crit).sum(axis=summation_axis)
    B_mixture = factor_b * sum_b

    return A_mixture, B_mixture


def _func_of_Z(Z, a_p, a_T, a_mf, critical_data):
    p_crit = np.array([item[1] for item in critical_data])
    T_crit = np.array([item[0] for item in critical_data])
    acent_fact = np.array([item[2] for item in critical_data])
    A_mixture, B_mixture = _calculate_A_B(a_p, a_T, a_mf, p_crit, T_crit, acent_fact)
    return Z ** 3 - Z ** 2 + Z *(A_mixture - B_mixture - B_mixture ** 2) - A_mixture * B_mixture


def _der_func_of_Z(Z, a_p, a_T, a_mf, critical_data):
    p_crit = np.array([item[1] for item in critical_data])
    T_crit = np.array([item[0] for item in critical_data])
    acent_fact = np.array([item[2] for item in critical_data])
    A_mixture, B_mixture = _calculate_A_B(a_p, a_T,a_mf, p_crit, T_crit, acent_fact)
    return  3 * Z **2 - 2 * Z + (A_mixture - B_mixture - B_mixture **2)


def calculate_mixture_compressibility_draft(a_mf, a_p, a_T, critical_data):
    critical_data_list = [item[0] for item in critical_data]
    nbr_node = np.shape(a_mf)[0]
    #Todo: if function ?
    start_value = 0.9
    start_value = np.array([list([start_value]) * nbr_node])[0]

    res_comp = newton(_func_of_Z, start_value, fprime=_der_func_of_Z, args=(a_p, a_T, a_mf, critical_data_list))
    res_comp_norm = newton(_func_of_Z, start_value, fprime=_der_func_of_Z, args=(p_n, T_n, a_mf, critical_data_list))
    res_comp / res_comp_norm
    return res_comp, res_comp_norm


#compr_mixture, compr_mixture_norm = calculate_mixture_compressibility_draft(molar_fraction, p, T, critical_data)
