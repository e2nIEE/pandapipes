import numpy as np
from scipy.optimize import newton
from pandapipes.constants import NORMAL_TEMPERATURE, NORMAL_PRESSURE, P_CONVERSION

p_n = NORMAL_PRESSURE * P_CONVERSION
T_n = NORMAL_TEMPERATURE

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
        nbr_fluid =  np.shape(a_mf)[0]
    else:
        nbr_node = np.shape(a_mf)[0]
        nbr_fluid = np.shape(a_mf)[1]

        a_T = np.tile(a_T, (nbr_fluid, 1)).T  # duplicate a_T horizontally and then transpose it
        a_p = np.tile(a_p, (nbr_fluid, 1)).T

        summation_axis = 1

    m = 0.480 + 1.574 * a_acent_fact - 0.176 * a_acent_fact ** 2

    T_red = a_T / a_T_crit

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
