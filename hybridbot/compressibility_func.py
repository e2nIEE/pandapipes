import numpy as np
from scipy.optimize import newton
from pandapipes.constants import NORMAL_TEMPERATURE, NORMAL_PRESSURE, P_CONVERSION

p_n = NORMAL_PRESSURE * P_CONVERSION
T_n = NORMAL_TEMPERATURE


def _calculate_A_B(_p, _T, _mf,  _p_crit, _T_crit, _acent_fact):
    """
    calculate initially for one node:

    _mf: molar fractions : (n * f)-d_array. n : number of nodes, f number of fluids.
    p_crit : 1d-array
    T_crit : 1d-array
    _acent_fact : 1d-array
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

    sum_a = (_mf * _T_crit * sqrt_alpha / _p_in_pa ** 0.5).sum(axis=summation_axis) ** 2
    A_mixture = factor_a * sum_a

    sum_b = (_mf * _T_crit / _p_in_pa).sum(axis=summation_axis)
    B_mixture = factor_b * sum_b

    return A_mixture, B_mixture


def _func_of_Z(Z, _p, _T, _mf, critical_data):

    p_crit = np.array([item[1] for item in critical_data])
    T_crit = np.array([item[0] for item in critical_data])
    acent_fact = np.array([item[2] for item in critical_data])

    A_mixture, B_mixture = _calculate_A_B(_p, _T, _mf, p_crit, T_crit, acent_fact)
    return Z ** 3 - Z ** 2 + Z *(A_mixture - B_mixture - B_mixture ** 2) - A_mixture * B_mixture


def _der_func_of_Z(Z, _p, _T, _mf, critical_data):

    p_crit = np.array([item[1] for item in critical_data])
    T_crit = np.array([item[0] for item in critical_data])
    acent_fact = np.array([item[2] for item in critical_data])

    A_mixture, B_mixture = _calculate_A_B(_p, _T,_mf, p_crit, T_crit, acent_fact)
    return  3 * Z **2 - 2 * Z + (A_mixture - B_mixture - B_mixture **2)


def calculate_mixture_compressibility_draft(_mf, _p, _T, critical_data):
    critical_data_list = [item[0] for item in critical_data]
    nbr_node = np.shape(_mf)[0]
    #Todo: if function ?
    start_value = 0.9
    start_value = np.array([list([start_value]) * nbr_node])[0]

    res_comp = newton(_func_of_Z, start_value, fprime=_der_func_of_Z, args=(_p, _T, _mf, critical_data_list))
    res_comp_norm = newton(_func_of_Z, start_value, fprime=_der_func_of_Z, args=(p_n, T_n, _mf, critical_data_list))
    res_comp / res_comp_norm
    return res_comp, res_comp_norm


#compr_mixture, compr_mixture_norm = calculate_mixture_compressibility_draft(molar_fraction, p, T, critical_data)
