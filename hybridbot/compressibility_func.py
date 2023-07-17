import numpy as np
from scipy.optimize import newton
from pandapipes.constants import NORMAL_TEMPERATURE, NORMAL_PRESSURE, P_CONVERSION

p_n = NORMAL_PRESSURE
T_n = NORMAL_TEMPERATURE


def _calculate_A_B(_p, _T, _mf,  _p_crit, _T_crit, _acent_fact):
    """
    Calculates the A-mixture and B-mixture of the Redlich-Kwong equation
    :param _p: pressure
    :type _p: array
    :param _T: temperature
    :type _T: array
    :param _mf: molar fractions of the gases
    :type _mf: array
    :param _p_crit: critical pressure of the gases
    :type _p_crit: array
    :param _T_crit: critical temperature of the gases
    :type _T_crit: array
    :param _acent_fact: acentric factor of the gases
    :type _acent_fact: array
    :return: A-mixture and B-mixture
    :rtype: array
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
    Extracts the critical data, calls calculate_A_B and returns the Redlich-Kwong equation
    :param Z:
    :type Z:
    :param _p:
    :type _p:
    :param _T:
    :type _T:
    :param _mf:
    :type _mf:
    :param critical_data:
    :type critical_data:
    :return:
    :rtype:
    """
    p_crit = np.array([item[1] for item in critical_data])
    T_crit = np.array([item[0] for item in critical_data])
    acent_fact = np.array([item[2] for item in critical_data])

    A_mixture, B_mixture = _calculate_A_B(_p, _T, _mf, p_crit, T_crit, acent_fact)
    return Z ** 3 - Z ** 2 + Z *(A_mixture - B_mixture - B_mixture ** 2) - A_mixture * B_mixture


def _der_func_of_Z(Z, _p, _T, _mf, critical_data):
    """

    :param Z:
    :type Z:
    :param _p:
    :type _p:
    :param _T:
    :type _T:
    :param _mf:
    :type _mf:
    :param critical_data:
    :type critical_data:
    :return:
    :rtype:
    """
    p_crit = np.array([item[1] for item in critical_data])
    T_crit = np.array([item[0] for item in critical_data])
    acent_fact = np.array([item[2] for item in critical_data])

    A_mixture, B_mixture = _calculate_A_B(_p, _T,_mf, p_crit, T_crit, acent_fact)
    return 3 * Z **2 - 2 * Z + (A_mixture - B_mixture - B_mixture **2)

def _second_der_func_of_Z(Z, _p, _T, _mf, critical_data):
    """

    :param Z:
    :type Z:
    :param _p:
    :type _p:
    :param _T:
    :type _T:
    :param _mf:
    :type _mf:
    :param critical_data:
    :type critical_data:
    :return:
    :rtype:
    """
    p_crit = np.array([item[1] for item in critical_data])
    T_crit = np.array([item[0] for item in critical_data])
    acent_fact = np.array([item[2] for item in critical_data])

    A_mixture, B_mixture = _calculate_A_B(_p, _T,_mf, p_crit, T_crit, acent_fact)
    return 6 * Z - 2

def calculate_mixture_compressibility_fact(_mf, _p, _T, critical_data):
    """

    :param _mf:
    :type _mf:
    :param _p:
    :type _p:
    :param _T:
    :type _T:
    :param critical_data:
    :type critical_data:
    :return:
    :rtype:
    """
    critical_data_list = [item[0] for item in critical_data]
    nbr_node = np.shape(_mf)[0]
    #Todo: if function ?
    start_value = 0.9
    start_value = np.array([list([start_value]) * nbr_node])[0]

    comp_fact = newton(_func_of_Z, start_value, fprime=_der_func_of_Z, args=(_p, _T, _mf, critical_data_list))
    comp_fact_norm = newton(_func_of_Z, start_value, fprime=_der_func_of_Z, args=(p_n, T_n, _mf, critical_data_list))
    return comp_fact, comp_fact_norm


def calculate_der_mixture_compressibility_fact(_mf, _p, _T, critical_data):
    """

    :param _mf:
    :type _mf:
    :param _p:
    :type _p:
    :param _T:
    :type _T:
    :param critical_data:
    :type critical_data:
    :return:
    :rtype:
    """
    critical_data_list = [item[0] for item in critical_data]
    nbr_node = np.shape(_mf)[0]
    #Todo: if function ?
    start_value = 0.9
    start_value = np.array([list([start_value]) * nbr_node])[0]

    der_comp_fact = newton(_der_func_of_Z, start_value, fprime=_second_der_func_of_Z, args=(_p, _T, _mf, critical_data_list))
    der_comp_fact_norm = newton(_der_func_of_Z, start_value, fprime=_second_der_func_of_Z, args=(p_n, T_n, _mf, critical_data_list))
    return der_comp_fact, der_comp_fact_norm

#compr_mixture, compr_mixture_norm = calculate_mixture_compressibility_draft(molar_fraction, p, T, critical_data)
