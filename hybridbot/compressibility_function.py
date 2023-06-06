import numpy as np
from scipy.optimize import newton


def initial_parameters(name):
    ac_omega = np.array([0.01142, -0.219])  # array
    p_crit = np.array([4599200.0, 1296400.0])  # array
    T_crit = np.array([190.564, 33.145])  # array
    return ac_omega, p_crit, T_crit

def T_red(T, T_crit):
    return T / T_crit


def m_i(ac_omega):
    return 0.480 + 1.574 * ac_omega - 0.176 * ac_omega ** 2


def alpha(m_i, T_red, T):
    return (1 + m_i() * (1 - T_red(T) ** 0.5)) ** 2


def a_mixture(T_crit, p, T, alpha, molar_fractions: np.ndarray, p_crit: np.ndarray):
    factor = 0.42747 * p / (T ** 2)
    sum = sum(molar_fractions * T_crit * alpha(m_i, T_red, T) ** 0.5 / p_crit ** 0.5) ** 2
    return factor * sum


def b_mixture(T_crit, p_crit, p, T, molar_fractions):
    factor = 0.08664 * p / T
    sum = sum(molar_fractions * T_crit / p_crit)
    return factor * sum


def compressibility_factor(Z, p, T, molar_fractions, ac_omega, p_crit, T_crit):
    return Z ** 3 - Z ** 2 + \
           Z * (a_mixture(T_crit, p, T, alpha, molar_fractions, p_crit) - b_mixture(T_crit, p_crit, p, T, molar_fractions) - b_mixture(T_crit, p_crit, p, T, molar_fractions) ** 2)\
           - a_mixture(T_crit, p, T, alpha, molar_fractions, p_crit) * b_mixture(T_crit, p_crit, p, T, molar_fractions)


def compressibilityfactor_der(Z, p, T, molar_fractions, T_crit, p_crit):
    return 3 * Z ** 2 - 2 * Z + (a_mixture(T_crit, p, T, alpha, molar_fractions, p_crit) - b_mixture(T_crit, p_crit, p, T, molar_fractions) - b_mixture(T_crit, p_crit, p, T, molar_fractions) ** 2)


def calculate(Temp, press, molar_fractions, ac_omega, p_crit, T_crit):
    res_comp = newton(compressibility_factor(array[i]), 0.9, fprime=compressibilityfactor_der,
                      args=(press, Temp, molar_fractions,))
    res_comp_norm = newton(compressibility_factor(Z, press, Temp, molar_fractions, ac_omega, p_crit, T_crit), 0.9, fprime=compressibilityfactor_der,
                           args=(press, Temp, molar_fractions,))
    res = res_comp / res_comp_norm
    return res


def return_results():
    return


if __name__ == "__main__":
    ac_omega, p_crit, T_crit = initial_parameters(name="hallo")
    Temp = 320
    press = 100000
    molar_fractions = [0.7, 0.3]
    calculate(Temp, press, molar_fractions, ac_omega, p_crit, T_crit)