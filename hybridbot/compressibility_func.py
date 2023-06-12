import numpy as np
from scipy.optimize import newton


#ToDo: 1. Make function work; 2. Put Parameters in arrays; 3. Make function work for n_nodes>1(2-dimensional array)

"""
molar_mass_list = [net.fluid[fluid].get_molar_mass() for fluid in net._fluid]
temperature_list = get_temperature()
pressure_list = get_pressure()
"""

T = np.array([283.15, 283.15])
#T = 283.15
p = np.array([50*1e5, 50*1e5])
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
molar_fraction = np.array([[0.11, 0.89], [0.6, 0.4]])

shape = np.shape(molar_fraction)

def _calculate_A_B(a_p, a_T, a_molar_fraction,  a_p_crit, a_T_crit, a_acent_fact):
    """
    calculate initially for one node:

    a_molar_fraction: x-d_array
    p_crit : 1d-array
    T_crit : 1d-array
    a_acent_fact : 1d-array
    """

    # shape = np.shape(a_molar_fraction)
    # com_array = np.empty([shape[0], 3], dtype=np.float64)
    shape = np.shape(a_p)
    m_val = 0.480 + 1.574 * a_acent_fact - 0.176 * a_acent_fact ** 2
    m = m_val
    fluid_length = 2
    if isinstance(a_p, int)==True:
        T_red = a_T / a_T_crit
        p_red = a_p / a_p_crit

    else:
        T_red = np.zeros([len(a_p),len(a_p_crit)])
        p_red = np.zeros([len(a_p), len(a_p_crit)])



        for node in range(len(a_p)):
            T_red[node][0] = a_T[node] / a_T_crit[0]
            T_red[node][1] = a_T[node] / a_T_crit[1]
            p_red[node][0] = a_p[node] / a_p_crit[0]
            p_red[node][1] = a_p[node] / a_p_crit[1]
        for node in range(shape[0] - 1): m = np.vstack([m, m_val])






    sqrt_alpha = (1 + m * (1 - T_red ** 0.5))

    # def A_mixture(self, a_p, a_T, m, a_molar_fraction : np.ndarray, t_crit: np.ndarray, p_crit: np.ndarray):
    factor_a = 0.42747 * a_p / (a_T ** 2)

    if isinstance(a_p, int)==True:
        sum_a = (a_molar_fraction * a_T_crit * sqrt_alpha / a_p_crit ** 0.5).sum() ** 2
        A_mixture = factor_a * sum_a

        # B_mixture(self, a_p, a_T, a_molar_fraction):
        factor_b = 0.08664 * a_p / a_T
        sum_b = (a_molar_fraction * a_T_crit / a_p_crit).sum()
        B_mixture = factor_b * sum_b

        return A_mixture, B_mixture
    else:
        sum_a = (a_molar_fraction * a_T_crit * sqrt_alpha / a_p_crit ** 0.5).sum(axis=1) ** 2
        A_mixture = factor_a * sum_a

        # B_mixture(self, a_p, a_T, a_molar_fraction):
        factor_b = 0.08664 * a_p / a_T
        sum_b = (a_molar_fraction * a_T_crit / a_p_crit).sum(axis=1)
        B_mixture = factor_b * sum_b

        return A_mixture, B_mixture



def _func_of_Z(Z, a_p, a_T, a_mf):
    A_mixture, B_mixture = _calculate_A_B(a_p, a_T, a_mf, p_crit, T_crit, acent_fact)
    return Z ** 3 - Z ** 2 + Z *(A_mixture - B_mixture - B_mixture ** 2) - A_mixture * B_mixture

def _der_func_of_Z(Z, a_p, a_T, a_mf):
    A_mixture, B_mixture = _calculate_A_B(a_p, a_T,a_mf, p_crit, T_crit, acent_fact)
    return  3 * Z **2 - 2 * Z + (A_mixture - B_mixture - B_mixture **2)

def calculate_mixture_compressibility_draft(mf, p, T):
    shape = np.shape(mf)
    T = np.array([283.15, 283.15, 283.15])
    start_value = 0.9
    start_value_vec = np.array([list([start_value]) * shape[0]])[0]


    p_n = 101325
    T_n = 273.15

    res_comp = newton(_func_of_Z, start_value_vec, fprime=_der_func_of_Z, args=(p, T, mf))
    # ToDo:check args of res_comp_norm (mf?)
    res_comp_norm = newton(_func_of_Z, start_value_vec, fprime=_der_func_of_Z, args=(p_n, T_n, mf))
    res_comp / res_comp_norm
    return res_comp, res_comp_norm


#compr_mixture, compr_mixture_norm = calculate_mixture_compressibility_draft()
