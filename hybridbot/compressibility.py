import numpy as np
from scipy.optimize import newton


# h2
# critical temperature [K]
t_crit_h23 = 33.20
# critical pressure [Bar]
p_crit_h2=  13


# methane
# critical temperature [K]
t_crit_meth = 190.55
# critical pressure [Bar]
p_crit_meth = 45.95

abc = compressibility_SRK(T, p, molar_fractions)
abc.calculate

# Zum Testen'
molar_fractions = [0.11, 0.89]

# Ziel
molar_fractions_real = [[0.11, 0.89], [0.6, 0.4]]

class compressibility_SRK:

    def __init__(self, name):
        self.ac_omega = np.array([0.01142, -0.219]) #array
        self.p_crit = np.array([4599200.0, 1296400.0]) #array
        self.T_crit = np.array([190.564, 33.145]) #array


    def T_red(self, T):
        return T / self.T_crit

    def m_i(self):
        return 0.480 + 1.574 * self.ac_omega - 0.176 * self.ac_omega**2

    def alpha(self, T):
        return (1 + self.m_i() * ( 1 - self.T_red(T)**0.5))**2

    def a_mixture(self, p, T, m, molar_fractions : np.ndarray, t_crit: np.ndarray, p_crit: np.ndarray):
        factor = 0.42747 * p / (T ** 2)
        sum = sum(molar_fractions * self.T_crit * alpha(T) ** 0.5 / self.p_crit ** 0.5)**2
        return factor * sum

    def b_mixture(self, p, T, molar_fractions):
        factor = 0.08664 * p / T
        sum = sum(molar_fractions * self.T_crit / self.p_crit)
        return factor * sum

    def compressibility_factor(self, Z, p, T, molar_fractions):
        return Z**3 - Z**2 + Z * (a_mixture(p, T, molar_fractions) - b_mixture(p, T, molar_fractions) - b_mixture(p, T, molar_fractions)**2) - a_mixture(p, T, molar_fractions) * b_mixture(p, T, molar_fractions)

    def compressibilityfactor_der(self, Z, p, T, molar_fractions):
        return 3*Z**2 - 2*Z + (a_mixture(p, T, molar_fractions) - b_mixture(p, T, molar_fractions) - b_mixture(p, T, molar_fractions)**2)


    def calculate(self, Temp, press , molar_fractions):
        res_comp = newton(compressibility_factor, 0.9 ,fprime=compressibilityfactor_der, args=(press, Temp, molar_fractions,))
        res_comp_norm = newton(compressibility_factor, 0.9, fprime=compressibilityfactor_der, args=(p=101325., T=273.15, molar_fractions,))
        res = res_comp / res_comp_norm
        return res

    def return_results(self):
        return

abc = compressibility_SRK(name)
