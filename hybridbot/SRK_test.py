import numpy as np
from CoolProp.CoolProp import PropsSI
from thermo.eos_mix import SRKMIX

T = 283.15
p = 50*1e5
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
molar_fraction = np.array([[0.6, 0.4]])

#srk = SRKMIX(Tcs=T_crit, Pcs=p_crit, omegas=acent_fact, zs=molar_fraction, T=T, P=p).Z_g

propsi = PropsSI("Z", "T", T, "P", p, "SRK::Methane[0.6]&Hydrogen[0.4]")