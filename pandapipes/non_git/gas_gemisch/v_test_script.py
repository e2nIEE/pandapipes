import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
import matplotlib.pyplot as pyplo
import pandas as pd

R_methan = 448.2
lges = 12000
numsec= int(lges/1000)
Vdot = 45000/3600
k = 0.5
pbound = 51e5
ubound = 11.606

lambda_pipe = np.full(numsec, 0.02455)
d = np.full(numsec, 0.2091)
l = np.full(numsec,  lges/(numsec))

fac = np.zeros(numsec+1)
T = np.full(numsec+1, 285.15)
fac[:] = 1/(R_methan * T[:])

psol = np.zeros(numsec+1)
usol = np.zeros(numsec+1)
psol[:] = 30e5
psol[0] = pbound #bound
usol[:] = 3
usol[-1] = ubound #bound

lookup = np.zeros((numsec, 2)).astype(np.int32)
for i in range(numsec):
    lookup[i][0] = i
    lookup[i][1] = i+1

dFdpi = np.zeros(numsec)
dFdpi1 = np.zeros(numsec)
dFdui = np.zeros(numsec)
dFdui1 = np.zeros(numsec)

dGdpi = np.zeros(numsec)
dGdpi1 = np.zeros(numsec)
dGdui = np.zeros(numsec)
dGdui1 = np.zeros(numsec)

system_data = np.zeros(2*(numsec + 1) * 2*(numsec + 1))
system_rows = np.zeros(2*(numsec + 1) * 2*(numsec + 1))
system_cols = np.zeros(2*(numsec + 1) * 2*(numsec + 1))

i= 0
while i<=50:
    # Lastvektor
    F = np.zeros(len(psol) + len(usol))
    F[0] = -(psol[lookup[0][0]] - pbound)
    F[1] = -(usol[lookup[-1][1]] - ubound)

    F[2:2 + numsec] = -(
                usol[lookup[:,0]] * psol[lookup[:,1]] * fac[lookup[:,1]] - usol[lookup[:,0]] * psol[lookup[:,0]] *
                fac[lookup[:,0]] + usol[lookup[:,1]] * psol[lookup[:,0]] * fac[lookup[:,0]] - usol[lookup[:,0]] *
                psol[lookup[:,0]] * fac[lookup[:,0]])
    F[2 + numsec:] = -(
                psol[lookup[:,0]] * usol[lookup[:,0]] * usol[lookup[:,1]] * fac[lookup[:,0]] - psol[lookup[:,0]] *
                usol[lookup[:,0]] * usol[lookup[:,0]] * fac[lookup[:,0]] + psol[lookup[:,1]] - psol[
                    lookup[:,0]] + lambda_pipe[:] * l[:] * psol[lookup[:,0]] * usol[lookup[:,0]] * usol[
                    lookup[:,0]] * fac[lookup[:,0]] / (2 * d[:]))

    # Konti-Gl
    dFdpi[:] = -usol[lookup[:,0]]*fac[lookup[:,0]] + usol[lookup[:,1]]*fac[lookup[:,0]] - usol[lookup[:,0]]*fac[lookup[:,0]]
    dFdpi1[:] = usol[lookup[:,0]]*fac[lookup[:,1]]
    dFdui[:] = psol[lookup[:,1]]*fac[lookup[:,1]] - 2* psol[lookup[:,0]]*fac[lookup[:,0]]
    dFdui1[:] = psol[lookup[:,0]]*fac[lookup[:,0]]

    # Impuls
    dGdpi[:] = usol[lookup[:,0]] * usol[lookup[:,1]] * fac[lookup[:,0]] - usol[lookup[:,0]] * usol[lookup[:,0]] * fac[
        lookup[:,0]] - 1 + lambda_pipe[:] * l[:] * usol[lookup[:,0]] * usol[lookup[:,0]] * fac[lookup[:,0]] / (2 * d[:])
    dGdpi1[:] = 1
    dGdui[:] = psol[lookup[:,0]] * usol[lookup[:,1]] * fac[lookup[:,0]] - psol[lookup[:,0]] * usol[lookup[:,0]] * \
               fac[lookup[:,0]] + lambda_pipe[:] * l[:] * psol[lookup[:,0]] * usol[lookup[:,0]] * fac[
                   lookup[:,0]] / d[:]
    dGdui1[:] = psol[lookup[:,0]]*usol[lookup[:,0]]*fac[lookup[:,0]]

    #Boundary Left
    dF_l = np.zeros(2*(numsec+1))
    dF_l[0] = 1

    #Boundary Right
    dF_r = np.zeros(2*(numsec+1))
    dF_r[-1] = 1

    num_col = len(usol) + len(psol)
    #konti
    row_nums = np.arange(numsec)
    sub_kont = np.zeros((len(dFdpi),num_col))
    sub_kont[row_nums, [lookup[:,0]]] = dFdpi
    sub_kont[row_nums, [lookup[:,1]]] = dFdpi1
    sub_kont[row_nums, [len(psol)+lookup[:,0]]] = dFdui
    sub_kont[row_nums, [len(psol) + lookup[:,1]]] = dFdui1

    #imp
    sub_imp = np.zeros((len(dGdpi),num_col))
    sub_imp[row_nums, [lookup[:,0]]] = dGdpi
    sub_imp[row_nums, [lookup[:,1]]] = dGdpi1
    sub_imp[row_nums, [len(psol) + lookup[:,0]]] = dGdui
    sub_imp[row_nums, [len(psol) + lookup[:,1]]] = dGdui1

    system_matrix = csr_matrix((num_col, num_col ))
    system_matrix[0] = dF_l
    system_matrix[1] = dF_r
    system_matrix[2:2+numsec] = sub_kont
    system_matrix[2+numsec:] = sub_imp

    u_old = usol
    p_old = psol
    x = spsolve(system_matrix,F)
    psol += x[0:len(psol)]
    usol += x[len(psol):]
    i+=1

    print("F",F)
print("p", psol)
print("u", usol)

panda_ak = pd.read_csv("C:\\Users\\dcronbach\\pandapipes\\pandapipes\\non_git\\gas_gemisch\\gas_sections_an.csv", sep=";", squeeze=True,
                               header=0)

fig, axes = pyplo.subplots(2, 1, sharex='all')
axes[0].plot(psol, label="pandapipes_navier_stokes")
axes[0].plot(panda_ak.loc[:, "p1"], label="pandapipes_aktuell")
axes[0].set_title("Pressure [Pa]")
axes[0].grid()
axes[0].legend()
axes[1].plot(usol, label="pandapipes_navier_stokes")
axes[1].plot(panda_ak.loc[:, "v"], label="pandapipes_aktuell")
axes[1].set_title("Flow Velocity [m/s]")
axes[1].grid()

pyplo.show()


