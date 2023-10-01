# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

def idx_branch(net):
    # branch types
    # no types defined

    # branch indices
    idx_branch = dict()
    idx_branch['TABLE_IDX'] = 0  # number of the table that this branch belongs to
    idx_branch['ELEMENT_IDX'] = 1  # index of the element that this branch belongs to (within the given table)
    idx_branch['FROM_NODE'] = 2  # f, from bus number
    idx_branch['TO_NODE'] = 3  # t, to bus number
    idx_branch['ACTIVE'] = 4
    idx_branch['LENGTH'] = 5  # Pipe length in [m]
    idx_branch['D'] = 6  # Diameter in [m]
    idx_branch['AREA'] = 7  # Area in [m²]
    idx_branch['K'] = 8  # Pipe roughness in [m]
    idx_branch['TINIT'] = 9  # Temperature in [K]
    idx_branch['VINIT'] = 10  # velocity in  [m/s]
    idx_branch['RE'] = 11  # Reynolds number
    idx_branch['LAMBDA'] = 12  # Lambda
    idx_branch['JAC_DERIV_DV'] = 13  # Slot for the derivative by velocity
    idx_branch['JAC_DERIV_DP'] = 14  # Slot for the derivative by pressure from_node
    idx_branch['JAC_DERIV_DP1'] = 15  # Slot for the derivative by pressure to_node
    idx_branch['LOAD_VEC_BRANCHES'] = 16  # Slot for the load vector for the branches
    idx_branch['JAC_DERIV_DV_NODE'] = 17  # Slot for the derivative by velocity for the nodes connected to branch
    idx_branch['LOAD_VEC_NODES'] = 18  # Slot for the load vector of the nodes connected to branch
    idx_branch['LOSS_COEFFICIENT'] = 19
    idx_branch['CP'] = 20  # Slot for fluid heat capacity values
    idx_branch['ALPHA'] = 21  # Slot for heat transfer coefficient
    idx_branch['JAC_DERIV_DT'] = 22
    idx_branch['JAC_DERIV_DT1'] = 23
    idx_branch['LOAD_VEC_BRANCHES_T'] = 24
    idx_branch['T_OUT'] = 25  # Internal slot for outlet pipe temperature
    idx_branch['JAC_DERIV_DT_NODE'] = 26  # Slot for the derivative fpr T for the nodes connected to branch
    idx_branch['LOAD_VEC_NODES_T'] = 27
    idx_branch['VINIT_T'] = 28
    idx_branch['FROM_NODE_T'] = 29
    idx_branch['TO_NODE_T'] = 30
    idx_branch['QEXT'] = 31  # heat input in [W]
    idx_branch['TEXT'] = 32
    idx_branch['PL'] = 33
    idx_branch['TL'] = 34  # Temperature lift [K]
    idx_branch['BRANCH_TYPE'] = 35  # branch type relevant for the pressure controller

    idx_branch['RHO'] = 36  # Density in [kg/m^3
    idx_branch['ETA'] = 37  # Dynamic viscosity in [Pas]
    idx_branch['V_FROM_NODE'] = 38  # node the fluid is coming from
    idx_branch['V_TO_NODE'] = 39  # node the fluid is going to

    counter = 39

    for key in net._fluid:
        counter += 1
        idx_branch[key + '_W'] = counter
        counter += 1
        idx_branch[key + '_RHO'] = counter
        counter += 1
        idx_branch[key + '_JAC_DERIV_RHO_SAME_W'] = counter
        counter += 1
        idx_branch[key + '_JAC_DERIV_RHO_DIFF_W'] = counter

    idx_branch['branch_cols'] = counter + 1

    net['_idx_branch'] = idx_branch
