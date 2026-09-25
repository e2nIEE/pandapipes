# Copyright (c) 2020-2026 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from pandapipes.idx_branch import IdxBranch
from pandapipes.idx_node import IdxNode
from pandapipes.pf.system_index import HydraulicSystemIndex, HeatSystemIndex, ComponentRegistry
from pandapipes.pf.pipeflow_setup import (
    get_net_options, get_net_option, get_lookup, reduce_pit, create_internal_results,
    identify_active_nodes_branches, hydraulic_slack_mask, heat_transfer_slack_mask, set_net_option,
    check_infeed_number, compute_infeed_nodes, write_internal_results, PipeflowNotConverged
)
from pandapipes.pf.result_extraction import (
    extract_results_active_pit_hydraulics, extract_results_active_pit_heat_transfer
)
try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)



def execute_hydraulics(net):
    calc = HydraulicCalculation()
    calc.run(net)
    if net.converged:
        calc.on_converged(net)
        calc.rerun(net)
    if not net.converged:
        calc.handle_non_convergence()
    calc.extract_results(net)


def execute_heat(net):
    calc = ThermalCalculation()
    calc.run(net)
    if net.converged:
        calc.rerun(net)
    if not net.converged:
        calc.handle_non_convergence()
    calc.extract_results(net)


def execute_bidirectional(net):
    calc = BidirectionalCalculation()
    calc.run(net)
    if not net.converged:
        calc.handle_non_convergence()
    calc.extract_results(net)


class Calculation:
    """Base class for one Newton-Raphson nonlinear solve (hydraulics, heat transfer, bidirectional).

    A subclass declares, as class attributes, what used to be passed around as parallel
    lists (``solver_vars``/``tols``/``pit_names``/``iter_name``) and implements
    :meth:`solve_step` to perform one linearized assemble-and-solve pass.

    Newton-Raphson iteration recap:
      1. Build the Jacobian df/dx at the current guess x
      2. Solve J @ dx = -f(x) (here: spsolve)
      3. Update x -= dx * alpha, repeat until the residual/variable changes fall below tol

    Class attributes to set on subclasses
    --------------------------------------
    MODE             : str        identifies this calculation in logs/internal results (e.g. "hydraulics")
    ITER             : str        net option name holding the max-iteration count (e.g. "max_iter_hyd")
    VARS             : list[str]  names of the variables tracked for convergence (e.g. ["mdot", "p"])
    TOLS             : list[str]  net option names holding the tolerance for each VARS entry
    PITS             : list[str]  which pit ("branch"/"node") each VARS entry lives in
    COLS             : list[int]  PIT column index for each VARS entry (used for damping-fallback writes)
    """

    MODE = None
    ITER = None
    VARS = []
    TOLS = []
    PITS = []
    COLS = []

    def handle_non_convergence(self):
        raise PipeflowNotConverged("The calculation did not converge to a solution.")

    def prepare(self, net):
        """One-time setup run before the Newton-Raphson loop starts.

        E.g. connectivity identification, pit reduction. Default: no-op.
        """

    def on_converged(self, net):
        """Hook run once, immediately after a successful Newton-Raphson solve. Default: no-op."""

    def rerun(self, net):
        """Hook run after a successful solve to let components request a full rerun.

        E.g. a pressure control adjusting its target. Default: no-op.
        """

    def extract_results(self, net):
        """Write the converged results from "_active_pit" back into the general pit structure."""
        raise NotImplementedError

    def solve_step(self, net):
        """Perform one linearized solve (assemble Jacobian, spsolve, update pit values).

        :param net: the pandapipesNet to solve on
        :return: (results, residual, filtered) where results is a flat list of
                 [var1_new, var1_old, var2_new, var2_old, ...] (one pair per solver_var,
                 in the same order as ``solver_vars``), residual is the raw load-vector
                 residual, and filtered contains a row-index array (or None) per solver_var
                 selecting which pit rows that var's damping-fallback should write back to.
        """
        raise NotImplementedError

    def tols(self, net):
        return list(get_net_options(net, *self.TOLS))

    def run(self, net):
        """Run the Newton-Raphson loop until convergence or ITER's max iterations."""
        net.converged = False
        self.prepare(net)
        max_iter, nonlinear_method, tol_res = get_net_options(
            net, self.ITER, "nonlinear_method", "tol_res"
        )
        tols = self.tols(net)
        niter = 0
        errors = {var: [] for var in self.VARS}
        create_internal_results(net)
        residual_norm = None

        while not net.converged and niter < max_iter:
            logger.debug("niter %d", niter)
            results, residual, filtered = self.solve_step(net)
            residual_norm = np.max(np.abs(residual))
            logger.debug("residual: %s", residual_norm.round(4))

            results = np.array(results, object)
            pos = np.arange(len(self.VARS) * 2)
            vals_new = results[pos[::2]]
            vals_old = results[pos[1::2]]
            for var, val_new, val_old in zip(self.VARS, vals_new, vals_old):
                dval = val_new - val_old
                errors[var].append(np.max(np.abs(dval)) if len(dval) else 0)

            self._finalize_iteration(net, niter, residual_norm, nonlinear_method, errors, tols,
                                     tol_res, vals_old, filtered)
            niter += 1

        write_internal_results(net, **errors)
        kwargs = {
            f'residual_norm_{self.MODE}': residual_norm,
            f'iterations_{self.MODE}': niter,
        }
        write_internal_results(net, **kwargs)
        self._log_final_results(net, niter, residual_norm, tols)

    def _finalize_iteration(self, net, niter, residual_norm, nonlinear_method, errors, tols, tol_res,
                            vals_old, filtered):
        if nonlinear_method == "automatic":
            errors_increased = set_damping_factor(net, niter, errors)
            logger.debug("alpha: %s", get_net_option(net, "alpha"))
            for error_increased, val, pit, col, f in zip(
                errors_increased, vals_old, self.PITS, self.COLS, filtered
            ):
                if error_increased:
                    if f is None:
                        # todo: not working in bidirectional mode as bidirectional is not
                        #  distinguishing between hydraulics and heat transfer active pit
                        net["_active_pit"][pit][:, col] = val
                    else:
                        net["_active_pit"][pit][f, col] = val
            if get_net_option(net, "alpha") != 1:
                net.converged = False
                return
        elif nonlinear_method != "constant":
            logger.warning("No proper nonlinear method chosen. Using constant settings.")
        converged = True
        for var, error, tol in zip(self.VARS, errors.values(), tols):
            converged = error[niter] <= tol
            if not converged:
                break
            logger.debug("error_%s: %s", var, error[niter])
        net.converged = converged and residual_norm <= tol_res

    def _log_final_results(self, net, niter, residual_norm, tols):
        logger.debug("--------------------------------------------------------------------------------")
        if not net.converged:
            logger.debug(
                "Maximum number of iterations reached but %s solver did not converge.", self.MODE)
            logger.debug("Norm of residual: %s", residual_norm)
        else:
            logger.debug("Calculation completed. Preparing results...")
            logger.debug("Converged after %d iterations.", niter)
            logger.debug("Norm of residual: %s", residual_norm)
            for var, tol in zip(self.VARS, tols):
                logger.debug("tolerance for %s: %s", var, tol)


class HydraulicCalculation(Calculation):
    """Newton-Raphson solve for pressure/mdot (see :func:`solve_hydraulics`)."""

    MODE = 'hydraulics'
    ITER = 'max_iter_hyd'
    VARS = ['mdot', 'p', 'mdotslack']
    TOLS = ['tol_m', 'tol_p', 'tol_m']
    PITS = ['branch', 'node', 'node']
    COLS = [IdxBranch.MDOTINIT, IdxNode.PINIT, IdxNode.MDOTSLACKINIT]

    def handle_non_convergence(self):
        raise PipeflowNotConverged("The hydraulic calculation did not converge to a solution.")

    def prepare(self, net):
        net["_lookups"]["node_active_hydraulics"], net["_lookups"]["branch_active_hydraulics"] = \
            identify_active_nodes_branches(net, hydraulic_slack_mask(net))
        reduce_pit(net, "hydraulics")

    def solve_step(self, net):
        return solve_hydraulics(net)

    def rerun(self, net):
        rerun = False
        options = net["_options"]
        branch_pit = net["_active_pit"]["branch"]
        node_pit = net["_active_pit"]["node"]
        branch_lookups = get_lookup(net, "branch", "from_to_active_hydraulics")
        for comp in net['component_list']:
            rerun |= comp.rerun_hydraulics(net, branch_pit, node_pit, branch_lookups, options)
        if rerun:
            extract_results_active_pit_hydraulics(net)
            execute_hydraulics(net)

    def extract_results(self, net):
        extract_results_active_pit_hydraulics(net)


class ThermalCalculation(Calculation):
    """Newton-Raphson solve for branch outlet / node temperature (see :func:`solve_temperature`)."""

    MODE = 'heat'
    ITER = 'max_iter_therm'
    VARS = ['Tout', 'T']
    TOLS = ['tol_T', 'tol_T']
    PITS = ['branch', 'node']
    COLS = [IdxBranch.TOUTINIT, IdxNode.TINIT]

    def handle_non_convergence(self):
        raise PipeflowNotConverged("The heat transfer calculation did not converge to a solution.")

    def prepare(self, net):
        # heat transfer only makes physical sense on branches that are also hydraulically active
        # (a temperature slack always needs a reachable pressure slack to actually move the fluid) -
        # narrow down the hydraulic connectivity further via the heat-transfer slacks. If hydraulics
        # hasn't run yet in this pipeflow() call (standalone mode='heat'), compute it once here.
        if "node_active_hydraulics" not in net["_lookups"]:
            net["_lookups"]["node_active_hydraulics"], net["_lookups"]["branch_active_hydraulics"] = \
                identify_active_nodes_branches(net, hydraulic_slack_mask(net))
        nodes_hyd = net["_lookups"]["node_active_hydraulics"]
        branches_hyd = net["_lookups"]["branch_active_hydraulics"]

        net["_lookups"]["node_active_heat_transfer"], net["_lookups"]["branch_active_heat_transfer"] = \
            identify_active_nodes_branches(net, heat_transfer_slack_mask(net), nodes_hyd, branches_hyd)
        reduce_pit(net, "heat_transfer")

    def solve_step(self, net):
        return solve_temperature(net)

    def rerun(self, net):
        rerun = False
        options = net["_options"]
        branch_pit = net["_active_pit"]["branch"]
        node_pit = net["_active_pit"]["node"]
        branch_lookups = get_lookup(net, "branch", "from_to_active_heat_transfer")
        for comp in net['component_list']:
            rerun |= comp.rerun_hydraulics(net, branch_pit, node_pit, branch_lookups, options)
        if rerun:
            extract_results_active_pit_heat_transfer(net)
            execute_heat(net)

    def extract_results(self, net):
        extract_results_active_pit_heat_transfer(net)


class BidirectionalCalculation(Calculation):
    """Newton-Raphson solve alternating hydraulics and heat transfer (see :func:`solve_bidirectional`)."""

    MODE = 'bidirectional'
    ITER = 'max_iter_bidirect'
    # solve_bidirectional() concatenates solve_hydraulics()'s 3 pairs (mdot, p, mdotslack) with
    # solve_temperature()'s 2 pairs (Tout, T) - VARS/TOLS/PITS/COLS/filtered must list all 5 in
    # that same order (this is exactly HydraulicCalculation's VARS/TOLS/PITS/COLS followed by
    # ThermalCalculation's), or Calculation.run()'s positional un-interleaving
    # (results[0::2]/results[1::2]) pairs each value array with the wrong variable name/pit/col -
    # a previous version listed only 4 entries (['mdot', 'p', 'TOUT', 'T']), which silently
    # shifted every entry from 'mdotslack' onward: 'TOUT' was actually paired with mdotslack's
    # values/branch pit/TOUTINIT col, 'T' was paired with Tout's values, and the real T pair was
    # dropped entirely (never damped, never convergence-checked). With
    # nonlinear_method="automatic", the mismatched (branch pit, slack_nodes) combination for the
    # 'TOUT' slot could then write mdotslack's node-indexed damping-fallback values into the
    # branch pit at those same (node-range) row indices, raising IndexError once a slack node's
    # index exceeded the branch pit's row count.
    VARS = ['mdot', 'p', 'mdotslack', 'Tout', 'T']
    TOLS = ['tol_m', 'tol_p', 'tol_m', 'tol_T', 'tol_T']
    PITS = ['branch', 'node', 'node', 'branch', 'node']
    COLS = [IdxBranch.MDOTINIT, IdxNode.PINIT, IdxNode.MDOTSLACKINIT, IdxBranch.TOUTINIT,
            IdxNode.TINIT]

    def handle_non_convergence(self):
        raise PipeflowNotConverged("The bidrectional calculation did not converge to a solution.")

    def prepare(self, net):
        # hydraulics and heat transfer each get their own, independent connectivity check, done
        # once here rather than (for heat) recomputed on every solve_bidirectional() iteration
        net["_lookups"]["node_active_hydraulics"], net["_lookups"]["branch_active_hydraulics"] = \
            identify_active_nodes_branches(net, hydraulic_slack_mask(net))
        net["_lookups"]["node_active_heat_transfer"], net["_lookups"]["branch_active_heat_transfer"] = \
            identify_active_nodes_branches(net, heat_transfer_slack_mask(net))

    def solve_step(self, net):
        return solve_bidirectional(net)

    def extract_results(self, net):
        pass  # solve_bidirectional() already extracts both results every iteration

def solve_bidirectional(net):
    reduce_pit(net, "hydraulics")
    res_hyd, residual_hyd, filter_hyd = solve_hydraulics(net)
    extract_results_active_pit_hydraulics(net)

    reduce_pit(net, "heat_transfer")
    res_heat, residual_heat, filter_heat = solve_temperature(net)
    extract_results_active_pit_heat_transfer(net)

    residual = np.concatenate([residual_hyd, residual_heat])
    res = res_hyd + res_heat
    filtered = filter_hyd + filter_heat
    return res, residual, filtered

def solve_hydraulics(net):
    """Create and solve the linearized system of equations to calculate hydraulic magnitudes.

    Builds a jacobian (scipy sparse matrix) and load vector (numpy array) to calculate
    pressure and velocity for the network nodes and branches.

    :param net: The pandapipesNet for which to solve the hydraulic matrix
    :type net: pandapipesNet
    :return: (results, residual, filtered) - see Calculation.solve_step for the exact shape
    """
    options = net["_options"]

    connected_restarted = True
    while connected_restarted:
        branch_pit = net["_active_pit"]["branch"]
        node_pit = net["_active_pit"]["node"]
        connected_restarted = _restart_connectivity_check(net)

    sys_idx = HydraulicSystemIndex(node_pit, branch_pit)
    eq_registry = ComponentRegistry()

    for comp in net['component_list']:
        comp.register_hydraulic_equations(net, branch_pit, node_pit, sys_idx, eq_registry)

    sz = sys_idx.size()
    rows, cols, data, epsilon = eq_registry.assemble(sz)
    jacobian = csr_matrix((data, (rows, cols)), shape=(sz, sz))

    m_init_old = branch_pit[:, IdxBranch.MDOTINIT].copy()
    p_init_old = node_pit[:, IdxNode.PINIT].copy()
    slack_nodes = np.where(node_pit[:, IdxNode.NODE_TYPE] == IdxNode.P)[0]
    msl_init_old = node_pit[slack_nodes, IdxNode.MDOTSLACKINIT].copy()

    x = spsolve(jacobian, epsilon)

    branch_pit[:, IdxBranch.MDOTINIT] -= x[len(node_pit):len(node_pit) + len(branch_pit)] * options["alpha"]
    node_pit[:, IdxNode.PINIT] -= x[:len(node_pit)] * options["alpha"]
    node_pit[slack_nodes, IdxNode.MDOTSLACKINIT] -= x[len(node_pit) + len(branch_pit):]

    filtered = [None, None, slack_nodes]

    return [branch_pit[:, IdxBranch.MDOTINIT], m_init_old, node_pit[:, IdxNode.PINIT], p_init_old,
            node_pit[slack_nodes, IdxNode.MDOTSLACKINIT], msl_init_old], epsilon, filtered

def solve_temperature(net):
    """Build and solve a linearized system of equations to calculate temperature values.

    Uses the underlying net and the necessary graph data structures. Returned are the
    solution vectors for the new iteration, the original solution vectors and a vector
    containing component indices for the system matrix entries.

    :param net: The pandapipesNet for which to solve the temperature matrix
    :type net: pandapipesNet
    :return: branch_pit
    """
    options = net["_options"]
    branch_pit = net["_active_pit"]["branch"]
    node_pit = net["_active_pit"]["node"]

    # Negative velocity values are turned to positive ones (including exchange of from_node and
    # to_node for temperature calculation
    branch_pit[:, IdxBranch.FROM_NODE_T_SWITCHED] = branch_pit[:, IdxBranch.MDOTINIT] < -2e-11

    node_pit[:, IdxNode.INFEED] = False
    compute_infeed_nodes(branch_pit, node_pit)

    sys_idx = HeatSystemIndex(node_pit, branch_pit)
    eq_registry = ComponentRegistry()

    for comp in net['component_list']:
        comp.register_thermal_equations(net, branch_pit, node_pit, sys_idx, eq_registry)

    t_init_old = node_pit[:, IdxNode.TINIT].copy()
    t_out_old = branch_pit[:, IdxBranch.TOUTINIT].copy()
    filtered = [None, None]
    if not check_infeed_number(node_pit):
        return [branch_pit[:, IdxBranch.TOUTINIT], t_out_old, node_pit[:, IdxNode.TINIT], t_init_old], np.array([
            np.nan]), filtered

    sz = sys_idx.size()
    rows, cols, data, epsilon = eq_registry.assemble(sz)
    jacobian = csr_matrix((data, (rows, cols)), shape=(sz, sz))

    x = spsolve(jacobian, epsilon)

    if np.any(np.isnan(x)):
        return [branch_pit[:, IdxBranch.TOUTINIT], t_out_old, node_pit[:, IdxNode.TINIT], t_init_old], np.array([
            np.nan]), filtered

    node_pit[:, IdxNode.TINIT] -= x[:len(node_pit)] * options["alpha"]
    branch_pit[:, IdxBranch.TOUTINIT] -= x[len(node_pit):] * options["alpha"]

    return [branch_pit[:, IdxBranch.TOUTINIT], t_out_old, node_pit[:, IdxNode.TINIT], t_init_old], epsilon, filtered


def set_damping_factor(net, niter, errors):
    """Set the value of the damping factor (factor for the newton step width) from current results.

    :param net: the net for which to perform the pipeflow
    :type net: pandapipesNet
    :param niter:
    :type niter:
    :param errors: an array containing the current residuals of all field variables solved for
    :return: No Output.

    Example
    -------
        set_damping_factor(net, niter, [error_p, error_v])

    """
    error_increased = []
    for error in errors.values():
        error_increased.append(error[niter] > error[niter - 1])
    current_alpha = get_net_option(net, "alpha")
    if np.all(error_increased):
        set_net_option(net, "alpha", current_alpha / 10 if current_alpha >= 0.1 else current_alpha)
    else:
        set_net_option(net, "alpha", current_alpha * 10 if current_alpha <= 0.1 else 1.0)
    return error_increased


def _restart_connectivity_check(net):
    nodes_connected = get_lookup(net, "node", "active_hydraulics")
    branches_connected = get_lookup(net, "branch", "active_hydraulics")
    rows_nodes = np.arange(net["_pit"]["node"].shape[0])[nodes_connected]
    rows_branches = np.arange(net["_pit"]["branch"].shape[0])[branches_connected]
    active_node_pit = net["_active_pit"]["node"]
    active_branch_pit = net["_active_pit"]["branch"]
    node_pit = net["_pit"]["node"][rows_nodes, IdxNode.ACTIVE]
    branch_pit = net["_pit"]["branch"][rows_branches, IdxBranch.ACTIVE]
    mask_diff_node = active_node_pit[:, IdxNode.ACTIVE] != node_pit
    mask_diff_branch = active_branch_pit[:, IdxBranch.ACTIVE]  != branch_pit
    if np.any(mask_diff_node) | np.any(mask_diff_branch):
        net["_pit"]["node"][rows_nodes, IdxNode.ACTIVE] = active_node_pit[:, IdxNode.ACTIVE]
        net["_pit"]["node"][rows_nodes, IdxNode.NODE_TYPE] = active_node_pit[:, IdxNode.NODE_TYPE]
        net["_pit"]["branch"][rows_branches, IdxBranch.ACTIVE] = active_branch_pit[:, IdxBranch.ACTIVE]
        net["_pit"]["branch"][rows_branches, IdxBranch.BRANCH_TYPE] = active_branch_pit[:, IdxBranch.BRANCH_TYPE]
        net["_lookups"]["node_active_hydraulics"], net["_lookups"]["branch_active_hydraulics"] = \
            identify_active_nodes_branches(net, hydraulic_slack_mask(net))
        reduce_pit(net, "hydraulics")
        return True
    return False
