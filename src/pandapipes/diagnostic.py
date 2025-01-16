# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import pandapipes as pp
import numpy as np

from pandapipes import PipeflowNotConverged

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def check_net(net, low_length_limit_km=0.01, check_scaling_factor=1e-5):
    """
    Run some diagnostic checks on the net to identify potential flaws.
    """
    net = net.deepcopy()  # do not modify the direct input
    try:
        pp.pipeflow(net)
        if net.converged:
            logger.info("The initial, unmodified pipeflow converges.")
        else:
            logger.warning("The initial, unmodified pipeflow does NOT converge.")
    except Exception as e:
        logger.info(f"The initial, unmodified pipeflow does NOT converge.\n"
                    f"\t\tThis exception is raised:\n\t\t{e}")

    # check ext_grid
    if net.fluid.is_gas & (not hasattr(net, "ext_grid") | net.ext_grid.empty):
        logger.warning("The net does not have an external grid! "
                       "An external grid is required for gas networks.")

    # check zero / low length
    zl = net.pipe.loc[net.pipe.length_km == 0]
    ll = net.pipe.loc[net.pipe.length_km <= low_length_limit_km]
    if not zl.empty:
        logger.warning(f"{len(zl.index)} pipes have a length of 0.0 km. (IDs: {zl.index})")
    if not ll.empty:
        logger.warning(f"{len(ll.index)} pipes have a length below"
                       f" {low_length_limit_km} km. "
                       f"This could lead to convergence issues. The lowest length in the net is "
                       f"{ll.length_km.min()} km.\n"
                       f"(IDs of pipelines with low length: {ll.index})")

    net2 = net.deepcopy()
    net2.pipe.loc[net2.pipe.length_km < low_length_limit_km].length_km = low_length_limit_km
    try:
        pp.pipeflow(net2)
        if net2.converged:
            logger.info(f"If all short pipelines (< {low_length_limit_km} km) were set to "
                        f"{low_length_limit_km} km, the pipeflow would converge.")
        else:
            logger.warning(f"If all short pipelines (< {low_length_limit_km} km) were set to "
                        f"{low_length_limit_km} km, the pipeflow would still NOT converge.")
    except Exception as e:
        logger.info(f"Pipeflow does not converge, even if all short pipelines (< 10 m) were set "
                    f"to {low_length_limit_km} km. \n"
                    f"\t\tThe error message is: {e}")

    # check iterations
    iterations = 200
    try:
        pp.pipeflow(net, iter=iterations)
        logger.info(f"The pipeflow converges after {net._internal_results['iterations']:d} "
                    f"iterations.")
    except PipeflowNotConverged:
        logger.info(f"After {iterations:d} iterations the pipeflow did NOT converge.")

    # check with little sink and source scaling
    logger.info("Testing with scaled-down sinks and sources.")
    net3 = net.deepcopy()
    if hasattr(net, "sink"):
        net3.sink.scaling *= check_scaling_factor
    if hasattr(net, "source"):
        net3.source.scaling *= check_scaling_factor
    try:
        pp.pipeflow(net3)
        if net3.converged:
            logger.info(f"If sinks and sources were scaled with a factor of to "
                        f"{check_scaling_factor}, the pipeflow would converge.")
        else:
            logger.warning(f"If sinks and sources were scaled with a factor of to "
                           f"{check_scaling_factor}, the pipeflow would still NOT converge.")
    except Exception as e:
        logger.info(f"Pipeflow does not converge with sinks/sources scaled by"
                    f" {check_scaling_factor}.\n"
                    f"\t\tThe error message is: {e}")

    # check k
    if any(net.pipe.k_mm > 0.5):
        logger.warning(f"Some pipes have a friction factor k_mm > 0.5 (extremely rough). The "
                       f"highest value in the net is {net.pipe.k_mm.max()}. Up to "
                       f"0.2 mm is a common value for old steel pipes."
                       f"\nRough pipes: {net.pipe.loc[net.pipe.k_mm > 0.5]}.")
    net4 = net.deepcopy()
    net4.pipe.k_mm = 1e-5
    try:
        pp.pipeflow(net4)
        if net4.converged:
            logger.info(f"If the friction factor would be reduced to 1e-5 for all pipes, "
                        f"the pipeflow would converge.")
        else:
            logger.warning(f"If the friction factor would be reduced to 1e-5 for all pipes, "
                           f"the pipeflow would still NOT converge.")
    except Exception as e:
        logger.info(f"Pipeflow does not converge with k_mm = 1-e5 for all pipes.\n"
                    f"\t\tThe error message is: {e}")

    # check sink and source junctions:
    node_component = ["sink", "source", "ext_grid"]
    for nc in node_component:
        if hasattr(net, nc):
            missing = np.setdiff1d(net[nc].junction, net.junction.index)
            if len(missing):
                logger.warning(f"Some {nc}s are connected to non-existing junctions!"
                               f"\n{nc}s:{net[nc].loc[net[nc].junction.isin(missing)]}"
                               f"\nmissing junctions:{missing}")

    # check from and to junctions
    branch_component = ["pipe", "valve", "compressor", "pump", "heat_exchanger", "circulation_pump"]
    for bc in branch_component:
        if hasattr(net, bc):
            missing_f = np.setdiff1d(net[bc].from_junction, net.junction.index)
            missing_t = np.setdiff1d(net[bc].to_junction, net.junction.index)
            if len(missing_t) | len(missing_t):
                logger.warning(f"Some {bc}s are connected to non-existing junctions!")
                logger.warning(f"missing 'from' junctions:{missing_f}")
                logger.warning(f"missing 'to' junctions:{missing_t}")


def pipeflow_alpha_sweep(net, **kwargs):
    """Run the pipeflow many times with different alpha (NR damping factor) settings between 0.1 and 1 in steps of 0.1"""
    net.converged = False
    alphas = [1]
    for i in range(1, 10):
        if i % 2 == 1:
            alphas.append(round(1 - (i // 2) * 0.1, 1))
        else:
            alphas.append(round((i // 2) * 0.1, 1))
    if kwargs is None:
        kwargs = {}

    for alpha in alphas:
        kwargs.update({"alpha": alpha})
        try:
            pps.pipeflow(net, **kwargs)
        except Exception as e:
            logger.debug(f"Pipeflow did not converge with alpha = {alpha}.\nError: {e}")
        if net.converged:
            logger.info(f"Pipeflow did converge with alpha = {alpha}.")
            return
    logger.warn(f"Pipeflow did not converge with any alpha in {alphas}.")


if __name__ == '__main__':
    import pandapipes.networks
    net = pandapipes.networks.schutterwald()
    net.ext_grid.p_bar = 0.08
    net.sink.loc[1505, "mdot_kg_per_s"] = 1000
    try:
        pandapipes.pipeflow(net)
    except Exception as e:
        print(f"pipeflow raised: \n {e}")
    check_net(net)
