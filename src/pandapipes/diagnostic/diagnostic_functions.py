import pandapipes as pp
import numpy as np
import pandas as pd

from pandapipes import PipeflowNotConverged
from pandapipes.idx_node import PINIT
from pandapipes.idx_branch import MDOTINIT
from pandapipes.diagnostic.diagnostic_helper import (
    DiagnosticFunction,
    check_boolean,
    check_greater_equal_zero,
    check_greater_zero,
    check_number,
    check_pos_int,
    check_existing_junction,
)

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)




default_argument_values = {
    "standard_pipe_length_km": 0.01,
    "pipe_length_limit_km": 10.0,
    "iteration_limit": 200,
    "sink_source_scaling_factor": 1e-5,
    "roughness_limit_mm": 0.5,
    "gas_diameter_threshold_mm": 6,
    "liquid_diameter_threshold_mm": 20,
    "diameter_increase_factor": 20,
    "heat_transfer_coefficient_limit": 5,
    "heat_transfer_coefficient_scaling_factor": 0.1,
    "heat_consumer_scaling_factor": 0.1,
    "deltat_scaling_factor": 2,
    "compressor_neutral_pressure_ratio": 1,
    "ext_grid_pressure_scaling_factor": 1e-5,
    "circ_pump_mass_flow_scaling_factor": 0.1,
    "compressor_pressure_ratio_limit": 5,


    "alpha_min": 0.1,
    "alpha_max": 1.0,
    "alpha_step": 0.1,
}



class InvalidValuesCheck(DiagnosticFunction):

    def diagnostic(self, net, **kwargs):
        result = {}

        important_values = {
            "junction": [
                ("pn_bar", ">0"),
                ("tfluid_k", ">0"),
                ("height_m", ">=0"),
            ],
            "pipe": [
                ("from_junction", "existing_junction"),
                ("to_junction", "existing_junction"),
                ("length_km", ">0"),
                ("inner_diameter_mm", ">0"),
                ("k_mm", ">0"),
                ("alpha_w_per_m2k", ">=0"),
                ("loss_coefficient", ">=0"),
                ("sections", "positive_integer"),
                ("in_service", "boolean"),
            ],
            "ext_grid": [
                ("junction", "existing_junction"),
                ("p_bar", ">0"),
                ("t_k", ">0"),
            ],
            "sink": [
                ("junction", "existing_junction"),
                ("mdot_kg_per_s", ">=0"),
                ("scaling", ">=0"),
                ("in_service", "boolean"),
            ],
            "source": [
                ("junction", "existing_junction"),
                ("mdot_kg_per_s", ">=0"),
                ("scaling", ">=0"),
                ("in_service", "boolean"),
            ],
            "valve": [
                ("from_junction", "existing_junction"),
                ("to_junction", "existing_junction"),
                ("diameter_m", ">0"),
                ("loss_coefficient", ">=0"),
                ("opened", "boolean"),
                ("in_service", "boolean"),
            ],
            "pump": [
                ("from_junction", "existing_junction"),
                ("to_junction", "existing_junction"),
                ("in_service", "boolean"),
            ],
            "compressor": [
                ("from_junction", "existing_junction"),
                ("to_junction", "existing_junction"),
                ("pressure_ratio", ">0"),
                ("in_service", "boolean"),
            ],
            "press_control": [
                ("from_junction", "existing_junction"),
                ("to_junction", "existing_junction"),
                ("controlled_junction", "existing_junction"),
                ("controlled_p_bar", ">0"),
                ("control_active", "boolean"),
                ("in_service", "boolean"),
            ],
            "flow_control": [
                ("from_junction", "existing_junction"),
                ("to_junction", "existing_junction"),
                ("controlled_mdot_kg_per_s", ">0"),
                ("control_active", "boolean"),
                ("in_service", "boolean"),
            ],
            "circ_pump_pressure": [
                ("from_junction", "existing_junction"),
                ("to_junction", "existing_junction"),
                ("p_flow_bar", ">0"),
                ("t_flow_k", ">0"),
                ("in_service", "boolean"),
            ],
            "circ_pump_mass": [
                ("from_junction", "existing_junction"),
                ("to_junction", "existing_junction"),
                ("mdot_flow_kg_per_s", ">0"),
                ("p_flow_bar", ">0"),
                ("t_flow_k", ">0"),
                ("in_service", "boolean"),
            ],
            "heat_consumer": [
                ("from_junction", "existing_junction"),
                ("to_junction", "existing_junction"),
                ("qext_w", "number"),
                ("controlled_mdot_kg_per_s", ">0"),
                ("deltat_k", ">0"),
                ("treturn_k", ">=0"),
                ("scaling", ">=0"),
                ("in_service", "boolean"),
            ],
        }

        type_checks = {
            ">0": check_greater_zero,
            ">=0": check_greater_equal_zero,
            "number": check_number,
            "boolean": check_boolean,
            "positive_integer": check_pos_int,
            "existing_junction": check_existing_junction,
        }


        for element, checks in important_values.items():
            if not hasattr(net, element):
                continue

            table = net[element]

            if table.empty:
                continue

            for idx, row in table.iterrows():
                for column, restriction in checks:
                    if column not in table.columns:
                        continue

                    value = row[column]

                    # heat_consumer has optional control columns; NaN can be valid there
                    if element == "heat_consumer" and column in [
                        "qext_w",
                        "controlled_mdot_kg_per_s",
                        "deltat_k",
                        "treturn_k",
                    ]:
                        if pd.isna(value):
                            continue

                    if restriction == "existing_junction":
                        check_result = type_checks[restriction](
                            row, idx, column, net.junction.index
                        )
                    else:
                        check_result = type_checks[restriction](
                            row, idx, column
                        )

                    if check_result is not None:
                        result.setdefault(element, []).append(
                            (idx, column, value, restriction)
                        )

        return result if result else None

    def report(self, error, result):
        if error is not None:
            self.out.warning("Invalid-values check failed due to the following error:")
            self.out.warning(error)
            return

        if result is None:
            return

        self.out.warning("invalid_values:")

        for element, violations in result.items():
            self.out.warning(f"{element}:")

            for idx, column, value, restriction in violations:
                self.out.warning(
                    f"{element} {idx}: '{column}' = {value} "
                    f"(restriction: {restriction})"
                )


# check ext_grid
class MissingExtGridCheck(DiagnosticFunction):

    def diagnostic(self, net, **kwargs):
        if net.fluid.is_gas and (
            not hasattr(net, "ext_grid") or net.ext_grid.empty
        ):
            return True

        return None

    def report(self, error, result):
        if error is not None:
            self.out.warning(
                "Missing ext_grid check failed due to the following error:"
            )
            self.out.warning(error)
            return

        if result is None:
            return

        self.out.warning(
            "The net does not have an external grid! "
            "An external grid is required for gas networks."
        )

# check with standard ext_grid pressure
class ExtGridPressureCheck(DiagnosticFunction):

    def __init__(self):
        super().__init__()
        self.ext_grid_pressure_scaling_factor = None

    def diagnostic(self, net, **kwargs):

        if not hasattr(net, "ext_grid") or net.ext_grid.empty:
            return None

        self.ext_grid_pressure_scaling_factor = kwargs["ext_grid_pressure_scaling_factor"]

        net0 = net.deepcopy()
        try:
            pp.pipeflow(net0)

            if net0.converged:
                return None
        except PipeflowNotConverged:
            pass

        if "p_bar" not in net.ext_grid.columns:
            return None

        net2 = net.deepcopy()
        net2.ext_grid.loc[
            net2.ext_grid.p_bar.notna(),
            "p_bar"
        ] *= self.ext_grid_pressure_scaling_factor

        try:
            pp.pipeflow(net2)
            return net2.converged

        except PipeflowNotConverged:
            return False

        except Exception:
            raise

    def report(self, error, result):

        if error is not None:
            self.out.warning(
                "Ext-grid pressure check failed due to the following error:"
            )
            self.out.warning(error)
            return

        if result is None:
            return

        self.out.detailed(
            "Checking ext_grid pressure...\n"
        )

        if result:
            self.out.warning(
                f"Ext-grid pressure problem suspected: "
                f"pipeflow converges if ext_grid pressures are scaled "
                f"by a factor of {self.ext_grid_pressure_scaling_factor}."
            )
        else:
            self.out.warning(
                f"Pipeflow still does not converge if ext_grid pressures "
                f"are scaled by a factor of "
                f"{self.ext_grid_pressure_scaling_factor}."
            )


# check with standardized pipe lengths
class PipeLengthCheck(DiagnosticFunction):

    def __init__(self):
        super().__init__()
        self.pipe_length_limit_km = None
        self.standard_pipe_length_km = None

    def diagnostic(self, net, **kwargs):
        self.pipe_length_limit_km = kwargs["pipe_length_limit_km"]
        self.standard_pipe_length_km = kwargs["standard_pipe_length_km"]

        if not hasattr(net, "pipe") or net.pipe.empty:
            return None

        # Check original network first
        net0 = net.deepcopy()

        try:
            pp.pipeflow(net0)
            if net0.converged:
                return None

        except PipeflowNotConverged:
            pass

        results = {
            "long_pipes": None,
            "all_pipes": None,
        }

        # Shorten only pipes above the defined length limit
        long_pipes = (net.pipe["length_km"] > self.pipe_length_limit_km)

        if long_pipes.any():
            net2 = net.deepcopy()

            net2.pipe.loc[long_pipes, "length_km"] = self.standard_pipe_length_km

            try:
                pp.pipeflow(net2)
                results["long_pipes"] = net2.converged

            except PipeflowNotConverged:
                results["long_pipes"] = False

            except Exception:
                raise

            if results["long_pipes"]:
                return results


        # Shorten all pipes if the first step was not sufficient
        net3 = net.deepcopy()
        net3.pipe["length_km"] = self.standard_pipe_length_km

        try:
            pp.pipeflow(net3)
            results["all_pipes"] = net3.converged

        except PipeflowNotConverged:
            results["all_pipes"] = False

        except Exception:
            raise

        return results

    def report(self, error, result):
        if error is not None:
            self.out.warning("Pipe-length check failed due to the following error:")
            self.out.warning(error)
            return

        if result is None:
            return

        logger.detailed("Checking pipe lengths...\n")

        if result["long_pipes"]:
            self.out.warning(
                f"Pipe-length problem suspected: "
                f"pipeflow converges if pipes longer than "
                f"{self.pipe_length_limit_km} km are set to "
                f"{self.standard_pipe_length_km} km."
            )

        elif result["all_pipes"]:
            self.out.warning(
                f"Pipe-length problem suspected: "
                f"shortening only pipes longer than "
                f"{self.pipe_length_limit_km} km was not sufficient, "
                f"but pipeflow converges if all pipe lengths are set to "
                f"{self.standard_pipe_length_km} km."
            )

        else:
            self.out.warning(
                f"Pipeflow still does not converge if all pipe lengths "
                f"are set to {self.standard_pipe_length_km} km."
            )

# check iterations
class IterationCheck(DiagnosticFunction):

    def __init__(self):
        super().__init__()

        self.iterations = None

    def diagnostic(self, net, **kwargs):
        self.iterations = kwargs["iteration_limit"]
        net0 = net.deepcopy()

        try:
            pp.pipeflow(net0)
            if net0.converged:
                return None
        except PipeflowNotConverged:
            pass

        net2 = net.deepcopy()

        try:
            pp.pipeflow(net2, iter=self.iterations)
            return True

        except PipeflowNotConverged:
            return False

        except Exception:
            raise

    def report(self, error, result):
        if error is not None:
            self.out.warning(
                "Iteration check failed due to the following error:"
            )
            self.out.warning(error)
            return

        if result is None:
            return

        logger.detailed("Checking iteration limit...\n")

        if result:
            self.out.warning(
                f"The pipeflow converges if the maximum number of iterations "
                f"is increased to {self.iterations}."
            )
        else:
            self.out.warning(
                f"After {self.iterations:d} iterations "
                f"the pipeflow did NOT converge."
            )

# check with little sink and source scaling
class SinkSourceScalingCheck(DiagnosticFunction):

    def __init__(self):
        super().__init__()
        self.scaling_factor = None

    def diagnostic(self, net, **kwargs):
        self.scaling_factor = kwargs["sink_source_scaling_factor"]

        if not hasattr(net, "sink") and not hasattr(net, "source"):
            return None

        try:
            pp.pipeflow(net.deepcopy())
            return None
        except PipeflowNotConverged:
            pass

        result = {
            "sink": False,
            "source": False,
            "both": False
        }

        has_sink = hasattr(net, "sink") and not net.sink.empty
        has_source = hasattr(net, "source") and not net.source.empty

        # 1) only sinks
        if has_sink and not has_source:
            net_sink = net.deepcopy()
            net_sink.sink.scaling *= self.scaling_factor

            try:
                pp.pipeflow(net_sink)
                result["sink"] = True
                return result
            except PipeflowNotConverged:
                pass

        # 2) only sources
        if has_source and not has_sink:
            net_source = net.deepcopy()
            net_source.source.scaling *= self.scaling_factor

            try:
                pp.pipeflow(net_source)
                result["source"] = True
                return result
            except PipeflowNotConverged:
                pass

        # 3) sinks and sources together
        net_both = net.deepcopy()
        changed_anything = False

        if has_sink and has_source:
            net_both.sink.scaling *= self.scaling_factor
            net_both.source.scaling *= self.scaling_factor
            changed_anything = True

        try:
            pp.pipeflow(net_both)
            result["both"] = True
        except PipeflowNotConverged:
            pass
        except Exception:
            raise

        return result

    def report(self, error, result):
        if error is not None:
            self.out.warning(
                "Sink/source scaling check failed due to the following error:"
            )
            self.out.warning(error)
            return

        if result is None:
            return

        self.out.detailed("Checking sink/source scaling...\n")

        if result["sink"]:
            self.out.warning(
                f"Sink overload suspected: pipeflow converges if sinks are "
                f"scaled by a factor of {self.scaling_factor}."
            )

        elif result["source"]:
            self.out.warning(
                f"Source overload suspected: pipeflow converges if sources are "
                f"scaled by a factor of {self.scaling_factor}."
            )

        elif result["both"]:
            self.out.warning(
                f"Sink/source overload suspected: pipeflow converges if sinks "
                f"and sources are scaled by a factor of {self.scaling_factor}."
            )

        else:
            self.out.warning(
                f"Pipeflow still does not converge if sinks and sources are "
                f"scaled by a factor of {self.scaling_factor}."
            )

# check pipe roughness values
class PipeRoughnessCheck(DiagnosticFunction):

    def __init__(self):
        super().__init__()
        self.roughness_limit_mm = None
        self.rough_pipes = None

    def diagnostic(self, net, **kwargs):
        self.roughness_limit_mm = kwargs["roughness_limit_mm"]

        if not hasattr(net, "pipe") or net.pipe.empty:
            return None

        if "k_mm" not in net.pipe.columns:
            return None

        self.rough_pipes = net.pipe.loc[
            net.pipe.k_mm > self.roughness_limit_mm
        ]

        if self.rough_pipes.empty:
            return None

        return {
            "rough_pipes": list(self.rough_pipes.index),
            "highest_k_mm": self.rough_pipes.k_mm.max(),
        }

    def report(self, error, result):
        if error is not None:
            self.out.warning(
                "Pipe-roughness check failed due to the following error:"
            )
            self.out.warning(error)
            return

        if result is None:
            return

        self.out.warning(
            f"Some pipes have a friction factor k_mm > "
            f"{self.roughness_limit_mm} mm (extremely rough). "
            f"The highest value in the net is {result['highest_k_mm']} mm. "
            f"Up to 0.2 mm is a common value for old steel pipes."
        )

# check sink and source junctions:
class MissingNodeJunctionsCheck(DiagnosticFunction):

    def diagnostic(self, net, **kwargs):
        node_components = ["sink", "source", "ext_grid"]
        result = {}

        for nc in node_components:
            if hasattr(net, nc):
                missing = np.setdiff1d(net[nc].junction, net.junction.index)

                if len(missing):
                    result[nc] = {
                        "missing_junctions": missing,
                        "elements": net[nc].loc[net[nc].junction.isin(missing)]
                    }

        return result if result else None

    def report(self, error, result):
        if error is not None:
            self.out.warning(
                "Node-junction check failed due to the following error:"
            )
            self.out.warning(error)
            return

        if result is None:
            return

        for nc, values in result.items():
            self.out.warning(
                f"Some {nc}s are connected to non-existing junctions!"
                f"\n{nc}s:{values['elements']}"
                f"\nmissing junctions:{values['missing_junctions']}"
            )


# check from and to junctions
class MissingBranchJunctionsCheck(DiagnosticFunction):

    def diagnostic(self, net, **kwargs):
        branch_components = ["pipe", "valve", "compressor", "pump", "heat_exchanger", "circ_pump_pressure", "circ_pump_mass"]
        result = {}

        for bc in branch_components:
            if not hasattr(net, bc):
                continue

            table = net[bc]

            if table.empty:
                continue

            if "from_junction" not in table.columns or "to_junction" not in table.columns:
                continue

            missing_f = np.setdiff1d(table.from_junction, net.junction.index)
            missing_t = np.setdiff1d(table.to_junction, net.junction.index)

            if len(missing_f) or len(missing_t):
                result[bc] = {
                    "missing_from_junctions": missing_f,
                    "missing_to_junctions": missing_t
                }

        return result if result else None

    def report(self, error, result):
        if error is not None:
            self.out.warning("Branch-junction check failed due to the following error:")
            self.out.warning(error)
            return

        if result is None:
            return

        for bc, values in result.items():
            self.out.warning(f"Some {bc}s are connected to non-existing junctions!")
            self.out.warning(f"missing 'from' junctions: {values['missing_from_junctions']}")
            self.out.warning(f"missing 'to' junctions: {values['missing_to_junctions']}")


# check with increased pipe inner_diameter_mm
class PipeDiameterCheck(DiagnosticFunction):

    def __init__(self):
        super().__init__()
        self.diameter_increase_factor = None
        self.diameter_threshold_mm = None

    def diagnostic(self, net, **kwargs):
        self.diameter_increase_factor = kwargs["diameter_increase_factor"]

        if not hasattr(net, "pipe") or net.pipe.empty:
            return None


        net0 = net.deepcopy()

        try:
            pp.pipeflow(net0)
            if net0.converged:
                return None

        except PipeflowNotConverged:
            pass

        # Only now check whether there are very small pipes.
        if net.fluid.is_gas:
            self.diameter_threshold_mm = kwargs["gas_diameter_threshold_mm"]
        else:
            self.diameter_threshold_mm = kwargs["liquid_diameter_threshold_mm"]

        small_pipes = net.pipe.inner_diameter_mm < self.diameter_threshold_mm

        if not small_pipes.any():
            return None

        # Test whether increasing small pipe diameters helps.
        net2 = net.deepcopy()
        net2.pipe.loc[
            small_pipes,
            "inner_diameter_mm"
        ] *= self.diameter_increase_factor

        try:
            pp.pipeflow(net2)
            return net2.converged

        except PipeflowNotConverged:
            return False

        except Exception:
            raise

    def report(self, error, result):
        if error is not None:
            self.out.warning("Pipe-diameter check failed due to the following error:")
            self.out.warning(error)
            return

        if result is None:
            return

        logger.detailed("Checking pipe diameters...\n")
        if result:
            self.out.warning(
                f"Pipe-diameter problem suspected: "
                f"pipeflow converges if pipe diameters below "
                f"{self.diameter_threshold_mm} mm are increased by factor "
                f"{self.diameter_increase_factor}."
            )
        else:
            self.out.warning(
                f"Pipeflow still does not converge if pipe diameters below "
                f"{self.diameter_threshold_mm} mm are increased by factor "
                f"{self.diameter_increase_factor}."
            )

# check heat transfer coefficient
class HeatTransferCoefficientCheck(DiagnosticFunction):

    def __init__(self):
        super().__init__()
        self.limit = None
        self.scaling_factor = None

    def diagnostic(self, net, **kwargs):

        self.limit = kwargs["heat_transfer_coefficient_limit"]
        self.scaling_factor = kwargs["heat_transfer_coefficient_scaling_factor"]

        if (
            not hasattr(net, "pipe")
            or net.pipe.empty
            or "u_w_per_m2k" not in net.pipe.columns
        ):
            return None

        if not (net.pipe.u_w_per_m2k > self.limit).any():
            return None

        net0 = net.deepcopy()

        try:
            pp.pipeflow(net0,  mode="bidirectional")
            if net0.converged:
                return None

        except PipeflowNotConverged:
            pass

        net2 = net.deepcopy()
        net2.pipe.loc[
            net2.pipe.u_w_per_m2k > self.limit,
            "u_w_per_m2k"
        ] *= self.scaling_factor

        try:
            pp.pipeflow(net2,  mode="bidirectional")
            return net2.converged

        except PipeflowNotConverged:
            return False

        except Exception:
            raise

    def report(self, error, result):
        if error is not None:
            self.out.warning(
                "Heat-transfer-coefficient check failed due to the following error:"
            )
            self.out.warning(error)
            return

        if result is None:
            return

        logger.detailed("Checking heat-transfer coefficients...\n")

        if result:
            self.out.warning(
                f"If pipe heat transfer coefficients above {self.limit} W/(m²K) were reduced "
                f"by factor {self.scaling_factor}, the pipeflow would converge."
            )
        else:
            self.out.warning(
                f"Pipeflow still does not converge if pipe heat transfer coefficients "
                f"above {self.limit} W/(m²K) are reduced by factor {self.scaling_factor}."
            )

# Check with all valves opened and all valves closed
class ValveConfigurationCheck(DiagnosticFunction):

    def diagnostic(self, net, **kwargs):
        if not hasattr(net, "valve") or net.valve.empty:
            return None

        # Check original network first
        net0 = net.deepcopy()

        try:
            pp.pipeflow(net0)

            if net0.converged:
                return None

        except PipeflowNotConverged:
            pass

        results = {
            "all_open": False,
            "all_closed": False,
        }

        # Check with all valves opened
        net_open = net.deepcopy()
        net_open.valve.opened = True

        try:
            pp.pipeflow(net_open)
            results["all_open"] = net_open.converged

        except PipeflowNotConverged:
            results["all_open"] = False

        except Exception:
            raise

        # Check with all valves closed
        net_closed = net.deepcopy()
        net_closed.valve.opened = False

        try:
            pp.pipeflow(net_closed)
            results["all_closed"] = net_closed.converged

        except PipeflowNotConverged:
            results["all_closed"] = False

        except Exception:
            raise

        return results

    def report(self, error, result):
        if error is not None:
            self.out.warning("Valve-configuration check failed due to the following error:")
            self.out.warning(error)
            return

        if result is None:
            return

        logger.detailed("Checking valve configuration...\n")

        if result["all_open"]:
            self.out.warning("If all valves were opened, the pipeflow would converge.")
        else:
            self.out.warning("Pipeflow still does not converge if all valves are opened.")

        if result["all_closed"]:
            self.out.warning("If all valves were closed, the pipeflow would converge.")
        else:
            self.out.warning("Pipeflow still does not converge if all valves are closed.")


class HeatConsumerControlParameterCheck(DiagnosticFunction):
    """
    Checks whether heat consumer control parameters are the reason
    for the pipeflow non-convergence.

    The idea is to reduce the thermal load of the heat consumers.
    Lower heat demand (qext_w) and mass flow
    (controlled_mdot_kg_per_s) reduce the hydraulic and thermal
    stress on the network.

    For configurations using deltat_k, the temperature difference
    is increased to reduce the required mass flow according to

    Q = m * cp * deltaT
    """
    def __init__(self):
        super().__init__()
        self.heat_consumer_scaling_factor = None
        self.deltat_scaling_factor = None
        self.affected_heat_consumers = None

    def diagnostic(self, net, **kwargs):

        self.heat_consumer_scaling_factor = kwargs["heat_consumer_scaling_factor"]
        self.deltat_scaling_factor = kwargs["deltat_scaling_factor"]

        if not hasattr(net, "heat_consumer") or net.heat_consumer.empty:
            return None

        net0 = net.deepcopy()
        try:
            pp.pipeflow(net0)
            if net0.converged:
                return None
        except PipeflowNotConverged:
            pass

        net2 = net.deepcopy()
        hc = net2.heat_consumer

        # combination 1: qext_w + controlled_mdot_kg_per_s
        # reduce heat demand and mass flow
        mask_qext_mdot = (
            hc.qext_w.notna()
            & hc.controlled_mdot_kg_per_s.notna()
            & hc.deltat_k.isna()
            & hc.treturn_k.isna()
        )

        # combination 2: qext_w + deltat_k
        # reduce heat demand and increase deltaT
        mask_qext_deltat = (
            hc.qext_w.notna()
            & hc.controlled_mdot_kg_per_s.isna()
            & hc.deltat_k.notna()
            & hc.treturn_k.isna()
        )

        # combination 3: qext_w + treturn_k
        # reduce heat demand
        mask_qext_treturn = (
            hc.qext_w.notna()
            & hc.controlled_mdot_kg_per_s.isna()
            & hc.deltat_k.isna()
            & hc.treturn_k.notna()
        )

        # combination 4: controlled_mdot_kg_per_s + deltat_k
        # reduce mass flow and increase deltaT
        mask_mdot_deltat = (
            hc.qext_w.isna()
            & hc.controlled_mdot_kg_per_s.notna()
            & hc.deltat_k.notna()
            & hc.treturn_k.isna()
        )

        # combination 5: controlled_mdot_kg_per_s + treturn_k
        # reduce mass flow
        mask_mdot_treturn = (
            hc.qext_w.isna()
            & hc.controlled_mdot_kg_per_s.notna()
            & hc.deltat_k.isna()
            & hc.treturn_k.notna()
        )

        affected_mask = (
            mask_qext_mdot
            | mask_qext_deltat
            | mask_qext_treturn
            | mask_mdot_deltat
            | mask_mdot_treturn
        )

        self.affected_heat_consumers = hc.index[affected_mask].tolist()

        hc.loc[mask_qext_mdot, "qext_w"] *= self.heat_consumer_scaling_factor
        hc.loc[mask_qext_mdot, "controlled_mdot_kg_per_s"] *= self.heat_consumer_scaling_factor

        hc.loc[mask_qext_deltat, "qext_w"] *= self.heat_consumer_scaling_factor
        hc.loc[mask_qext_deltat, "deltat_k"] *= self.deltat_scaling_factor

        hc.loc[mask_qext_treturn, "qext_w"] *= self.heat_consumer_scaling_factor

        hc.loc[mask_mdot_deltat, "controlled_mdot_kg_per_s"] *= self.heat_consumer_scaling_factor
        hc.loc[mask_mdot_deltat, "deltat_k"] *= self.deltat_scaling_factor

        hc.loc[mask_mdot_treturn, "controlled_mdot_kg_per_s"] *= self.heat_consumer_scaling_factor

        try:
            pp.pipeflow(net2)
            return net2.converged
        except PipeflowNotConverged:
            return False
        except Exception as e:
            raise e

    def report(self, error, result):
        if error is not None:
            self.out.warning(
                "Heat-consumer control-parameter check failed due to the following error:"
            )
            self.out.warning(error)
            return

        if result is None:
            return

        self.out.warning("Testing with adjusted heat consumer control parameters.")

        if self.affected_heat_consumers is not None:
            self.out.warning(f"Adjusted heat consumer IDs: {self.affected_heat_consumers}")

        if result:
            self.out.warning(
                "If heat consumer control parameters were adjusted, "
                "the pipeflow would converge."
            )
        else:
            self.out.warning(
                "Pipeflow still does not converge if heat consumer control "
                "parameters are adjusted."
            )

# check with flattened junction heights
class JunctionHeightCheck(DiagnosticFunction):

    def diagnostic(self, net, **kwargs):
        if net.junction.height_m.nunique() <= 1:
            return None

        net0 = net.deepcopy()

        try:
            pp.pipeflow(net0)
            if net0.converged:
                return None
        except PipeflowNotConverged:
            pass

        net2 = net.deepcopy()
        net2.junction.height_m = 0

        try:
            pp.pipeflow(net2)
            return net2.converged

        except PipeflowNotConverged:
            return False

        except Exception:
            raise

    def report(self, error, result):
        if error is not None:
            self.out.warning(
                "Junction-height check failed due to the following error:"
            )
            self.out.warning(error)
            return

        if result is None:
            return

        logger.detailed(
            "Checking junction-height configuration...\n"
        )
        if result:
            self.out.warning(
                "If all junction heights were set to 0 m, "
                "the pipeflow would converge."
            )
        else:
            self.out.warning(
                "Pipeflow still does not converge if all junction heights "
                "are set to 0 m."
            )


class CalculationModeCheck(DiagnosticFunction):
    """
    Checks whether the non-convergence originates from hydraulics,
    heat transfer, or the coupling between both calculations.

    Heat mode cannot be executed independently because it requires
    hydraulic results (node pressures and branch mass flows) as input.
    Therefore, a hydraulic calculation is executed first and the
    resulting PINIT and MDOTINIT values are passed to the heat
    calculation via sol_vec.
    """
    def __init__(self, modes=None):
        super().__init__()
        self.modes = modes or ["hydraulics", "heat", "sequential", "bidirectional"]

    def diagnostic(self, net, **kwargs):
        net0 = net.deepcopy()

        try:
            pp.pipeflow(net0)
            if net0.converged:
                return None
        except PipeflowNotConverged:
            pass

        results = {}

        for mode in self.modes:
            net2 = net.deepcopy()

            try:
                if mode == "heat":
                    pp.pipeflow(net2, mode="hydraulics")
                    sol_vec = np.r_[
                        net2["_pit"]["node"][:, PINIT],
                        net2["_pit"]["branch"][:, MDOTINIT]
                    ]
                    pp.pipeflow(net2, mode="heat", sol_vec=sol_vec)
                else:
                    pp.pipeflow(net2, mode=mode)

                results[mode] = net2.converged

            except PipeflowNotConverged:
                results[mode] = False

            except Exception:
                raise

        return results

    def report(self, error, result):
        if error is not None:
            self.out.warning(
                "Calculation-mode check failed due to the following error:"
            )
            self.out.warning(error)
            return

        if result is None:
            return

        logger.detailed("Checking calculation modes...\n")

        for mode, converged in result.items():
            if converged:
                self.out.warning(
                    f"The pipeflow converges in mode '{mode}'."
                )
            else:
                self.out.warning(
                    f"Pipeflow still does not converge in mode '{mode}'."
                )


# check with changed friction model
class FrictionModelCheck(DiagnosticFunction):

    def __init__(self, friction_models=None):
        super().__init__()
        self.friction_models = friction_models or ["nikuradse", "colebrook", "swamee-jain"]

    def diagnostic(self, net, **kwargs):
        # First check the original pipeflow.
        # If it already converges, this comparison is not needed.
        net0 = net.deepcopy()

        try:
            pp.pipeflow(net0)
            if net0.converged:
                return None
        except PipeflowNotConverged:
            pass

        results = {}

        for friction_model in self.friction_models:
            net2 = net.deepcopy()

            try:
                pp.pipeflow(net2, friction_model=friction_model)
                results[friction_model] = net2.converged

            except PipeflowNotConverged:
                results[friction_model] = False

            except Exception:
                raise

        return results

    def report(self, error, result):
        if error is not None:
            self.out.warning("Friction-model check failed due to the following error:")
            self.out.warning(error)
            return

        if result is None:
            return

        logger.detailed("Checking friction models...\n")

        for friction_model, converged in result.items():
            if converged:
                self.out.warning(
                    f"Friction-model dependency suspected: "
                    f"pipeflow converges with friction model '{friction_model}'."
                )
            else:
                self.out.warning(
                    f"Pipeflow still does not converge with friction model "
                    f"'{friction_model}'."
                )

class AlphaSweepCheck(DiagnosticFunction):
    """
    Checks whether the pipeflow converges with a different Newton damping factor alpha.
    """

    def __init__(self):
        super().__init__()
        self.alphas = None
        self.successful_alpha = None

    def diagnostic(self, net, **kwargs):
        alpha_min = kwargs["alpha_min"]
        alpha_max = kwargs["alpha_max"]
        alpha_step = kwargs["alpha_step"]

        # First check the original pipeflow.
        # If it already converges, this check is not needed.
        net0 = net.deepcopy()

        try:
            pp.pipeflow(net0)
            if net0.converged:
                return None
        except PipeflowNotConverged:
            pass

        # Build alpha list similar to the original helper function.
        # Example: [1.0, 0.9, 0.1, 0.8, 0.2, ...]
        high_values = np.arange(alpha_max, alpha_min - alpha_step / 2, -alpha_step)
        low_values = np.arange(alpha_min, alpha_max + alpha_step / 2, alpha_step)

        self.alphas = []
        for high, low in zip(high_values, low_values):
            high = round(float(high), 10)
            low = round(float(low), 10)

            if high not in self.alphas:
                self.alphas.append(high)

            if low not in self.alphas:
                self.alphas.append(low)

        for alpha in self.alphas:
            net2 = net.deepcopy()

            try:
                pp.pipeflow(net2, alpha=alpha)

                if net2.converged:
                    self.successful_alpha = alpha
                    return True

            except PipeflowNotConverged:
                pass

            except Exception:
                raise

        return False

    def report(self, error, result):
        if error is not None:
            self.out.warning(
                "Alpha-sweep check failed due to the following error:"
            )
            self.out.warning(error)
            return

        if result is None:
            return

        if result:
            self.out.warning(
                f"Pipeflow converges with alpha = {self.successful_alpha}."
            )
        else:
            self.out.warning(
                f"Pipeflow did not converge with any alpha in {self.alphas}."
            )

# check with inactive pressure controls
class InactivePressureControlsCheck(DiagnosticFunction):

    def diagnostic(self, net, **kwargs):
        if not hasattr(net, "press_control") or net.press_control.empty:
            return None

        # first check original network
        net0 = net.deepcopy()

        try:
            pp.pipeflow(net0)

            if net0.converged:
                return None

        except PipeflowNotConverged:
            pass

        # hypothesis test
        net2 = net.deepcopy()
        net2.press_control.control_active = False

        try:
            pp.pipeflow(net2)
            return net2.converged

        except PipeflowNotConverged:
            return False

        except Exception:
            raise

    def report(self, error, result):
        if error is not None:
            self.out.warning(
                "Inactive-pressure-controls check failed due to the following error:"
            )
            self.out.warning(error)
            return

        if result is None:
            return

        logger.detailed(
            "Checking pressure-control configuration...\n"
        )

        if result:
            self.out.warning(
                "If all pressure controls were deactivated, "
                "the pipeflow would converge."
            )
        else:
            self.out.warning(
                "Pipeflow still does not converge if all pressure controls "
                "are deactivated."
            )

# check for unrealistic compressor pressure ratios
class CompressorPressureRatioCheck(DiagnosticFunction):

    def __init__(self):
        super().__init__()

        self.compressor_pressure_ratio_limit = None
        self.high_pressure_ratio_compressors = None

    def diagnostic(self, net, **kwargs):

        self.compressor_pressure_ratio_limit = kwargs[
            "compressor_pressure_ratio_limit"
        ]

        if (
            not hasattr(net, "compressor")
            or net.compressor.empty
        ):
            return None

        self.high_pressure_ratio_compressors = net.compressor.loc[
            net.compressor.pressure_ratio >
            self.compressor_pressure_ratio_limit
        ]

        if self.high_pressure_ratio_compressors.empty:
            return None

        return True

    def report(self, error, result):

        if error is not None:
            self.out.warning(
                "Compressor-pressure-ratio check failed due to "
                "the following error:"
            )
            self.out.warning(error)
            return

        if result is None:
            return

        logger.detailed(
            "Checking compressor pressure ratios...\n"
        )

        self.out.warning(
            f"Some compressors have pressure_ratio > "
            f"{self.compressor_pressure_ratio_limit}. "

        )
# check with reduced circulation pump mass flow
class CircPumpMassFlowCheck(DiagnosticFunction):

    def __init__(self):
        super().__init__()

        self.scaling_factor = None

    def diagnostic(self, net, **kwargs):

        self.scaling_factor = kwargs[
            "circ_pump_mass_flow_scaling_factor"
        ]

        if (
            not hasattr(net, "circ_pump_mass") or net.circ_pump_mass.empty
        ):
            return None

        net0 = net.deepcopy()

        try:
            pp.pipeflow(net0)
            if net0.converged:
                return None

        except PipeflowNotConverged:
            pass

        net2 = net.deepcopy()
        net2.circ_pump_mass["mdot_flow_kg_per_s"] *= self.scaling_factor

        try:
            pp.pipeflow(net2)
            return net2.converged

        except PipeflowNotConverged:
            return False

        except Exception:
            raise

    def report(self, error, result):

        if error is not None:
            self.out.warning(
                "Circulation-pump-mass-flow check failed due to "
                "the following error:"
            )
            self.out.warning(error)
            return

        if result is None:
            return

        logger.detailed(
            "Checking circulation-pump mass flow...\n"
        )

        if result:
            self.out.warning(
                f"pipeflow converges if all "
                f"circ_pump_const_mass_flow elements have their "
                f"mdot_flow_kg_per_s reduced by factor "
                f"{self.scaling_factor}."
            )
        else:
            self.out.warning(
                f"Pipeflow still does not converge if all "
                f"circ_pump_const_mass_flow elements have their "
                f"mdot_flow_kg_per_s reduced by factor "
                f"{self.scaling_factor}."
            )

default_diagnostic_functions = [
    ("invalid_values", InvalidValuesCheck(), []),
    ("missing_ext_grid", MissingExtGridCheck(), []),
    ("ext_grid_pressure", ExtGridPressureCheck(), None),
    ("pipe_length", PipeLengthCheck(), None),
    ("iteration_check", IterationCheck(), None),
    ("sink_source_scaling", SinkSourceScalingCheck(), None),
    ("pipe_roughness", PipeRoughnessCheck(), None),
    ("missing_node_junctions", MissingNodeJunctionsCheck(), []),
    ("missing_branch_junctions", MissingBranchJunctionsCheck(), []),
    ("pipe_diameter", PipeDiameterCheck(), None),
    ("heat_transfer_coefficient", HeatTransferCoefficientCheck(), None),
    ("valve_opening", ValveConfigurationCheck(), []),
    ("heat_consumer_control_parameter", HeatConsumerControlParameterCheck(), None),
    ("junction_height", JunctionHeightCheck(), []),
    ("calculation_mode", CalculationModeCheck(), []),
    ("friction_model", FrictionModelCheck(), []),
    ("alpha_sweep", AlphaSweepCheck(), None),
    ("compressor_pressure_ratio", CompressorPressureRatioCheck(), None),
    ("inactive_pressure_controls", InactivePressureControlsCheck(), []),
    ("circ_pump_mass_flow", CircPumpMassFlowCheck(), None),
]