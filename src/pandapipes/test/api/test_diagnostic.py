import copy
import pytest
import numpy as np
import pandapipes as pp
from unittest.mock import patch
from pandapipes import PipeflowNotConverged, pandapipesNet
from pandapipes.diagnostic.diagnostic_functions import(
    default_argument_values,
    InvalidValuesCheck,
    MissingExtGridCheck,
    ExtGridPressureCheck,
    IterationCheck,
    SinkSourceScalingCheck,
    MissingNodeJunctionsCheck,
    MissingBranchJunctionsCheck,
    PipeDiameterCheck,
    ValveConfigurationCheck,
    JunctionHeightCheck,
    PipeLengthCheck,
    PipeRoughnessCheck,
    CircPumpMassFlowCheck,
    CompressorPressureRatioCheck,
    HeatTransferCoefficientCheck,
    AlphaSweepCheck,
    InactivePressureControlsCheck,
)

@pytest.fixture(scope="function")
def diag_params():
    return default_argument_values.copy()


def simple_gas_grid():
    net = pp.create_empty_network(fluid="lgas")

    j1 = pp.create_junction(net, 1.05, 293.15, height_m=0)
    j2 = pp.create_junction(net, 1.05, 293.15, height_m=0)
    j3 = pp.create_junction(net, 1.05, 293.15, height_m=0)
    j4 = pp.create_junction(net, 1.05, 293.15, height_m=0)
    j5 = pp.create_junction(net, 1.05, 293.15, height_m=0)
    j6 = pp.create_junction(net, 1.05, 293.15, height_m=0)

    pp.create_ext_grid(net, junction=j1, p_bar=1.1, t_k=293.15)

    pp.create_pipe_from_parameters(net, j1, j2, length_km=10, inner_diameter_mm=300)
    pp.create_pipe_from_parameters(net, j2, j3, length_km=2, inner_diameter_mm=300)
    pp.create_pipe_from_parameters(net, j2, j4, length_km=2.5, inner_diameter_mm=300)
    pp.create_pipe_from_parameters(net, j3, j5, length_km=1, inner_diameter_mm=300)
    pp.create_pipe_from_parameters(net, j4, j6, length_km=1, inner_diameter_mm=300)

    pp.create_valve(net, junction=j5, element=j6, et="ju", inner_diameter_mm=50, opened=True)

    pp.create_sink(net, junction=j4, mdot_kg_per_s=0.545)
    pp.create_source(net, junction=j3, mdot_kg_per_s=0.234)

    net.pipe["alpha_w_per_m2k"] = 0.0

    return net

def multi_pump_dh_network():
    net = pp.create_empty_network(fluid="water")


    j1 = pp.create_junction(net, 1.05, tfluid_k=358.15, geodata=(0, 1000))
    j2 = pp.create_junction(net, 1.05, tfluid_k=358.15, geodata=(0, 0))
    j3 = pp.create_junction(net, 1.05, tfluid_k=358.15, geodata=(500, 0))
    j4 = pp.create_junction(net, 1.05, tfluid_k=358.15, geodata=(1000, 0))
    j5 = pp.create_junction(net, 1.05, tfluid_k=358.15, geodata=(1500, 0))
    j6 = pp.create_junction(net, 1.05, tfluid_k=358.15, geodata=(1500, 1000))
    j7 = pp.create_junction(net, 1.05, tfluid_k=358.15, geodata=(1000, 1000))
    j8 = pp.create_junction(net, 1.05, tfluid_k=358.15, geodata=(500, 1000))

    pp.create_circ_pump_const_pressure(net, j1, j2, p_flow_bar=4, plift_bar=1.5, t_flow_k=358.15)

    pp.create_pipe_from_parameters(net, j2, j3, length_km=0.5, inner_diameter_mm=107.1, k_mm=0.1, sections=5, u_w_per_m2k=0, text_k=283)
    pp.create_pipe_from_parameters(net, j3, j4, length_km=0.5, inner_diameter_mm=107.1, k_mm=0.1, sections=5, u_w_per_m2k=0, text_k=283)
    pp.create_pipe_from_parameters(net, j4, j5, length_km=0.5, inner_diameter_mm=107.1, k_mm=0.1, sections=5, u_w_per_m2k=0, text_k=283)

    pp.create_heat_consumer(net, j5, j6, qext_w=500000, treturn_k=333.15)
    pp.create_heat_consumer(net, j4, j7, qext_w=200000, treturn_k=328.15)

    pp.create_pipe_from_parameters(net, j6, j7, length_km=0.5, inner_diameter_mm=107.1, k_mm=0.1, sections=5, u_w_per_m2k=0, text_k=283)
    pp.create_pipe_from_parameters(net, j7, j8, length_km=0.5, inner_diameter_mm=107.1, k_mm=0.1, sections=5, u_w_per_m2k=0, text_k=283)
    pp.create_pipe_from_parameters(net, j8, j1, length_km=0.5, inner_diameter_mm=107.1, k_mm=0.1, sections=5, u_w_per_m2k=0, text_k=283)

    j9 = pp.create_junction(net, 1.05, tfluid_k=358.15, geodata=(2000, 0))
    j10 = pp.create_junction(net, 1.05, tfluid_k=358.15, geodata=(2000, 1000))
    j11 = pp.create_junction(net, 1.05, tfluid_k=358.15, geodata=(2000, 500))

    pp.create_pipe_from_parameters(net, j5, j9, length_km=0.5, inner_diameter_mm=107.1, k_mm=0.1, sections=5, u_w_per_m2k=0, text_k=283)
    pp.create_pipe_from_parameters(net, j10, j6, length_km=0.5, inner_diameter_mm=107.1, k_mm=0.1, sections=5, u_w_per_m2k=0, text_k=283)

    pp.create_circ_pump_const_mass_flow(net, j10, j11, p_flow_bar=4, mdot_flow_kg_per_s=1, t_flow_k=358.15)
    pp.create_flow_control(net, j11, j9, controlled_mdot_kg_per_s=1)

    return net

def gas_grid_with_compressor_pressure_control():
    net = pp.create_empty_network(fluid="lgas")

    j1 = pp.create_junction(net, pn_bar=16.0, tfluid_k=293.15)
    j2 = pp.create_junction(net, pn_bar=16.0, tfluid_k=293.15)
    j3 = pp.create_junction(net, pn_bar=14.0, tfluid_k=293.15)
    j4 = pp.create_junction(net, pn_bar=14.0, tfluid_k=293.15)
    j5 = pp.create_junction(net, pn_bar=14.0, tfluid_k=293.15)
    j6 = pp.create_junction(net, pn_bar=14.0, tfluid_k=293.15)
    j7 = pp.create_junction(net, pn_bar=12.0, tfluid_k=293.15)
    j8 = pp.create_junction(net, pn_bar=5.0, tfluid_k=293.15)
    j9 = pp.create_junction(net, pn_bar=5.0, tfluid_k=293.15)
    j10 = pp.create_junction(net, pn_bar=5.0, tfluid_k=293.15)
    j11 = pp.create_junction(net, pn_bar=5.0, tfluid_k=293.15)

    pp.create_ext_grid(net, junction=j1, p_bar=16.0, t_k=293.15)

    pp.create_compressor(net, from_junction=j1, to_junction=j2, pressure_ratio=1.10)

    pp.create_pipe_from_parameters(net, j2, j3, length_km=15.0, inner_diameter_mm=400, k_mm=0.1)
    pp.create_pipe_from_parameters(net, j3, j4, length_km=5.0, inner_diameter_mm=250, k_mm=0.1)
    pp.create_pipe_from_parameters(net, j3, j5, length_km=4.0, inner_diameter_mm=200, k_mm=0.1)
    pp.create_pipe_from_parameters(net, j3, j6, length_km=4.5, inner_diameter_mm=200, k_mm=0.1)

    pp.create_valve(net, junction=j5, element=j6, et="ju", inner_diameter_mm=200, opened=True)

    pp.create_pipe_from_parameters(net, j6, j7, length_km=3.0, inner_diameter_mm=200, k_mm=0.1)

    pp.create_pressure_control(
        net,
        from_junction=j7,
        to_junction=j8,
        controlled_junction=j8,
        controlled_p_bar=5.0,
        control_active=True,
    )

    pp.create_pipe_from_parameters(net, j8, j9, length_km=1.5, inner_diameter_mm=150, k_mm=0.1)
    pp.create_pipe_from_parameters(net, j8, j10, length_km=1.5, inner_diameter_mm=150, k_mm=0.1)
    pp.create_pipe_from_parameters(net, j9, j10, length_km=1.0, inner_diameter_mm=150, k_mm=0.1)
    pp.create_pipe_from_parameters(net, j11, j9, length_km=1.0, inner_diameter_mm=100, k_mm=0.1)

    pp.create_sink(net, junction=j4, mdot_kg_per_s=1.0)
    pp.create_sink(net, junction=j5, mdot_kg_per_s=0.45)
    pp.create_sink(net, junction=j6, mdot_kg_per_s=0.40)
    pp.create_sink(net, junction=j9, mdot_kg_per_s=0.20)
    pp.create_sink(net, junction=j10, mdot_kg_per_s=0.20)

    pp.create_source(net, junction=j11, mdot_kg_per_s=0.15)

    return net



@pytest.fixture(scope="function")
def test_nets():
    return [simple_gas_grid(), multi_pump_dh_network(), gas_grid_with_compressor_pressure_control()]


def check_report_function(func, error, result):
    try:
        func.report(error, result)
    except Exception as e:
        raise AssertionError(f"Report function '{func.__class__.__name__}' failed: {e}")


def _apply_invalid_values(net, changes):
    expected = {}

    for element, element_changes in changes.items():
        if not hasattr(net, element):
            continue

        table = net[element]

        if table.empty:
            continue

        for row_pos, column, value, restriction in element_changes:
            if column not in table.columns:
                continue

            if row_pos >= len(table.index):
                continue

            idx = table.index[row_pos]
            net[element].at[idx, column] = value

            expected.setdefault(element, []).append(
                (idx, column, value, restriction)
            )

    return expected


def _run_invalid_values_case(test_nets, changes):
    for net in test_nets:
        net = copy.deepcopy(net)

        expected = _apply_invalid_values(net, changes)

        if not expected:
            continue

        diag_function = InvalidValuesCheck()
        check_result = diag_function.diagnostic(net)
        diag_results = {"invalid_values": check_result} if check_result else {}

        assert diag_results["invalid_values"] == expected

        check_report_function(
            diag_function,
            None,
            diag_results.get("invalid_values", None),
        )


class TestInvalidValuesCheck:

    def test_greater_zero(self, test_nets):
        changes = {
            "junction": [
                (0, "pn_bar", -1.0, ">0"),
                (1, "tfluid_k", 0.0, ">0"),
            ],
            "pipe": [
                (0, "length_km", -1.0, ">0"),
                (1, "inner_diameter_mm", 0.0, ">0"),
                (2, "k_mm", -1.0, ">0"),
            ],
            "ext_grid": [
                (0, "p_bar", 0.0, ">0"),
                (0, "t_k", -1.0, ">0"),
            ],
            "valve": [
                (0, "diameter_m", 0.0, ">0"),
            ],
            "flow_control": [
                (0, "controlled_mdot_kg_per_s", 0.0, ">0"),
            ],
            "circ_pump_pressure": [
                (0, "p_flow_bar", 0.0, ">0"),
                (0, "t_flow_k", -1.0, ">0"),
            ],
            "circ_pump_mass": [
                (0, "mdot_flow_kg_per_s", 0.0, ">0"),
                (0, "p_flow_bar", 0.0, ">0"),
                (0, "t_flow_k", -1.0, ">0"),
            ],
            "heat_consumer": [
                (0, "controlled_mdot_kg_per_s", 0.0, ">0"),
                (0, "deltat_k", 0.0, ">0"),
            ],
        }

        _run_invalid_values_case(test_nets, changes)

    def test_greater_equal_zero(self, test_nets):
        changes = {
            "junction": [
                (0, "height_m", -1.0, ">=0"),
            ],
            "pipe": [
                (0, "alpha_w_per_m2k", -1.0, ">=0"),
                (1, "loss_coefficient", -1.0, ">=0"),
            ],
            "sink": [
                (0, "mdot_kg_per_s", -1.0, ">=0"),
                (0, "scaling", -1.0, ">=0"),
            ],
            "source": [
                (0, "mdot_kg_per_s", -1.0, ">=0"),
                (0, "scaling", -1.0, ">=0"),
            ],
            "valve": [
                (0, "loss_coefficient", -1.0, ">=0"),
            ],
            "heat_consumer": [
                (0, "treturn_k", -1.0, ">=0"),
                (0, "scaling", -1.0, ">=0"),
            ],
        }

        _run_invalid_values_case(test_nets, changes)

    def test_boolean(self, test_nets):
        changes = {
            "pipe": [
                (0, "in_service", "yes", "boolean"),
            ],
            "sink": [
                (0, "in_service", "no", "boolean"),
            ],
            "source": [
                (0, "in_service", "no", "boolean"),
            ],
            "valve": [
                (0, "opened", "False", "boolean"),
                (0, "in_service", "True", "boolean"),
            ],
            "pump": [
                (0, "in_service", "yes", "boolean"),
            ],
            "compressor": [
                (0, "in_service", "yes", "boolean"),
            ],
            "press_control": [
                (0, "control_active", "yes", "boolean"),
                (0, "in_service", "yes", "boolean"),
            ],
            "flow_control": [
                (0, "control_active", "yes", "boolean"),
                (0, "in_service", "yes", "boolean"),
            ],
            "circ_pump_pressure": [
                (0, "in_service", "yes", "boolean"),
            ],
            "circ_pump_mass": [
                (0, "in_service", "yes", "boolean"),
            ],
            "heat_consumer": [
                (0, "in_service", "yes", "boolean"),
            ],
        }

        _run_invalid_values_case(test_nets, changes)

    def test_existing_junction(self, test_nets):
        changes = {
            "pipe": [
                (0, "from_junction", 9999, "existing_junction"),
                (1, "to_junction", 9999, "existing_junction"),
            ],
            "ext_grid": [
                (0, "junction", 9999, "existing_junction"),
            ],
            "sink": [
                (0, "junction", 9999, "existing_junction"),
            ],
            "source": [
                (0, "junction", 9999, "existing_junction"),
            ],
            "valve": [
                (0, "from_junction", 9999, "existing_junction"),
                (0, "to_junction", 9999, "existing_junction"),
            ],
            "pump": [
                (0, "from_junction", 9999, "existing_junction"),
                (0, "to_junction", 9999, "existing_junction"),
            ],
            "compressor": [
                (0, "from_junction", 9999, "existing_junction"),
                (0, "to_junction", 9999, "existing_junction"),
            ],
            "press_control": [
                (0, "from_junction", 9999, "existing_junction"),
                (0, "to_junction", 9999, "existing_junction"),
                (0, "controlled_junction", 9999, "existing_junction"),
            ],
            "flow_control": [
                (0, "from_junction", 9999, "existing_junction"),
                (0, "to_junction", 9999, "existing_junction"),
            ],
            "circ_pump_pressure": [
                (0, "from_junction", 9999, "existing_junction"),
                (0, "to_junction", 9999, "existing_junction"),
            ],
            "circ_pump_mass": [
                (0, "from_junction", 9999, "existing_junction"),
                (0, "to_junction", 9999, "existing_junction"),
            ],
            "heat_consumer": [
                (0, "from_junction", 9999, "existing_junction"),
                (0, "to_junction", 9999, "existing_junction"),
            ],
        }

        _run_invalid_values_case(test_nets, changes)

    def test_positive_integer(self, test_nets):
        changes = {
            "pipe": [
                (0, "sections", 0, "positive_integer"),
                (1, "sections", 1.5, "positive_integer"),
                (2, "sections", "2", "positive_integer"),
            ],
        }

        _run_invalid_values_case(test_nets, changes)

    def test_number(self, test_nets):
        changes = {
            "heat_consumer": [
                (0, "qext_w", "1000", "number"),
            ],
        }

        _run_invalid_values_case(test_nets, changes)

def test_missing_ext_grid():
    net = simple_gas_grid()

    net.ext_grid = net.ext_grid.drop(net.ext_grid.index)
    diag_function = MissingExtGridCheck()
    check_result = diag_function.diagnostic(net)

    assert check_result is True
    check_report_function(
        diag_function,
        None,
        check_result
    )

def test_ext_grid_pressure_check(diag_params):


    net = simple_gas_grid()
    diag_function = ExtGridPressureCheck()

    check_result = diag_function.diagnostic(net, **diag_params)

    assert check_result is None
    check_report_function(diag_function, None, check_result)


    net = simple_gas_grid()
    net.ext_grid.p_bar = 100000

    diag_function = ExtGridPressureCheck()

    check_result = diag_function.diagnostic(net, **diag_params)

    assert check_result == True
    check_report_function(diag_function, None, check_result)


    net = simple_gas_grid()
    net.ext_grid.p_bar = 100000
    net.junction.loc[3, "height_m"] = 10000

    diag_function = ExtGridPressureCheck()

    check_result = diag_function.diagnostic(net, **diag_params)

    assert check_result is False
    check_report_function(diag_function, None, check_result)



def test_iteration_check(diag_params):


    net = simple_gas_grid()

    diag_function = IterationCheck()

    check_result = diag_function.diagnostic(net, **diag_params)

    assert check_result is None
    check_report_function(diag_function,None, check_result)

    net = simple_gas_grid()

    def fake_pipeflow_success(net_arg, **kwargs):
        if "iter" not in kwargs:
            raise PipeflowNotConverged()
        net_arg.converged = True

    with patch(
        "pandapipes.pipeflow",
        side_effect=fake_pipeflow_success,):
        check_result = diag_function.diagnostic(net, **diag_params)

    assert check_result is True

    check_report_function(diag_function, None, check_result)


    net = simple_gas_grid()

    with patch(
        "pandapipes.pipeflow",
        side_effect=PipeflowNotConverged(),
    ):
        check_result = diag_function.diagnostic(net, **diag_params)

    assert check_result is False

    check_report_function(diag_function, None, check_result)


def test_sink_source_scaling(diag_params):

    net = simple_gas_grid()
    diag_function = SinkSourceScalingCheck()

    check_result = diag_function.diagnostic(net, **diag_params)


    assert check_result is None
    check_report_function(diag_function, None, check_result)


    net = simple_gas_grid()
    net.source.drop(net.source.index, inplace=True)
    net.sink.mdot_kg_per_s *= 1e4

    diag_function = SinkSourceScalingCheck()

    check_result = diag_function.diagnostic(net, **diag_params)

    assert check_result == {
        "sink": True,
        "source": False,
        "both": False,
    }

    check_report_function(diag_function, None, check_result)


    net = simple_gas_grid()
    net.sink.drop(net.sink.index, inplace=True)
    net.source.mdot_kg_per_s *= 1e4

    diag_function = SinkSourceScalingCheck()

    check_result = diag_function.diagnostic(net, **diag_params)

    assert check_result == {
        "sink": False,
        "source": True,
        "both": False,
    }

    check_report_function(diag_function, None, check_result)

    net = simple_gas_grid()
    net.sink.mdot_kg_per_s *= 1e4
    net.source.mdot_kg_per_s *= 1e4

    diag_function = SinkSourceScalingCheck()

    check_result = diag_function.diagnostic(net, **diag_params)

    assert check_result == {
        "sink": False,
        "source": False,
        "both": True,
    }

    check_report_function(diag_function, None, check_result)


    net = simple_gas_grid()
    net.sink.mdot_kg_per_s *= 1e4
    net.source.mdot_kg_per_s *= 1e4
    net.junction.loc[3, "height_m"] = 10000

    diag_function = SinkSourceScalingCheck()

    check_result = diag_function.diagnostic(net, **diag_params)

    assert check_result == {
        "sink": False,
        "source": False,
        "both": False,
    }

    check_report_function(diag_function, None, check_result)


def test_missing_node_junctions():
    net = simple_gas_grid()
    check_function = "missing_node_junctions"

    net.sink.loc[0, "junction"] = 9999
    net.source.loc[0, "junction"] = 9998
    net.ext_grid.loc[0, "junction"] = 9997

    diag_function = MissingNodeJunctionsCheck()

    check_result = diag_function.diagnostic(net)
    diag_results = {check_function: check_result} if check_result else {}

    assert list(diag_results[check_function]["sink"]["missing_junctions"]) == [9999]
    assert list(diag_results[check_function]["source"]["missing_junctions"]) == [9998]
    assert list(diag_results[check_function]["ext_grid"]["missing_junctions"]) == [9997]

    assert list(diag_results[check_function]["sink"]["elements"].index) == [0]
    assert list(diag_results[check_function]["source"]["elements"].index) == [0]
    assert list(diag_results[check_function]["ext_grid"]["elements"].index) == [0]

    check_report_function(diag_function, None, diag_results.get(check_function, None),)



def test_missing_branch_junctions():
    test_nets = [simple_gas_grid(), multi_pump_dh_network()]
    check_function = "missing_branch_junctions"

    changes = {
        "pipe": (9991, 9992),
        "valve": (9993, 9994),
        "compressor": (9995, 9996),
        "pump": (9997, 9998),
        "heat_exchanger": (9999, 10000),
        "circ_pump_pressure": (10001, 10002),
        "circ_pump_mass": (10003, 10004),
    }

    for net in test_nets:
        expected = {}

        for element, (missing_from, missing_to) in changes.items():
            if not hasattr(net, element):
                continue

            table = net[element]

            if table.empty:
                continue

            if "from_junction" not in table.columns or "to_junction" not in table.columns:
                continue

            idx = table.index[0]

            net[element].at[idx, "from_junction"] = missing_from
            net[element].at[idx, "to_junction"] = missing_to

            expected[element] = {
                "missing_from_junctions": [missing_from],
                "missing_to_junctions": [missing_to],
            }

        if not expected:
            continue

        diag_function = MissingBranchJunctionsCheck()

        check_result = diag_function.diagnostic(net)
        diag_results = {
            check_function: check_result
        } if check_result else {}

        for element, values in expected.items():
            assert list(
                diag_results[check_function][element]["missing_from_junctions"]
            ) == values["missing_from_junctions"]

            assert list(
                diag_results[check_function][element]["missing_to_junctions"]
            ) == values["missing_to_junctions"]

        check_report_function(diag_function, None, diag_results.get(check_function, None),)



def test_pipe_diameter(diag_params):
    net = simple_gas_grid()
    diag_function = PipeDiameterCheck()

    check_result = diag_function.diagnostic(net, **diag_params)

    assert check_result is None
    check_report_function(diag_function, None, check_result)


    net = simple_gas_grid()
    net.pipe["inner_diameter_mm"] = 0.1

    diag_function = PipeDiameterCheck()

    check_result = diag_function.diagnostic(net, **diag_params)

    assert check_result == True
    check_report_function(diag_function, None, check_result)


    net = multi_pump_dh_network()
    net.pipe["inner_diameter_mm"] = 0.1

    diag_function = PipeDiameterCheck()

    check_result = diag_function.diagnostic(net, **diag_params)

    assert check_result == True
    check_report_function(diag_function, None, check_result)


    net = simple_gas_grid()
    net.pipe["inner_diameter_mm"] = 0.1
    net.sink.mdot_kg_per_s *= 1e8

    diag_function = PipeDiameterCheck()

    check_result = diag_function.diagnostic(net, **diag_params)

    assert check_result == False
    check_report_function(diag_function, None, check_result)


    net = multi_pump_dh_network()
    net.pipe["inner_diameter_mm"] = 0.1
    net.circ_pump_mass.mdot_flow_kg_per_s = 1000

    diag_function = PipeDiameterCheck()

    check_result = diag_function.diagnostic(net, **diag_params)

    assert check_result == False
    check_report_function(diag_function, None, check_result)


def test_valve_configuration():
    check_function = "valve_configuration"


    # Original network already converges
    net = simple_gas_grid()

    diag_function = ValveConfigurationCheck()
    check_result = diag_function.diagnostic(net)

    assert check_result is None
    check_report_function(diag_function, None, check_result)


    # Opening all valves makes the network converge
    net = simple_gas_grid()
    net.valve.opened = False

    diag_function = ValveConfigurationCheck()

    def fake_pipeflow_open_valves_help(net_arg, **kwargs):
        if net_arg.valve.opened.all():
            net_arg.converged = True
            return

        raise PipeflowNotConverged()

    with patch(
        "pandapipes.pipeflow",
        side_effect=fake_pipeflow_open_valves_help,
    ):
        check_result = diag_function.diagnostic(net)

    diag_results = {
        check_function: check_result
    }

    assert diag_results[check_function] == {
        "all_open": True,
        "all_closed": False,
    }

    check_report_function(
        diag_function,
        None,
        diag_results.get(check_function, None),
    )


    # Closing all valves makes the network converge
    net = simple_gas_grid()
    net.valve.opened = True

    diag_function = ValveConfigurationCheck()

    def fake_pipeflow_closed_valves_help(net_arg, **kwargs):
        if not net_arg.valve.opened.any():
            net_arg.converged = True
            return

        raise PipeflowNotConverged()

    with patch(
        "pandapipes.pipeflow",
        side_effect=fake_pipeflow_closed_valves_help,
    ):
        check_result = diag_function.diagnostic(net)

    diag_results = {
        check_function: check_result
    }

    assert diag_results[check_function] == {
        "all_open": False,
        "all_closed": True,
    }

    check_report_function(
        diag_function,
        None,
        diag_results.get(check_function, None),
    )


    # Neither opening nor closing all valves solves the problem
    net = simple_gas_grid()

    net.valve.opened = False
    net.sink.mdot_kg_per_s *= 1e8

    diag_function = ValveConfigurationCheck()

    check_result = diag_function.diagnostic(net)

    diag_results = {
        check_function: check_result
    }

    assert diag_results[check_function] == {
        "all_open": False,
        "all_closed": False,
    }

    check_report_function(
        diag_function,
        None,
        diag_results.get(check_function, None),
    )

def test_junction_height():
    diag_function = JunctionHeightCheck()


    net = simple_gas_grid()
    net.junction.loc[3, "height_m"] = 10000

    check_result = diag_function.diagnostic(net)
    assert check_result == True

    check_report_function(diag_function, None, check_result,)


    net = simple_gas_grid()
    net.junction.loc[3, "height_m"] = 10000
    net.sink.mdot_kg_per_s *= 1e4

    check_result = diag_function.diagnostic(net)
    assert check_result == False

    check_report_function(diag_function, None, check_result,)


def test_pipe_length_check(diag_params):
    net = simple_gas_grid()
    diag_function = PipeLengthCheck()

    check_result = diag_function.diagnostic(net, **diag_params)

    assert check_result is None
    check_report_function(diag_function, None, check_result)


    net = simple_gas_grid()
    net.pipe.loc[net.pipe.index[0], "length_km"] = 1000

    diag_function = PipeLengthCheck()

    check_result = diag_function.diagnostic(net, **diag_params)

    assert check_result == {
        "long_pipes": True,
        "all_pipes": None,
    }
    check_report_function(diag_function, None, check_result)


    net = simple_gas_grid()
    net.pipe.loc[net.pipe.index[0], "length_km"] = 1000

    diag_function = PipeLengthCheck()

    def fake_pipeflow_all_pipes_help(net_arg, **kwargs):
        if (
            net_arg.pipe["length_km"]
            == diag_params["standard_pipe_length_km"]
        ).all():
            net_arg.converged = True
            return

        raise PipeflowNotConverged()

    with patch(
        "pandapipes.pipeflow",
        side_effect=fake_pipeflow_all_pipes_help,
    ):
        check_result = diag_function.diagnostic(net, **diag_params)

    assert check_result == {
        "long_pipes": False,
        "all_pipes": True,
    }
    check_report_function(diag_function, None, check_result)


    net = simple_gas_grid()
    net.pipe.loc[net.pipe.index[0], "length_km"] = 1000
    net.sink.mdot_kg_per_s *= 1e8

    diag_function = PipeLengthCheck()

    check_result = diag_function.diagnostic(net, **diag_params)

    assert check_result == {
        "long_pipes": False,
        "all_pipes": False,
    }
    check_report_function(diag_function, None, check_result)

def test_pipe_roughness(diag_params):

    net = multi_pump_dh_network()

    diag_function = PipeRoughnessCheck()
    check_result = diag_function.diagnostic(net, **diag_params)

    assert check_result is None

    check_report_function(diag_function, None, check_result)


    net = multi_pump_dh_network()
    net.pipe["k_mm"] = 2.0

    diag_function = PipeRoughnessCheck()

    check_result = diag_function.diagnostic(net, **diag_params)

    assert check_result == {
        "rough_pipes": list(net.pipe.index),
        "highest_k_mm": 2.0,
    }

    check_report_function(diag_function, None, check_result)


def test_circ_pump_mass_flow(diag_params):

    net = multi_pump_dh_network()
    diag_function = CircPumpMassFlowCheck()

    check_result = diag_function.diagnostic(net, **diag_params)

    assert check_result is None
    check_report_function(diag_function, None, check_result)

    net = multi_pump_dh_network()
    net.circ_pump_mass.mdot_flow_kg_per_s = 10

    diag_function = CircPumpMassFlowCheck()

    check_result = diag_function.diagnostic(net, **diag_params)

    assert check_result == True
    check_report_function(diag_function, None, check_result)

    net = multi_pump_dh_network()
    net.circ_pump_mass.mdot_flow_kg_per_s = 10
    net.pipe["inner_diameter_mm"] = 0.1

    diag_function = CircPumpMassFlowCheck()

    check_result = diag_function.diagnostic(net, **diag_params)

    assert check_result == False
    check_report_function(diag_function, None, check_result)


def test_compressor_pressure_ratio(diag_params):

    net = gas_grid_with_compressor_pressure_control()

    diag_function = CompressorPressureRatioCheck()

    check_result = diag_function.diagnostic(net, **diag_params)

    assert check_result is None
    check_report_function(diag_function, None, check_result)

    net = gas_grid_with_compressor_pressure_control()
    net.compressor.pressure_ratio = 10

    diag_function = CompressorPressureRatioCheck()

    check_result = diag_function.diagnostic(net, **diag_params)

    assert check_result is True
    check_report_function(diag_function, None, check_result)



def test_heat_transfer_coefficient(diag_params):

    # Original network converges -> check is not required
    net = multi_pump_dh_network()
    diag_function = HeatTransferCoefficientCheck()

    check_result = diag_function.diagnostic(net, **diag_params)

    assert check_result is None
    check_report_function(diag_function, None, check_result)


    # High heat-transfer coefficient causes non-convergence.
    # Reducing it by factor 0.1 makes the network converge.
    net = multi_pump_dh_network()
    net.pipe["u_w_per_m2k"] = 10

    diag_function = HeatTransferCoefficientCheck()

    check_result = diag_function.diagnostic(net, **diag_params)

    assert check_result == True
    check_report_function(diag_function, None, check_result)


    # Reducing the heat-transfer coefficient is not sufficient,
    # because the extremely small pipe diameter still prevents convergence.
    net = multi_pump_dh_network()
    net.pipe["u_w_per_m2k"] = 10
    net.pipe["inner_diameter_mm"] = 0.1

    diag_function = HeatTransferCoefficientCheck()

    check_result = diag_function.diagnostic(net, **diag_params)

    assert check_result is False
    check_report_function(diag_function, None, check_result)



def test_alpha_sweep(diag_params):

    # Original network already converges
    net = simple_gas_grid()
    diag_function = AlphaSweepCheck()

    check_result = diag_function.diagnostic(net, **diag_params)

    assert check_result is None
    check_report_function(diag_function, None, check_result)


    # Original calculation fails, but alpha = 0.5 converges
    net = simple_gas_grid()
    diag_function = AlphaSweepCheck()

    def fake_pipeflow_success(net_arg, **kwargs):
        alpha = kwargs.get("alpha")

        if alpha == 0.5:
            net_arg.converged = True
            return

        raise PipeflowNotConverged()

    with patch(
        "pandapipes.pipeflow",
        side_effect=fake_pipeflow_success,
    ):
        check_result = diag_function.diagnostic(net, **diag_params)

    assert check_result is True
    assert diag_function.successful_alpha == 0.5
    assert 0.5 in diag_function.alphas

    check_report_function(diag_function, None, check_result)


    # No alpha value causes convergence
    net = simple_gas_grid()
    diag_function = AlphaSweepCheck()

    with patch(
        "pandapipes.pipeflow",
        side_effect=PipeflowNotConverged(),
    ):
        check_result = diag_function.diagnostic(net, **diag_params)

    assert check_result is False
    assert diag_function.successful_alpha is None
    assert diag_function.alphas == [
        1.0, 0.1, 0.9, 0.2, 0.8,
        0.3, 0.7, 0.4, 0.6, 0.5,
    ]

    check_report_function(diag_function, None, check_result)


def test_inactive_pressure_controls():

    # Original network converges -> check is not required
    net = gas_grid_with_compressor_pressure_control()
    diag_function = InactivePressureControlsCheck()

    check_result = diag_function.diagnostic(net)

    assert check_result is None
    check_report_function(diag_function, None, check_result)


    # Active pressure control with an unrealistic target causes non-convergence.
    # Deactivating the pressure control makes the network converge.
    net = gas_grid_with_compressor_pressure_control()
    net.press_control.controlled_p_bar = 100000
    net.press_control.control_active = True

    diag_function = InactivePressureControlsCheck()

    check_result = diag_function.diagnostic(net)

    assert check_result == True
    check_report_function(diag_function, None, check_result)


    # Even after deactivating the pressure control,
    # another severe network problem still prevents convergence.
    net = gas_grid_with_compressor_pressure_control()
    net.press_control.controlled_p_bar = 100000
    net.press_control.control_active = True
    net.pipe["inner_diameter_mm"] = 0.1

    diag_function = InactivePressureControlsCheck()

    check_result = diag_function.diagnostic(net)

    assert check_result is False
    check_report_function(diag_function, None, check_result)

if __name__ == "__main__":
    pytest.main([__file__, "-xs"])
