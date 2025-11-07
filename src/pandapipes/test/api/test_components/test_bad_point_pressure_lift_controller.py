import numpy as np
import pytest
import pandapipes as pp
from pandapipes.control import BadPointPressureLiftController

@pytest.fixture
def district_heating_net():
    """Create a simple district heating network with pump and heat consumers."""
    net = pp.create_empty_network(fluid="water")

    supply_temperature_k = 85 + 273.15
    pipetype = "110/202 PLUS"

    # Create ring network
    j1 = pp.create_junction(net, pn_bar=1.05, tfluid_k=supply_temperature_k, geodata=(0, 10))
    j2 = pp.create_junction(net, pn_bar=1.05, tfluid_k=supply_temperature_k, geodata=(0, 0))
    j3 = pp.create_junction(net, pn_bar=1.05, tfluid_k=supply_temperature_k, geodata=(10, 0))
    j4 = pp.create_junction(net, pn_bar=1.05, tfluid_k=supply_temperature_k, geodata=(60, 0))
    j5 = pp.create_junction(net, pn_bar=1.05, tfluid_k=supply_temperature_k, geodata=(85, 0))
    j6 = pp.create_junction(net, pn_bar=1.05, tfluid_k=supply_temperature_k, geodata=(85, 10))
    j7 = pp.create_junction(net, pn_bar=1.05, tfluid_k=supply_temperature_k, geodata=(60, 10))
    j8 = pp.create_junction(net, pn_bar=1.05, tfluid_k=supply_temperature_k, geodata=(10, 10))

    pp.create_circ_pump_const_pressure(net, j1, j2, p_flow_bar=4, plift_bar=1.5,
                                       t_flow_k=supply_temperature_k, type="auto")

    pp.create_pipe(net, j2, j3, std_type=pipetype, length_km=0.01, k_mm=0.1, sections=5, text_k=283)
    pp.create_pipe(net, j3, j4, std_type=pipetype, length_km=0.05, k_mm=0.1, sections=5, text_k=283)
    pp.create_pipe(net, j4, j5, std_type=pipetype, length_km=0.025, k_mm=0.1, sections=5, text_k=283)
    pp.create_pipe(net, j6, j7, std_type=pipetype, length_km=0.25, k_mm=0.1, sections=5, text_k=283)
    pp.create_pipe(net, j7, j8, std_type=pipetype, length_km=0.05, k_mm=0.1, sections=5, text_k=283)
    pp.create_pipe(net, j8, j1, std_type=pipetype, length_km=0.01, k_mm=0.1, sections=5, text_k=283)

    pp.create_heat_consumer(net, j5, j6, loss_coefficient=0, qext_w=100000, treturn_k=328.15)
    pp.create_heat_consumer(net, j4, j7, loss_coefficient=0, qext_w=200000, treturn_k=333.15)

    pp.pipeflow(net, mode="bidirectional", iter=100)
    return net

def test_bad_point_pressure_lift_controller(district_heating_net):
    """Test that controller maintains target pressure difference at worst point."""
    net = district_heating_net
    target_dp = 1.5
    tolerance = 0.1

    # Add controller to network
    controller = BadPointPressureLiftController(net, target_dp_min_bar=target_dp,
                                                tolerance=tolerance, proportional_gain=0.3)
    net.controller.loc[len(net.controller)] = [controller, True, -1, -1, False, False]

    # Run pipeflow with control
    for _ in range(5):
        pp.pipeflow(net, mode="bidirectional", iter=100)
        if controller.is_converged(net):
            break

    # Verify convergence and target achievement
    dp_min, worst_point_idx = controller.calculate_worst_point(net)

    assert net.converged, "Network should converge with controller"
    assert worst_point_idx >= 0, "Controller should identify worst point"
    assert abs(dp_min - target_dp) < tolerance, \
        f"Controller should reach target {target_dp} bar, got {dp_min:.3f} bar"

def test_bad_point_controller_standby_mode(district_heating_net):
    """Test that controller enters standby mode when no heat demand is present."""
    net = district_heating_net

    # Remove heat demand
    net.heat_consumer["qext_w"] = 0

    controller = BadPointPressureLiftController(net, target_dp_min_bar=1.5,
                                                min_plift=1.5, min_pflow=3.5)
    controller.control_step(net)

    # Verify standby mode sets minimum pressures
    assert net.circ_pump_pressure["plift_bar"].iloc[0] == 1.5
    assert net.circ_pump_pressure["p_flow_bar"].iloc[0] == 3.5

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
