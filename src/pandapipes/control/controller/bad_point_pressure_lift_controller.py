from pandapower.control.basic_controller import BasicCtrl

class BadPointPressureLiftController(BasicCtrl):
    """
    A controller for maintaining the pressure difference at the worst point (German: Differenzdruckregelung im Schlechtpunkt) in the network.
    
    The BadPointPressureLiftController is a custom controller designed for district heating networks 
    modeled with pandapipes. Its main purpose is to maintain a minimum pressure difference at the 
    network's "worst point"—the heat exchanger with the lowest pressure difference (Schlechtpunktregelung).

    Key Features:
    - **Automatic Worst Point Detection:** Identifies the heat exchanger with the lowest pressure difference where heat flow is present.
    - **Pressure Regulation:** Adjusts the circulation pump's lift and flow pressures to ensure the pressure difference at the worst point meets a specified minimum target.
    - **Proportional Control:** Uses a proportional gain to determine the adjustment magnitude based on the deviation from the target pressure difference.
    - **Standby Mode:** If no heat flow is detected, the controller switches the pump to a standby mode with minimum lift and flow pressures.
    - **Convergence Check:** Determines if the pressure difference is within a specified tolerance of the target, signaling convergence.

    Args:
        net (pandapipesNet): The pandapipes network.
        circ_pump_pressure_idx (int, optional): Index of the circulation pump. Defaults to 0.
        target_dp_min_bar (float, optional): Target minimum pressure difference in bar. Defaults to 1.
        tolerance (float, optional): Tolerance for pressure difference. Defaults to 0.2.
        proportional_gain (float, optional): Proportional gain for the controller. Defaults to 0.2.
        min_plift (float, optional): Minimum lift pressure in bar. Defaults to 1.5.
        min_pflow (float, optional): Minimum flow pressure in bar. Defaults to 3.5.
        **kwargs: Additional keyword arguments.
    """
    def __init__(self, net, circ_pump_pressure_idx=0, target_dp_min_bar=1, tolerance=0.2,
                 proportional_gain=0.2, min_plift=1.5, min_pflow=3.5, **kwargs):
        super(BadPointPressureLiftController, self).__init__(net, **kwargs)
        self.circ_pump_pressure_idx = circ_pump_pressure_idx
        self.target_dp_min_bar = target_dp_min_bar
        self.tolerance = tolerance
        self.proportional_gain = proportional_gain

        self.min_plift = min_plift  # Minimum pressure in bar
        self.min_pflow = min_pflow  # Minimum lift pressure in bar

        self.iteration = 0  # Add iteration counter

        self.dp_min, self.heat_consumer_idx = self.calculate_worst_point(net)

    def calculate_worst_point(self, net):
        """
        Calculate the worst point in the heating network, defined as the heat exchanger with the lowest pressure difference.

        Args:
            net (pandapipesNet): The pandapipes network.

        Returns:
            tuple: The minimum pressure difference and the index of the worst point.
        """

        dp = []

        for idx, qext, p_from, p_to in zip(net.heat_consumer.index, net.heat_consumer["qext_w"], 
                                           net.res_heat_consumer["p_from_bar"], net.res_heat_consumer["p_to_bar"]):
            if qext != 0:
                dp_diff = p_from - p_to
                dp.append((dp_diff, idx))

        if not dp:
            return 0, -1

        # Find the minimum delta p where the heat flow is not zero
        dp_min, idx_min = min(dp, key=lambda x: x[0])

        return dp_min, idx_min

    def time_step(self, net, time_step):
        """
        Reset the iteration counter at the start of each time step.

        Args:
            net (pandapipesNet): The pandapipes network.
            time_step (int): The current time step.

        Returns:
            int: The current time step.
        """
        self.iteration = 0  # reset iteration counter
        self.dp_min, self.heat_consumer_idx = self.calculate_worst_point(net)

        return time_step

    def is_converged(self, net):
        """
        Check if the controller has converged.

        Args:
            net (pandapipesNet): The pandapipes network.

        Returns:
            bool: True if converged, False otherwise.
        """

        if all(net.heat_consumer["qext_w"] == 0):
            return True
        
        current_dp_bar = net.res_heat_consumer["p_from_bar"].at[self.heat_consumer_idx] - \
                        net.res_heat_consumer["p_to_bar"].at[self.heat_consumer_idx]

        # Check if the pressure difference is within tolerance
        dp_within_tolerance = abs(current_dp_bar - self.target_dp_min_bar) < self.tolerance

        if dp_within_tolerance:
            return dp_within_tolerance

    def control_step(self, net):
        """
        Adjust the pump pressure to maintain the target pressure difference.

        Args:
            net (pandapipesNet): The pandapipes network.
        """
        # Increment iteration counter
        self.iteration += 1

        # Adjust the pump pressure or switch to standby mode when heat flow is zero
        if all(net.heat_consumer["qext_w"] == 0):
            # Switch to standby mode
            print("No heat flow detected. Switching to standby mode.")
            net.circ_pump_pressure["plift_bar"].iloc[:] = self.min_plift  # Minimum lift pressure
            net.circ_pump_pressure["p_flow_bar"].iloc[:] = self.min_pflow  # Minimum flow pressure
            return super(BadPointPressureLiftController, self).control_step(net)

        # Check whether the heat flow in the heat exchanger is zero
        current_dp_bar = net.res_heat_consumer["p_from_bar"].at[self.heat_consumer_idx] - \
                        net.res_heat_consumer["p_to_bar"].at[self.heat_consumer_idx]
        current_plift_bar = net.circ_pump_pressure["plift_bar"].at[self.circ_pump_pressure_idx]
        current_pflow_bar = net.circ_pump_pressure["p_flow_bar"].at[self.circ_pump_pressure_idx]

        dp_error = self.target_dp_min_bar - current_dp_bar

        plift_adjustment = dp_error * self.proportional_gain
        pflow_adjustment = dp_error * self.proportional_gain

        new_plift = current_plift_bar + plift_adjustment
        new_pflow = current_pflow_bar + pflow_adjustment

        net.circ_pump_pressure["plift_bar"].at[self.circ_pump_pressure_idx] = new_plift
        net.circ_pump_pressure["p_flow_bar"].at[self.circ_pump_pressure_idx] = new_pflow

        return super(BadPointPressureLiftController, self).control_step(net)