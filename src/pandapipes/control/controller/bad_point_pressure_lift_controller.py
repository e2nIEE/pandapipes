from pandapower.control.basic_controller import BasicCtrl

class BadPointPressureLiftController(BasicCtrl):
    """
    A controller for maintaining the pressure difference at the worst point 
    (German: Differenzdruckregelung im Schlechtpunkt) in the network.
    
    The BadPointPressureLiftController is a custom controller designed for district heating networks 
    modeled with pandapipes. Its main purpose is to maintain a minimum pressure difference at the 
    network's "worst point"—the heat exchanger with the lowest pressure difference (Schlechtpunktregelung).

    Key Features:

    - **Automatic Worst Point Detection:** Identifies the heat exchanger with the lowest pressure difference where heat flow is present.
    - **Pressure Regulation:** Adjusts the circulation pump's lift and flow pressures to ensure the pressure difference at the worst point meets a specified minimum target.
    - **Proportional Control:** Uses a proportional gain to determine the adjustment magnitude based on the deviation from the target pressure difference.
    - **Control Modes:** Supports two operating modes - fixed flow pressure (adjusts only plift) or fixed return pressure (adjusts both plift and pflow).
    - **Standby Mode:** If no heat flow is detected, the controller switches the pump to a standby mode with minimum lift and flow pressures.
    - **Convergence Check:** Determines if the pressure difference is within a specified tolerance of the target, signaling convergence.

    :param net: The pandapipes network
    :type net: pandapipesNet
    :param circ_pump_pressure_idx: Index of the circulation pump, defaults to 0
    :type circ_pump_pressure_idx: int, optional
    :param target_dp_min_bar: Target minimum pressure difference in bar, defaults to 1
    :type target_dp_min_bar: float, optional
    :param tolerance: Tolerance for pressure difference, defaults to 0.2
    :type tolerance: float, optional
    :param proportional_gain: Proportional gain for the controller, defaults to 0.2
    :type proportional_gain: float, optional
    :param mode: Control mode - 'fixed_pflow' (adjusts only plift, keeps p_flow constant) or 'fixed_preturn' (adjusts both plift and pflow, keeps p_return constant), defaults to 'fixed_preturn'
    :type mode: str, optional
    :param min_plift: Minimum lift pressure in bar, defaults to 1.5
    :type min_plift: float, optional
    :param min_pflow: Minimum flow pressure in bar, defaults to 3.5
    :type min_pflow: float, optional
    :param min_preturn: Minimum return pressure in bar (for fixed_pflow mode), defaults to 2.0
    :type min_preturn: float, optional
    :param kwargs: Additional keyword arguments
    :type kwargs: dict, optional
    """
    def __init__(self, net, circ_pump_pressure_idx=0, target_dp_min_bar=1, tolerance=0.2,
                 proportional_gain=0.2, mode='fixed_preturn', min_plift=1.5, min_pflow=3.5,
                 min_preturn=2.0, **kwargs):
        super(BadPointPressureLiftController, self).__init__(net, **kwargs)
        self.circ_pump_pressure_idx = circ_pump_pressure_idx
        self.target_dp_min_bar = target_dp_min_bar
        self.tolerance = tolerance
        self.proportional_gain = proportional_gain

        if mode not in ['fixed_pflow', 'fixed_preturn']:
            raise ValueError("mode must be either 'fixed_pflow' or 'fixed_preturn'")
        self.mode = mode

        self.min_plift = min_plift  # Minimum lift pressure in bar
        self.min_pflow = min_pflow  # Minimum flow pressure in bar
        self.min_preturn = min_preturn  # Minimum return pressure in bar

        self.iteration = 0  # Add iteration counter

        self.dp_min, self.heat_consumer_idx = self.calculate_worst_point(net)

    def calculate_worst_point(self, net):
        """
        Calculate the worst point in the heating network, defined as the heat exchanger 
        with the lowest pressure difference.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :return: Tuple of (minimum pressure difference, index of worst point)
        :rtype: tuple(float, int)
        """
        # Calculate pressure difference for all heat consumers with heat flow
        diff = net.res_heat_consumer["p_from_bar"] - net.res_heat_consumer["p_to_bar"]

        # Filter out heat consumers with no heat flow
        qext = net.heat_consumer["qext_w"]
        active_consumers = qext != 0

        if not active_consumers.any():
            return 0, -1

        # Find the minimum delta p where the heat flow is not zero
        diff_active = diff[active_consumers]
        dp_min = diff_active.min()
        idx_min = diff_active.idxmin()

        return dp_min, idx_min

    def time_step(self, net, time_step):
        """
        Reset the iteration counter at the start of each time step.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param time_step: The current time step
        :type time_step: int
        :return: The current time step
        :rtype: int
        """
        self.iteration = 0  # reset iteration counter
        self.dp_min, self.heat_consumer_idx = self.calculate_worst_point(net)

        return time_step

    def is_converged(self, net):
        """
        Check if the controller has converged.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :return: True if converged, False otherwise
        :rtype: bool
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

        :param net: The pandapipes network
        :type net: pandapipesNet
        """
        # Increment iteration counter
        self.iteration += 1

        # Adjust the pump pressure or switch to standby mode when heat flow is zero
        if all(net.heat_consumer["qext_w"] == 0):
            # Switch to standby mode
            print("No heat flow detected. Switching to standby mode.")
            net.circ_pump_pressure.at[self.circ_pump_pressure_idx, "plift_bar"] = self.min_plift  # Minimum lift pressure
            net.circ_pump_pressure.at[self.circ_pump_pressure_idx, "p_flow_bar"] = self.min_pflow  # Minimum flow pressure
            return super(BadPointPressureLiftController, self).control_step(net)

        # Check whether the heat flow in the heat exchanger is zero
        current_dp_bar = net.res_heat_consumer["p_from_bar"].at[self.heat_consumer_idx] - \
                        net.res_heat_consumer["p_to_bar"].at[self.heat_consumer_idx]
        current_plift_bar = net.circ_pump_pressure["plift_bar"].at[self.circ_pump_pressure_idx]
        current_pflow_bar = net.circ_pump_pressure["p_flow_bar"].at[self.circ_pump_pressure_idx]

        dp_error = self.target_dp_min_bar - current_dp_bar

        if self.mode == 'fixed_pflow':
            # Mode 1: Keep p_flow constant, adjust only plift
            plift_adjustment = dp_error * self.proportional_gain
            new_plift = current_plift_bar + plift_adjustment

            # Check if p_return would fall below minimum
            new_preturn = current_pflow_bar - new_plift
            if new_preturn < self.min_preturn:
                new_plift = current_pflow_bar - self.min_preturn

            net.circ_pump_pressure["plift_bar"].at[self.circ_pump_pressure_idx] = new_plift
        else:  # fixed_preturn
            # Mode 2: Keep p_return constant, adjust both plift and pflow
            plift_adjustment = dp_error * self.proportional_gain
            pflow_adjustment = dp_error * self.proportional_gain
            new_plift = current_plift_bar + plift_adjustment
            new_pflow = current_pflow_bar + pflow_adjustment
            net.circ_pump_pressure["plift_bar"].at[self.circ_pump_pressure_idx] = new_plift
            net.circ_pump_pressure["p_flow_bar"].at[self.circ_pump_pressure_idx] = new_pflow

        return super(BadPointPressureLiftController, self).control_step(net)