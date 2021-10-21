import numpy as np
import pandas as pd
from numpy.linalg import inv


class StratThermStor():
    def __init__(self, init_strata_temp_c, flag_input, initial_input_source, initial_input_sink,
                 mdot_source_max_kg_per_s, mdot_sink_max_kg_per_s, delta_t_s, tank_height_mm=1700, tank_diameter_mm=810,
                 wall_thickness_mm=160, source_ind=(0, -1), sink_ind=(-1, 0), tol=1e-6):
        """
        Model of a stratified thermal storage. Heat medium: water.
        Heat capacity c_p and density are assumed to be constant.

        @param init_strata_temp_c: Each stratum's temperature in Celsius from highest to lowest stratum.
                                   Also defines the number of strata.
        @type init_strata_temp_c: numpy.ndarray
        @param flag_input: Containing two flags of 'hf' (heat flow), 'mf' (mass flow) and 'ct' (circuit temperature)
                           defining for both source and sink circuit:
                           the parameter set on initialization of the storage object to the values of the input
                           parameters 'initial_input_source' and 'initial_input_sink' (first flag).
                           the parameter whose value is to be defined for each time step (second flag).
                           The unused flag is the flag of the parameter that will be calculated from the other two.
        @type flag_input: tuple of str
        @param initial_input_source: Value of the potential input parameter for the source circuit that will be constant
                                     during the simulation, defined by first element in 'flag_input'. Possibilities:
                                     Heat flow: The heat injected to the storage from the source circuit [W]
                                     Mass flow: The mass of water injected to the storage from the source circuit [kg/s]
                                     Circuit temperature: The temperature of water entering the storage
                                                          from the source circuit [°C]
        @type initial_input_source: float
        @param initial_input_sink: Value of the potential input parameter for the sink circuit that will be constant
                                   during the simulation, defined by first element in 'flag_input'. Possibilities:
                                   Heat flow: The heat withdrawn from the storage by the sink circuit [W]
                                   Mass flow: The mass of water withdrawn from the storage by the sink circuit [kg/s]
                                   Circuit temperature: The temperature of water entering the storage
                                                        from the sink circuit [°C]
        @type initial_input_sink: float
        @param mdot_source_kg_per_ts: Mass flow from source circuit in kg per time step.
        @type mdot_source_kg_per_ts: float
        @param mdotl: Mass flow from sink circuit in kg per time step.
        @type mdotl: float
        @param delta_t_s: Time step size.
        @type delta_t_s: float
        @param tank_height_mm: Inner height of the thermal storage tank in meter.
        @type tank_height_mm: float
        @param tank_diameter_mm: Inner diameter of the thermal storage tank in meter.
        @type tank_diameter_mm: float
        @param wall_thickness_mm: Diameter of the tank's wall.
        @type wall_thickness_mm: float
        @param source_ind: Indices of the strata with first the storage's inlet from and second the storage's outlet to 
                           the source circuit.
        @type source_ind: tuple(int, int)
        @param sink_ind: Indices of the strata with first the storage's inlet from and second the storage's outlet to 
                        the sink circuit.
        @type sink_ind: tuple(int, int)
        @param tol: Tolerance of error in iteration.
        @type tol: float
        """
        # tank measures
        self.strata = len(init_strata_temp_c)  # number of strata
        self.z_m = tank_height_mm / 1000 / self.strata  # height of each stratum
        self.A_m2 = np.pi * (tank_diameter_mm / 1000 - 2 * wall_thickness_mm / 1000) ** 2 / 4  # cross section area
        self.A_ext_m2 = np.pi * tank_diameter_mm / 1000 * self.z_m   # barrel area of one stratum
        self.m_strat_kg = self.A_m2 * self.z_m * 994.3025  # each stratum's mass

        # heat medium and tank properties
        self.c_p_w_s_per_kg_k, self.k_w_per_m2_k, self.lambda_eff_w_per_m_k = 4180, .5, 1.5  # bei cp = 9489 ähnlich
        # k = 0.5 set to 0 for comparision with parantapas model  # toDo: Typo in Mail so .5 is correct?

        # parameters: ambient air temperature; temperature, mass flow and inlet position of source and sink circuit
        self.t_amb_c = 20
        self.flag_input = flag_input
        self.t_source_c, self.t_sink_c,  self.q_source_w, self.q_sink_w, \
        self.mdot_source_kg_per_ts, self.mdot_sink_kg_per_ts, self.mdot_kg_per_ts = None, None, None, None, None, None, None
        message = "The input parameter 'flag_input' must be a tuple with two of the strings: 'hf', 'mf' and 'ct'."
        if self.flag_input[0] == 'hf':
            self.q_source_w, self.q_sink_w = initial_input_source, initial_input_sink
            if self.flag_input[1] == 'mf':
                self.initialize_time_step = self.calculate_temperature_spread_with_mass_flow_input
            elif self.flag_input[1] == 'ct':
                self.initialize_time_step = self.calculate_mass_flow_with_temperature_input
            else:
                raise ValueError(message)
        elif self.flag_input[0] == 'mf':
            self.mdot_source_kg_per_ts, self.mdot_sink_kg_per_ts = initial_input_source, initial_input_sink
            self.mdot_kg_per_ts = self.mdot_source_kg_per_ts - self.mdot_sink_kg_per_ts
            if self.flag_input[1] == 'hf':
                self.initialize_time_step = self.calculate_temperature_spread_with_heat_flow_input
            elif self.flag_input[1] == 'ct':
                self.initialize_time_step = self.calculate_heat_flow_with_temperature_input
            else:
                raise ValueError(message)
        elif self.flag_input[0] == 'ct':
            self.t_source_c, self.t_sink_c = initial_input_source, initial_input_sink
            if self.flag_input[1] == 'hf':
                self.initialize_time_step = self.calculate_mass_flow_with_heat_flow_input
            elif self.flag_input[1] == 'mf':
                self.initialize_time_step = self.calculate_heat_flow_with_mass_flow_input
            else:
                raise ValueError(message)
        else:
            raise ValueError(message)
        self.mdot_source_max_kg_per_ts, self.mdot_sink_max_kg_per_ts = mdot_source_max_kg_per_s * delta_t_s, \
                                                                       mdot_sink_max_kg_per_s * delta_t_s
        self.so_inl_pos, self.si_inl_pos = np.zeros((self.strata,)), np.zeros((self.strata,))
        self.so_inl_pos[source_ind[0]], self.si_inl_pos[sink_ind[0]] = 1, 1
        self.so_outl_ind, self.si_outl_ind = source_ind[1], sink_ind[1]  # ToDo doesn't it need to be the other way round?

        # calculation parameters
        self.tol = tol
        self.alpha = 1  # ToDo: Was genau macht alpha?
        self.itr = 0
        self.count = 1

        # time parameters
        self.delta_t_s = delta_t_s

        # Array to store each stratum's temperature for previous time step i
        self.t_i_c = init_strata_temp_c + 0.  # + 0. needed to convert values to floats
        # Array to store each stratum's temperature for currently calculated time step i+1 from previous iteration
        self.t_ip1_old_c = self.t_i_c
        # Array to store each stratum's temperature for currently calculated time step i+1 when calculated in iteration
        self.t_ip1_new_c = self.t_ip1_old_c
        self.results = None

        self.results = pd.DataFrame(self.t_i_c)
        self.results.loc["source_revised"] = [False]
        self.results.loc["sink_revised"] = [False]

        self.F = np.copy(self.t_ip1_old_c)

    def get_delta(self, plus):
        if plus:
            return (1 + np.sign(self.mdot_kg_per_ts)) / 2
        else:
            return (1 - np.sign(self.mdot_kg_per_ts)) / 2

    def do_time_step(self, source, sink):
        flag_source, flag_sink = self.initialize_time_step(source, sink)
        self.iterate()
        self.t_i_c = self.t_ip1_new_c
        self.finalize_time_step(flag_source, flag_sink)

    def finalize_time_step(self, flag_source, flag_sink):
        t = len(self.results.loc[0])
        self.results[t] = np.append(self.t_i_c, [flag_source, flag_sink])

    def calculate_mass_flow(self, q_w, t_in_c, t_stratum_c, max):
        flag = False
        try:
            mdot_kg_per_ts = abs(q_w * self.delta_t_s / self.c_p_w_s_per_kg_k * (t_in_c - t_stratum_c))
            if mdot_kg_per_ts > max:
                print("The mass flow ist to big and set to its maximum value.")
                flag, mdot_kg_per_ts = True, max
        except ZeroDivisionError:
            print("Since the temperature doesn't change in the circuit, the circuit mass flow is set to 0.")
            flag, mdot_kg_per_ts = True, 0
        return flag, mdot_kg_per_ts

    def calculate_mass_flow_with_temperature_input(self, source, sink):
        flag_source, self.mdot_source_kg_per_ts = self.calculate_mass_flow(
            self.q_source_w, source, self.t_i_c[self.so_outl_ind], self.mdot_source_max_kg_per_ts)
        flag_sink, self.mdot_sink_kg_per_ts = self.calculate_mass_flow(
            self.q_sink_w, sink, self.t_i_c[self.si_outl_ind], self.mdot_sink_max_kg_per_ts)
        self.mdot_kg_per_ts = self.mdot_source_kg_per_ts - self.mdot_sink_kg_per_ts
        return flag_source, flag_sink

    def calculate_mass_flow_with_heat_flow_input(self, source, sink):
        flag_source, self.mdot_source_kg_per_ts = self.calculate_mass_flow(
            source, self.t_source_c, self.t_i_c[self.so_outl_ind], self.mdot_source_max_kg_per_ts)
        flag_sink, self.mdot_sink_kg_per_ts = self.calculate_mass_flow(
            sink, self.t_sink_c, self.t_i_c[self.si_outl_ind], self.mdot_sink_max_kg_per_ts)
        self.mdot_kg_per_ts = self.mdot_source_kg_per_ts - self.mdot_sink_kg_per_ts
        return flag_source, flag_sink

    def calculate_heat_flow(self, mdot_kg_per_s, t_in_c, t_stratum_c):
        return abs(mdot_kg_per_s * self.c_p_w_s_per_kg_k * (t_in_c - t_stratum_c))

    def calculate_heat_flow_with_mass_flow_input(self, source, sink):
        self.q_source_w = self.calculate_heat_flow(source, self.t_source_c, self.t_i_c[self.so_outl_ind])
        self.q_sink_w = self.calculate_heat_flow(sink, self.t_sink_c, self.t_i_c[self.si_outl_ind])
        return False, False

    def calculate_heat_flow_with_temperature_input(self, source, sink):
        self.q_source_w = self.calculate_heat_flow(self.mdot_source_kg_per_ts, source, self.t_i_c[self.so_outl_ind])
        self.q_sink_w = self.calculate_heat_flow(self.mdot_sink_kg_per_ts, sink, self.t_i_c[self.si_outl_ind])
        return False, False

    def calculate_temperature_spread(self, q_w, mdot_kg_per_s):
        try:
            return False, q_w / mdot_kg_per_s / self.c_p_w_s_per_kg_k
        except ZeroDivisionError:
            print("Since there is no mass flow, the temperature spread is set to 0.")
            return True, 0

    def calculate_temperature_spread_with_heat_flow_input(self, source, sink):
        flag_source, self.t_source_c = self.t_i_c[self.so_outl_ind] + \
                                       self.calculate_temperature_spread(source, self.mdot_source_kg_per_ts)
        flag_sink, self.t_sink_c = self.t_i_c[self.si_outl_ind] + \
                                   self.calculate_temperature_spread(sink, self.mdot_sink_kg_per_ts)
        return flag_source, flag_sink

    def calculate_temperature_spread_with_mass_flow_input(self, source, sink):
        flag_source, self.t_source_c = self.t_i_c[self.so_outl_ind] + self.calculate_temperature_spread(self.q_source_w,
                                                                                                        source)
        flag_sink, self.t_sink_c = self.t_i_c[self.si_outl_ind] + self.calculate_temperature_spread(self.q_sink_w, sink)
        return flag_source, flag_sink

    def jacobian(self, temps):
        """
        Calculate jacobian matrix for Newton Raphson.
        @param temps: Each stratum's temperature for next time step from top to bottom.
        @type temps: numpy.ndarray
        @return: Jacobian matrix.
        @rtype: numpy.ndarray
        """
        h = 1e-6  # Value the next time step's value should be changed by for Euler backwards method to calculate slope.
        J = np.zeros((self.strata, self.strata))  # Jacobian matrix is zero fot not adjacent strata.
        temps = np.append(temps, None)  # The temperature above highest and below lowest stratum should be read as None.

        for row, s, l in zip(range(self.strata), self.so_inl_pos, self.si_inl_pos):
            if row:  # should not be calculated for top stratum since there's no stratum above
                J[row, row - 1] = (self.f(self.t_i_c[row], temps[row], temps[row-1] + h, temps[row+1], s, l, not (row-1)) -
                                   self.f(self.t_i_c[row], temps[row], temps[row-1],     temps[row+1], s, l, not (row-1))) / h
            if row + 1 < self.strata:  # should not be calculated for bottom stratum since there's no stratum below
                J[row, row + 1] = (self.f(self.t_i_c[row], temps[row], temps[row-1], temps[row+1] + h, s, l) -
                                   self.f(self.t_i_c[row], temps[row], temps[row-1], temps[row+1],     s, l)) / h
            J[row, row] = (self.f(self.t_i_c[row], temps[row] + h, temps[row-1], temps[row+1], s, l, not row) -
                           self.f(self.t_i_c[row], temps[row],     temps[row-1], temps[row+1], s, l, not row)) / h

        return J

    def f(self, t_i_c, t_ip1_c, t_above_ip1_c, t_below_ip1_c, source=False, sink=False, top=False):
        """
        Differential equation to calculate a stratum's temperature in form 0 = ...

        @param t_i_c: Stratum's previous temperature
        @type t_i_c: float
        @param t_ip1_c: Stratum's next temperature
        @type t_ip1_c: float
        @param t_above_ip1_c: Next temperature of stratum above
        @type t_above_ip1_c: float or None
        @param t_below_ip1_c: Next temperature of stratum below
        @type t_below_ip1_c: float or None
        @param source: If there's an inlet from the source circuit in this stratum
        @type source: bool
        @param sink: If there's an inlet from the sink circuit in this stratum
        @type sink: bool
        @param top: If this is the top stratum so the heat transferring area is bigger
        @type top: bool
        @return: Result of f. 0 when equation is true.
        @rtype: float
        """
        if t_above_ip1_c is None:
            deltap, t_above_ip1_c, inf = 0, 0, 0
        else:
            deltap, inf = self.get_delta(True), 1  # inferior to at least one stratum / not highest

        if t_below_ip1_c is None:
            deltam, t_below_ip1_c, sup = 0, 0, 0
        else:
            deltam, sup = self.get_delta(False), 1  # superior to at least one stratum / not lowest
        fac_ka_c = self.k_w_per_m2_k * (self.A_ext_m2 + top * self.A_m2) / self.c_p_w_s_per_kg_k
        fac_t_m = self.delta_t_s / self.m_strat_kg
        fac_al_zc = self.A_m2 * self.lambda_eff_w_per_m_k / self.z_m / self.c_p_w_s_per_kg_k
        return - t_i_c \
               + t_ip1_c * (1 + fac_t_m * (source * self.mdot_source_kg_per_ts + sink * self.mdot_sink_kg_per_ts
                                           + self.mdot_kg_per_ts * (deltap - deltam)
                                           + fac_ka_c + (sup + inf) * fac_al_zc)) \
               - fac_t_m * (t_above_ip1_c * (deltap * self.mdot_kg_per_ts + inf * fac_al_zc)
                            + t_below_ip1_c * (sup * fac_al_zc - deltam * self.mdot_kg_per_ts)
                            + self.t_source_c * source * self.mdot_source_kg_per_ts
                            + self.t_sink_c * sink * self.mdot_sink_kg_per_ts
                            + self.t_amb_c * fac_ka_c)

    def iterate(self):
        """
        Uses Newton Raphson to calculate temperature for each stratum and time step.

        @param tol: Error tolerance in iteration.
        @type tol: float
        """
        self.itr = 0
        error = 1

        while error > self.tol:
            J = self.jacobian(self.t_ip1_old_c)

            if self.strata == 1:  # case one stratum storage (no stratum above nor below)
                self.F[0] = self.f(self.t_i_c[0], self.t_ip1_old_c[0], None, None,
                                   self.so_inl_pos[0], self.si_inl_pos[0], True)
            else:
                # highest stratum -> no stratum above
                self.F[0] = self.f(self.t_i_c[0], self.t_ip1_old_c[0], None, self.t_ip1_old_c[1],
                                   self.so_inl_pos[0], self.si_inl_pos[0], True)
                # lowest stratum -> no stratum below
                self.F[-1] = self.f(self.t_i_c[-1], self.t_ip1_old_c[-1], self.t_ip1_old_c[-2], None,
                                    self.so_inl_pos[-1], self.si_inl_pos[-1])
                # all strata between
                for s, source, sink in zip(range(1, self.strata - 1), self.so_inl_pos[1:-1], self.si_inl_pos[1:-1]):
                    self.F[s] = self.f(self.t_i_c[s], self.t_ip1_old_c[s], self.t_ip1_old_c[s-1],
                                       self.t_ip1_old_c[s+1], source, sink)

            self.t_ip1_new_c = self.t_ip1_old_c - self.alpha * np.matmul(inv(J), self.F)  # ToDo: Get this
            error = max(abs(self.t_ip1_new_c - self.t_ip1_old_c))
            self.t_ip1_old_c = self.t_ip1_new_c

            self.itr += 1
            self.count += 1

    def save_simulation(self):
        self.results.to_csv("STS_" + str(self.strata) + "strata_" + str(len(self.results.loc[0])) + "steps.csv")


def create_heat_flows(steps):
    np.random.seed(1234)
    # return np.random.random(steps) * 100, np.random.random(steps) * 100
    return np.zeros(steps), np.zeros(steps)


if __name__ == "__main__":
    q_source_w, q_sink_w = create_heat_flows(101)
    sts = StratThermStor(np.array([40, 40 + 10/9, 40 + 20/9, 40 + 30/9, 40 + 40/9, 40 + 50/9, 40 + 60/9, 40 + 70/9,
                                   40 + 80/9, 50]), ('ct', 'hf'), 90, 25, 1, 1, 900, 1800, 825, 25)
    for s, l in zip(q_source_w, q_sink_w):
        sts.do_time_step(s, l)
    results = sts.results[0:10].T
    results.columns = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10']
    # sts.save_simulation()
