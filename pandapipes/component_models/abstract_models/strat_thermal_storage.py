import numpy as np
import pandas as pd
from numpy.linalg import inv


class StratThermStor():
    def __init__(self, init_strata_temp_c, t_source_c, t_sink_c, mdot_source_max_kg_per_s, mdot_sink_max_kg_per_s,
                 delta_t_s, tank_height_mm=1700, tank_diameter_mm=810, wall_thickness_mm=160,
                 source_ind=(0, -1), load_ind=(-1, 0), tol=1e-6):
        """
        Model of a stratified thermal storage. Heat medium: water.
        Heat capacity c_p and density are assumed to be constant.

        @param init_strata_temp_c: Each stratum's temperature in Celsius from highest to lowest stratum.
                                   Also defines the number of strata.
        @type init_strata_temp_c: numpy.ndarray
        @param t_source_c: Temperature of heated water entering from the source circuit in Celsius.
        @type t_source_c: float
        @param t_sink_c: Temperature of cooled water entering from the load circuit in Celsius.
        @type t_sink_c: float
        @param mdot_source_kg_per_ts: Mass flow from source circuit in kg per time step.
        @type mdot_source_kg_per_ts: float
        @param mdotl: Mass flow from load circuit in kg per time step.
        @type mdotl: float
        @param delta_t_s: Time step size.
        @type delta_t_s: float
        @param tank_height_mm: Inner height of the thermal storage tank in meter.
        @type tank_height_mm: float
        @param tank_diameter_mm: Inner diameter of the thermal storage tank in meter.
        @type tank_diameter_mm: float
        @param wall_thickness_mm: Diameter of the tank's wall.
        @type wall_thickness_mm: float
        @param source_ind: Indices of the strata with first the inlet and second the outlet of the source circuit.
        @type source_ind: tuple(int, int)
        @param load_ind: Indices of the strata with first the inlet and second the outlet of the load circuit.
        @type load_ind: tuple(int, int)
        @param tol: Tolerance of error in iteration.
        @type tol: float
        """
        # tank measures
        self.strata = len(init_strata_temp_c)  # number of strata
        self.z_m = tank_height_mm / 1000 / self.strata  # height of each stratum
        self.A_m2 = np.pi * (tank_diameter_mm / 1000 - 2 * wall_thickness_mm / 1000) ** 2 / 4  # cross section area
        self.A_ext_m2 = np.pi * tank_diameter_mm / 1000 * self.z_m   # barrel area of one stratum
        self.m_strat_kg = self.A_m2 * self.z_m * 996  # each stratum's mass

        # heat medium and tank properties
        self.c_p_w_s_per_kg_k, self.k_w_per_m2_k, self.lambda_eff_w_per_m_k = 4183, 0.5, 1.5

        # parameters: ambient air temperature; temperature, mass flow and inlet position of source and load circuit
        self.t_amb_c = 20
        self.t_source_c, self.t_sink_c = t_source_c, t_sink_c
        self.mdot_source_kg_per_ts, self.mdot_sink_kg_per_ts, self.mdot_kg_per_ts = None, None, None
        self.mdot_source_max_kg_per_ts, self.mdot_sink_max_kg_per_ts = mdot_source_max_kg_per_s * delta_t_s, \
                                                                       mdot_sink_max_kg_per_s * delta_t_s
        self.q_source_w, self.q_sink_w = 0., 0.
        self.s_inl_pos, self.l_inl_pos = np.zeros((self.strata,)), np.zeros((self.strata,))
        self.s_inl_pos[source_ind[0]], self.l_inl_pos[load_ind[0]] = 1, 1
        self.s_outl_ind, self.l_outl_ind = source_ind[1], load_ind[1]

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

    def initialize_time_step(self, q_source_w, q_sink_w, mdot_source_kg_per_s, mdot_sink_kg_per_s, t_source_c, t_sink_c):
        flag_source, self.q_source_w, self.mdot_source_kg_per_ts, self.t_source_c = \
            self.calculate_missing_values(q_source_w, mdot_source_kg_per_s, self.mdot_source_max_kg_per_ts,
                                          t_source_c, self.t_i_c[self.s_outl_ind])
        flag_sink, self.q_sink_w, self.mdot_sink_kg_per_ts, self.t_sink_c = \
            self.calculate_missing_values(q_sink_w, mdot_sink_kg_per_s, self.mdot_sink_max_kg_per_ts, t_sink_c,
                                          self.t_i_c[self.l_outl_ind])
        self.mdot_kg_per_ts = self.mdot_source_kg_per_ts - self.mdot_sink_kg_per_ts
        return [flag_source, flag_sink]

    def calculate_missing_values(self, q, mdot, mdot_max, t_circuit, t_stratum):
        message = "One value of heat flow, mass flow and temperature spread " \
                  "of source and sink circuit each must not be defined."
        flag = "orig"
        if q is None:
            if None in [mdot, t_circuit]:
                raise ValueError(message)
            else:
                q = self.calculate_heat_flow(mdot, t_circuit, t_stratum)
        elif mdot is None:
            if t_circuit is None:
                raise ValueError(message)
            else:
                if t_circuit == t_stratum:
                    flag, mdot = "min", 0
                else:
                    flag, mdot = self.calculate_mass_flow(q, t_circuit, t_stratum, mdot_max)
        elif t_circuit is None:
            if mdot == 0:
                flag, t_circuit, q = "m0", t_stratum, 0
            else:
                t_circuit = t_stratum + self.calculate_temperature_spread(q, mdot)
        return flag, q, mdot, t_circuit





                # def do_time_step_from_heat_flow(self, q_source_w, q_sink_w):
    #     self.q_source_w, self.q_sink_w = q_source_w, q_sink_w
    #     flag = self.calculate_mass_flow()
    #     self.mdot_kg_per_ts = self.mdot_source_kg_per_ts - self.mdot_sink_kg_per_ts
    #     self.do_time_step(flag)
    #
    # def do_time_step_from_mass_flow(self, mdot_source_kg_per_s, mdot_sink_kg_per_s):
    #     self.mdot_source_kg_per_ts, self.mdot_sink_kg_per_ts = mdot_source_kg_per_s * self.delta_t_s, \
    #                                                            mdot_sink_kg_per_s * self.delta_t_s
    #     self.mdot_kg_per_ts = self.mdot_source_kg_per_ts - self.mdot_sink_kg_per_ts
    #     flag = self.calculate_heat_flow()
    #     self.do_time_step(flag)

    def do_time_step(self, q_source_w=None, q_sink_w=None, mdot_source_kg_per_s=None, mdot_sink_kg_per_s=None,
                             t_source_c=None, t_sink_c=None):
        flag = self.initialize_time_step(q_source_w, q_sink_w, mdot_source_kg_per_s, mdot_sink_kg_per_s,
                                         t_source_c, t_sink_c)
        self.iterate()
        self.t_i_c = self.t_ip1_new_c
        self.finalize_time_step(flag)

    def finalize_time_step(self, flag):
        t = len(self.results.loc[0])
        self.results[t] = np.append(self.t_i_c, [False, False])
        if flag[0] == "orig":
            self.results.loc["source_revised", t] = False
        else:
            self.results.loc["source_revised", t] = True
            if flag[0] == "max":
                print("          Energy from source circuit can't be stored completely: Maximum mass flow exceeded")
            elif flag[0] == "min":
                print("          Energy from source circuit can't be stored: Storage is completly charged")
        if flag[1] == "orig":
            self.results.loc["sink_revised", t] = False
        else:
            self.results.loc["sink_revised", t] = True
            if flag[1] == "max":
                print("          Energy for load circuit can't be provided completely: Maximum mass flow exceeded")
            if flag[0] == "min":
                print("          Energy for load circuit can't be withdrawn: Storage is completely discharged")

    def calculate_mass_flow(self, q_w, t_in_c, t_stratum_c, max):
        flag = "orig"
        mdot_kg_per_ts = abs(q_w * self.delta_t_s / self.c_p_w_s_per_kg_k * (t_in_c - t_stratum_c))
        if mdot_kg_per_ts > max:
            flag, mdot_kg_per_ts = "max", max
        return flag, mdot_kg_per_ts


        flag = ["orig", "orig"]
        try:
            self.mdot_source_kg_per_ts = self.q_source_w * self.delta_t_s / \
                                         (self.c_p_w_s_per_kg_k * (self.t_source_c - self.t_i_c[self.s_outl_ind]))
            if self.mdot_source_kg_per_ts > self.mdot_source_max_kg_per_ts:  # ToDo: Add revision of heat flow?
                self.mdot_source_kg_per_ts = self.mdot_source_max_kg_per_ts
                flag[0] = "max"
        except ZeroDivisionError:
            self.mdot_source_kg_per_ts = 0
            flag[0] = "min"
        try:
            self.mdot_sink_kg_per_ts = self.q_sink_w * self.delta_t_s / \
                                       (self.c_p_w_s_per_kg_k * (self.t_i_c[self.l_outl_ind] - self.t_sink_c))
            if self.mdot_sink_kg_per_ts > self.mdot_sink_max_kg_per_ts:
                self.mdot_sink_kg_per_ts = self.mdot_sink_max_kg_per_ts
                flag[1] = "max"
        except ZeroDivisionError:
            self.mdot_sink_kg_per_ts = 0
            flag[1] = min
        return flag

    def calculate_heat_flow(self, mdot_kg_per_s, t_in_c, t_stratum_c):
        return abs(mdot_kg_per_s * self.c_p_w_s_per_kg_k * (t_in_c - t_stratum_c))
        # self.q_source_w = self.mdot_source_kg_per_ts / self.delta_t_s * \
        #                   (self.c_p_w_s_per_kg_k * (self.t_source_c - self.t_i_c[self.s_outl_ind]))
        # self.q_sink_w = self.mdot_sink_kg_per_ts / self.delta_t_s * \
        #                 (self.c_p_w_s_per_kg_k * (self.t_i_c[self.l_outl_ind] - self.t_sink_c))

    def calculate_temperature_spread(self, q_w, mdot_kg_per_s):
        return q_w / mdot_kg_per_s / self.c_p_w_s_per_kg_k

        deltat_source_k = self.q_source_w * self.delta_t_s / self.mdot_source_kg_per_ts / self.c_p_w_s_per_kg_k
        deltat_sink_k = self.q_sink_w * self.delta_t_s / self.mdot_sink_kg_per_ts / self.c_p_w_s_per_kg_k
        self.t_source_c = self.t_i_c[self.s_outl_ind] + deltat_source_k
        self.t_sink_c = self.t_i_c[self.l_outl_ind] + deltat_sink_k
        return flag

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

        for row, s, l in zip(range(self.strata), self.s_inl_pos, self.l_inl_pos):
            if row:  # should not be calculated for top stratum since there's no stratum above
                J[row, row - 1] = (self.f(self.t_i_c[row], temps[row], temps[row-1] + h, temps[row+1], s, l, not (row-1)) -
                                   self.f(self.t_i_c[row], temps[row], temps[row-1],     temps[row+1], s, l, not (row-1))) / h
            if row + 1 < self.strata:  # should not be calculated for bottom stratum since there's no stratum below
                J[row, row + 1] = (self.f(self.t_i_c[row], temps[row], temps[row-1], temps[row+1] + h, s, l) -
                                   self.f(self.t_i_c[row], temps[row], temps[row-1], temps[row+1],     s, l)) / h
            J[row, row] = (self.f(self.t_i_c[row], temps[row] + h, temps[row-1], temps[row+1], s, l, not row) -
                           self.f(self.t_i_c[row], temps[row],     temps[row-1], temps[row+1], s, l, not row)) / h

        return J

    def f(self, t_i_c, t_ip1_c, t_above_ip1_c, t_below_ip1_c, source=False, load=False, top=False):
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
        @param source: If there's a source inlet in this stratum
        @type source: bool
        @param load: If there's a load inlet in this stratum
        @type load: bool
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
               + t_ip1_c * (1 + fac_t_m * (source * self.mdot_source_kg_per_ts + load * self.mdot_sink_kg_per_ts
                                           + self.mdot_kg_per_ts * (deltap - deltam)
                                           + fac_ka_c + (sup + inf) * fac_al_zc)) \
               - fac_t_m * (t_above_ip1_c * (deltap * self.mdot_kg_per_ts + inf * fac_al_zc)
                            + t_below_ip1_c * (sup * fac_al_zc - deltam * self.mdot_kg_per_ts)
                            + self.t_source_c * source * self.mdot_source_kg_per_ts
                            + self.t_sink_c * load * self.mdot_sink_kg_per_ts
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
                                   self.s_inl_pos[0], self.l_inl_pos[0], True)
            else:
                # highest stratum -> no stratum above
                self.F[0] = self.f(self.t_i_c[0], self.t_ip1_old_c[0], None, self.t_ip1_old_c[1],
                                   self.s_inl_pos[0], self.l_inl_pos[0], True)
                # lowest stratum -> no stratum below
                self.F[-1] = self.f(self.t_i_c[-1], self.t_ip1_old_c[-1], self.t_ip1_old_c[-2], None,
                                    self.s_inl_pos[-1], self.l_inl_pos[-1])
                # all strata between
                for s, source, load in zip(range(1, self.strata - 1), self.s_inl_pos[1:-1], self.l_inl_pos[1:-1]):
                    self.F[s] = self.f(self.t_i_c[s], self.t_ip1_old_c[s], self.t_ip1_old_c[s-1],
                                       self.t_ip1_old_c[s+1], source, load)

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
    q_source_w, q_sink_w = np.zeros(292), np.zeros(292)  # create_heat_flows(60*24)
    sts = StratThermStor(np.array([40, 40, 50, 60, 70, 80, 80, 80, 80, 80]), 90, 25, 1, 1, 60, 1700, 810, 160)
    for s, l in zip(q_source_w, q_sink_w):
        sts.do_time_step(s, l, None, None, 90, 25)
    results = sts.results[0:10].T
    sts.save_simulation()
