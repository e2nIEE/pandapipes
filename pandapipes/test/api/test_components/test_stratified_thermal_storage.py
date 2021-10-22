from pandapipes.component_models.abstract_models.strat_thermal_storage import StratThermStor
import math
import numpy as np
import pandas as pd
from numpy.linalg import inv


class StratThermStorTestOneStratum(StratThermStor):
    def __init__(self, init_strata_temp_c, flag_input, initial_input_source, initial_input_sink,
                 mdot_source_max_kg_per_s, mdot_sink_max_kg_per_s, delta_t_s, tank_height_mm=1700, tank_diameter_mm=810,
                 wall_thickness_mm=160, source_ind=(0, -1), load_ind=(-1, 0), tol=1e-6):
        super().__init__(init_strata_temp_c, flag_input, initial_input_source, initial_input_sink,
                         mdot_source_max_kg_per_s, mdot_sink_max_kg_per_s, delta_t_s, tank_height_mm, tank_diameter_mm,
                         wall_thickness_mm, source_ind, load_ind, tol)
        self.t_np1 = 0

    def calculate_temperature(self):
        self.t_np1 = (self.t_i_c[0] + self.delta_t_s / self.m_strat_kg *
                      (self.mdot_source_kg_per_ts * self.t_source_c + self.mdot_sink_kg_per_ts * self.t_sink_c +
                       self.k_w_per_m2_k * (self.A_ext_m2 + self.A_m2) / self.c_p_w_s_per_kg_k * self.t_amb_c)) / \
                     (1 + self.delta_t_s / self.m_strat_kg *
                      (self.mdot_source_kg_per_ts + self.mdot_sink_kg_per_ts +
                       self.k_w_per_m2_k * (self.A_ext_m2 + self.A_m2) / self.c_p_w_s_per_kg_k))

    def iterate(self):
        super().iterate()
        self.calculate_temperature()


class StratThermStorTestTwoStrata(StratThermStor):
    def __init__(self, init_strata_temp_c, flag_input, initial_input_source, initial_input_sink,
                 mdot_source_max_kg_per_s, mdot_sink_max_kg_per_s, delta_t_s, tank_height_mm=1700, tank_diameter_mm=810,
                 wall_thickness_mm=160, source_ind=(0, -1), load_ind=(-1, 0), tol=1e-6):
        super().__init__(init_strata_temp_c, flag_input, initial_input_source, initial_input_sink,
                         mdot_source_max_kg_per_s, mdot_sink_max_kg_per_s, delta_t_s, tank_height_mm, tank_diameter_mm,
                         wall_thickness_mm, source_ind, load_ind, tol)
        self.a1, self.b1, self.c1, self.a2, self.b2, self.c2 = 0, 0, 0, 0, 0, 0
        self.t1_np1, self.t2_np1 = 0, 0

    def calculate_test_matrix_two_strata(self):
        deltap, deltam = self.get_delta(True), self.get_delta(False)
        fac_t_m = self.delta_t_s / self.m_strat_kg
        fac_al_zc = self.A_m2 * self.lambda_eff_w_per_m_k / self.z_m / self.c_p_w_s_per_kg_k
        fac_kaext_c = self.k_w_per_m2_k * self.A_ext_m2 / self.c_p_w_s_per_kg_k
        fac_kaexttop_c = self.k_w_per_m2_k * (self.A_ext_m2 + self.A_m2) / self.c_p_w_s_per_kg_k
        self.a1 = 1 + fac_t_m * (fac_kaexttop_c - self.mdot_kg_per_ts * deltam + fac_al_zc + self.mdot_source_kg_per_ts)
        self.a2 = - fac_t_m * (deltap * self.mdot_kg_per_ts + fac_al_zc)
        self.b1 = fac_t_m * (deltam * self.mdot_kg_per_ts - fac_al_zc)
        self.b2 = 1 + fac_t_m * (deltap * self.mdot_kg_per_ts + fac_kaext_c + fac_al_zc + self.mdot_sink_kg_per_ts)
        self.c1 = self.t_i_c[0] + fac_t_m * (fac_kaexttop_c * self.t_amb_c + self.t_source_c * self.mdot_source_kg_per_ts)
        self.c2 = self.t_i_c[1] + fac_t_m * (fac_kaext_c * self.t_amb_c + self.t_sink_c * self.mdot_sink_kg_per_ts)

    def calculate_temperatures_from_test_matrix_two_strata(self):
        try:
            self.t1_np1 = (self.c2 - self.c1 * self.b2 / self.b1) / (self.a2 - self.a1 * self.b2 / self.b1)
        except ZeroDivisionError:
            print("ZeroDivisionError")
            self.t1_np1 = self.c1 / self.a1
        try:
            self.t2_np1 = (self.c2 - self.c1 * self.a2 / self.a1) / (self.b2 - self.b1 * self.a2 / self.a1)
        except ZeroDivisionError:
            print("ZeroDivisionError")
            self.t2_np1 = self.c2 / self.b2

    def iterate(self):
        super().iterate()
        self.calculate_test_matrix_two_strata()
        self.calculate_temperatures_from_test_matrix_two_strata()


class StratThermStorTestFiveStrata(StratThermStor):
    def __init__(self, init_strata_temp_c, flag_input, initial_input_source, initial_input_sink,
                 mdot_source_max_kg_per_s, mdot_sink_max_kg_per_s, delta_t_s, tank_height_mm=1700, tank_diameter_mm=810,
                 wall_thickness_mm=160, source_ind=(0, -1), load_ind=(-1, 0), tol=1e-6):
        super().__init__(init_strata_temp_c, flag_input, initial_input_source, initial_input_sink,
                         mdot_source_max_kg_per_s, mdot_sink_max_kg_per_s, delta_t_s, tank_height_mm, tank_diameter_mm,
                         wall_thickness_mm, source_ind, load_ind, tol)
        self.a_self_top, self.a_self_mid, self.a_self_bot, self.a_above, self.a_below, \
        self.c1, self.c2, self.c3, self.c4, self.c5 = 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
        self.t1, self.t2, self.t3, self.t4, self.t5 = 0., 0., 0., 0., 0.

    def calculate_test_matrix_five_strata(self):
        deltap, deltam = self.get_delta(True), self.get_delta(False)
        dt_m = self.delta_t_s / self.m_strat_kg
        kAx_cp = self.k_w_per_m2_k * self.A_ext_m2 / self.c_p_w_s_per_kg_k
        kAx_cp_top = self.k_w_per_m2_k * (self.A_ext_m2 + self.A_m2) / self.c_p_w_s_per_kg_k
        Al_zcp = self.A_m2 * self.lambda_eff_w_per_m_k / self.z_m / self.c_p_w_s_per_kg_k
        self.a_self_top = 1 + dt_m * (self.mdot_source_kg_per_ts + kAx_cp_top - deltam * self.mdot_kg_per_ts + Al_zcp)
        self.a_self_mid = 1 + dt_m * (kAx_cp + self.mdot_kg_per_ts * (deltap - deltam) + 2 * Al_zcp)
        self.a_self_bot = 1 + dt_m * (self.mdot_sink_kg_per_ts + kAx_cp + deltap * self.mdot_kg_per_ts + Al_zcp)
        self.a_above = - dt_m * (deltap * self.mdot_kg_per_ts + Al_zcp)
        self.a_below = dt_m * (deltam * self.mdot_kg_per_ts - Al_zcp)
        c = dt_m * kAx_cp * self.t_amb_c
        self.c1 = self.t_i_c[0] + dt_m * (self.mdot_source_kg_per_ts * self.t_source_c + kAx_cp_top * self.t_amb_c)
        self.c2, self.c3, self.c4 = self.t_i_c[1] + c, self.t_i_c[2] + c, self.t_i_c[3] + c
        self.c5 = self.t_i_c[4] + c + dt_m * self.mdot_sink_kg_per_ts * self.t_sink_c

    def calculate_temperatures_from_test_matrix_five_strata(self):
        a00 = self.a_self_top
        a01 = self.a_below
        b0 = self.c1
        a11 = self.a_self_mid - self.a_above * a01 / a00
        a12 = self.a_below
        b1 = self.c2 - self.a_above * b0 / a00
        a22 = self.a_self_mid - self.a_above * a12 / a11
        a23 = self.a_below
        b2 = self.c3 - self.a_above * b1 / a11
        a33 = self.a_self_mid - self.a_above * a23 / a22
        a34 = self.a_below
        b3 = self.c4 - self.a_above * b2 / a22
        a44 = self.a_self_bot - self.a_above * a34 / a33
        b4 = self.c5 - self.a_above * b3 / a33

        self.t5 = b4 / a44
        self.t4 = (b3 - a34 * self.t5) / a33
        self.t3 = (b2 - a23 * self.t4) / a22
        self.t2 = (b1 - a12 * self.t3) / a11
        self.t1 = (b0 - a01 * self.t2) / a00

    def iterate(self):
        super().iterate()
        self.calculate_test_matrix_five_strata()
        self.calculate_temperatures_from_test_matrix_five_strata()


def test_one_stratum():
    q_source_w, q_sink_w = np.array([20, 60, 40, 80, 80]), np.array([40, 50, 70, 70, 40])  # create_heat_flows(60*24)
    sts = StratThermStorTestOneStratum(np.array([50]), ('ct', 'hf'), 90, 25, 1, 1, 60, 1700, 810, 160)
    for s, l in zip(q_source_w, q_sink_w):
        sts.do_time_step(s, l)
        assert not np.any(np.around(sts.t_ip1_new_c - np.array([sts.t_np1]), int(-math.log(sts.tol, 10))))
    print("This is your lucky day!")


def test_two_strata():
    q_source_w, q_sink_w = np.array([20, 60, 40, 80, 80]), np.array([40, 50, 70, 70, 40])  # create_heat_flows(60*24)
    sts = StratThermStorTestTwoStrata(np.array([80, 30]), ('ct', 'hf'), 90, 25, 1, 1, 60, 1700, 810, 160)
    for s, l in zip(q_source_w, q_sink_w):
        sts.do_time_step(s, l)
        assert not np.any(np.around(sts.t_ip1_new_c - np.array([sts.t1_np1, sts.t2_np1]), int(-math.log(sts.tol, 10))))
    print("No problemo ^^")


def test_five_strata():
    q_source_w, q_sink_w = np.array([20, 60, 40, 80, 80]), np.array(
        [40, 50, 70, 70, 40])  # create_heat_flows(60*24)
    sts = StratThermStorTestFiveStrata(np.array([80, 65, 50, 35, 30]), ('ct', 'hf'), 90, 25, 1, 1, 60, 1700, 810, 160)
    for s, l in zip(q_source_w, q_sink_w):
        sts.do_time_step(s, l)
        assert not np.any(np.around(sts.t_ip1_new_c - np.array([sts.t1, sts.t2, sts.t3, sts.t4, sts.t5]),
                                    int(-math.log(sts.tol, 10))))
    print("Eaaasy o.o")


if __name__ == '__main__':
    test_one_stratum()
    test_two_strata()
    test_five_strata()  # ToDo: There were no warnings before
