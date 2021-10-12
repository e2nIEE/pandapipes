from pandapipes.component_models.abstract_models.strat_thermal_storage import StratThermStor
import math
import numpy as np
import pandas as pd
from numpy.linalg import inv

class TestStratThermStor(StratThermStor):
    def __init__(self, init_strata_temp_c, t_source_c, t_sink_c, mdot_source_max_kg_per_s, mdot_sink_max_kg_per_s,
                 delta_t_s, tank_height_mm=1700, tank_diameter_mm=810, wall_thickness_mm=160,
                 source_ind=(0, -1), load_ind=(-1, 0), tol=1e-6):
        super().__init__(init_strata_temp_c, t_source_c, t_sink_c, mdot_source_max_kg_per_s, mdot_sink_max_kg_per_s,
                 delta_t_s, tank_height_mm, tank_diameter_mm, wall_thickness_mm,
                 source_ind, load_ind, tol)
        self.a1, self.b1, self.c1, self.a2, self.b2, self.c2 = 0, 0, 0, 0, 0, 0
        self.t1_np1, self.t2_np1 = 0, 0

    def calculate_test_matrix_two_strata(self):
        deltap, deltam = self.get_delta(True), self.get_delta(False)
        fac_t_m = self.delta_t_s / self.m_strat_kg
        fac_al_zc = self.A_m2 * self.lambda_eff_w_per_m_k / self.z_m / self.c_p_w_s_per_kg_k
        fac_kaext_c = self.k_w_per_m2_k * self.A_ext_m2 / self.c_p_w_s_per_kg_k
        fac_kaexttop_c = self.k_w_per_m2_k * (self.A_ext_m2 + self.A_m2) / self.c_p_w_s_per_kg_k
        self.a1 = 1 + fac_t_m * (fac_kaext_c - self.mdot_kg_per_ts * deltam + fac_al_zc + self.mdot_source_kg_per_ts)
        self.a2 = - fac_t_m * (deltap * self.mdot_kg_per_ts + fac_al_zc)
        self.b1 = fac_t_m * (deltam * self.mdot_kg_per_ts - fac_al_zc)
        self.b2 = 1 + fac_t_m * (deltap * self.mdot_kg_per_ts + fac_kaext_c + fac_al_zc + self.mdot_sink_kg_per_ts)
        self.c1 = self.t_i_k[0] + fac_t_m * (fac_kaext_c * self.t_amb_k + self.t_source_k * self.mdot_source_kg_per_ts)
        self.c2 = self.t_i_k[1] + fac_t_m * (fac_kaext_c * self.t_amb_k + self.t_sink_k * self.mdot_sink_kg_per_ts)

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


def test_two_strata():
    q_source_w, q_sink_w = np.array([20, 60, 40, 80, 80]), np.array([40, 50, 70, 70, 40])  # create_heat_flows(60*24)
    sts = TestStratThermStor(np.array([80, 30]), 90, 25, 1, 1, 60, 1700, 810, 160)
    for s, l in zip(q_source_w, q_sink_w):
        sts.do_time_step_from_heat_flow(s, l)
        assert not np.any(np.around(sts.t_ip1_k_new_k - np.array([sts.t1_np1, sts.t2_np1]), int(-math.log(sts.tol, 10))))
    print("No problemo ^^")


    # print('{0:.0f} min, {1}: mdot = {2: 3f} kg/s: {3}'.format(t * sts.delta_t_s / 60, sts.itr, sts.mdot_kg_per_ts,
    #                                                           np.around(sts.t_ip1_k_new_k, 5)))
    # print("a1: {0:3f}, a2: {1:3f}, b1: {2:3f}, b2: {3:3f}, c1: {4:3f}, c2: {5:3f}".format
    #       (sts.a1, sts.a2, sts.b1, sts.b2, sts.c1, sts.c2))
