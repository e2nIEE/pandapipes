import numpy as np
import pandapipes as ps
import pandapower as pp
import pandas as pd
from matplotlib import pyplot as plt
from pandapower.control import ConstControl
from pandapower.plotting import draw_collections

from pandapower.timeseries.run_time_series import run_timeseries as run_timeseries_pp
from pandapipes.timeseries.run_time_series import run_timeseries as run_timeseries_pps
from pandapipes.multinet.timeseries.run_time_series_multinet import run_timeseries as run_timeseries_mn
from pandapower.timeseries import DFData
from pandapower.timeseries import OutputWriter

from workshop.controller import couple_two_nets


def prepare_data_power():
    data_pv = pd.read_excel(r'workshop/net_data_power/time_series.xlsx', sheet_name='pv_solar', index_col='time')
    data_ho = pd.read_excel(r'workshop/net_data_power/time_series.xlsx', sheet_name='households', index_col='time')
    data_ho = data_ho.loc[data_pv.index]
    return data_pv.loc[:, data_pv.columns.str.contains('nom')], data_ho


def prepare_data_gas(n_timesteps=10):
    profiles = pd.DataFrame()
    profiles['electrolyser'] = np.random.random(n_timesteps) * 0.001
    return profiles


def prepare_data_multinet(n_timesteps=10):
    profiles = pd.DataFrame()
    profiles['power to gas consumption'] = np.random.random(n_timesteps) * 0.2
    return profiles

    # profiles, ds = create_data_source(10)
    # ows = create_output_writers(mn, 10)


def time_series_simulation_power(net):
    data_pv, data_ho = prepare_data_power()
    profs_ho = np.random.choice(['A', 'B', 'C'], len(net.load.index))
    profs_pv = np.random.choice(['pv_south_nom', 'pv_southwest_nom'], len(net.sgen.index))
    profs_ho_p = ['H0-%s_pload' % x for x in profs_ho]
    profs_ho_q = ['H0-%s_qload' % x for x in profs_ho]
    data_source_p = data_ho.loc[:, data_ho.columns.str.contains('pload')] * 0.06
    data_source_q = data_ho.loc[:, data_ho.columns.str.contains('qload')] * 0.02
    data_source_pv = data_pv * 0.03

    ConstControl(net, 'load', 'p_mw', net.load.index, profs_ho_p, DFData(data_source_p))
    ConstControl(net, 'load', 'q_mvar', net.load.index, profs_ho_q, DFData(data_source_q))
    ConstControl(net, 'sgen', 'p_mw', net.sgen.index, profs_pv, DFData(data_source_pv))

    log_variables = [('res_bus', 'vm_pu'),
                     ('res_line', 'loading_percent'),
                     ('res_trafo', 'loading_percent')]

    OutputWriter(net, data_pv.index, r'workshop/net_data_power/results', output_file_type='.csv',
                 log_variables=log_variables)

    run_timeseries_pp(net, data_pv.index, numba=False)
    # net.output_writer.loc[0, 'object'].output['res_bus.vm_pu']
    return net


def time_series_simulation_gas(net):
    profiles = prepare_data_gas()

    ConstControl(net, 'source', 'mdot_kg_per_s', 3, 'electrolyser', DFData(profiles))

    log_variables = [('res_source', 'mdot_kg_per_s'),
                     ('res_pipe', 'v_mean_m_per_s'),
                     ('res_junction', 'p_bar')]

    OutputWriter(net, profiles.index, r'workshop/net_data_pipe/results', output_file_type='.csv',
                 log_variables=log_variables)
    run_timeseries_pps(net, profiles.index, iter=100)


def time_series_sumulation_multinet(multinet):
    profiles = prepare_data_multinet(10)

    enet = multinet['nets']['power']
    gnet = multinet['nets']['gas']

    ConstControl(enet, 'load', 'p_mw', 8, 'power to gas consumption', DFData(profiles))

    log_variables = [('res_bus', 'vm_pu'),
                     ('res_line', 'loading_percent'),
                     ('res_trafo', 'loading_percent'),
                     ('res_load', 'p_mw')]

    OutputWriter(enet, profiles.index, r'workshop/results', output_file_type='.csv',
                 log_variables=log_variables)

    log_variables = [('res_source', 'mdot_kg_per_s'),
                     ('res_pipe', 'v_mean_m_per_s'),
                     ('res_junction', 'p_bar')]

    OutputWriter(gnet, profiles.index, r'workshop/results', output_file_type='.csv',
                 log_variables=log_variables)
    run_timeseries_mn(multinet, profiles.index, iter=100)



if __name__ == '__main__':
    enet = pp.from_json('workshop/pp/net.json')
    time_series_simulation_power(enet)
    multinet = ps.from_json('workshop/multinet.json')
    gnet = multinet['nets']['gas']
    time_series_simulation_gas(gnet)
    time_series_sumulation_multinet(multinet)