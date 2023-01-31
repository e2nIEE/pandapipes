import pandapipes as ps

import pandas as pd
from os.path import join
from pandapipes.plotting import simple_plot


def pipe_net_example():
    input_dir = r'workshop/net_data_pipe'
    in_junctions = pd.read_csv(join(input_dir, 'example_net-junctions.CSV'))
    in_pipes = pd.read_csv(join(input_dir, 'example_net-pipes.CSV'))

    fluid = 'hgas'  # water, lgas, hydrogen
    net = ps.create_empty_network('pandapipes workshop - net 1', fluid=fluid)

    geodata = in_junctions[['lon', 'lat']].values

    ps.create_junctions(net, nr_junctions=31, pn_bar=1, tfluid_k=283.15,
                        height_m=in_junctions['height'], geodata=geodata)
    ps.create_pipes_from_parameters(net, in_pipes['from_junction'], in_pipes['to_junction'],
                                    length_km=in_pipes['length_km'], diameter_m=0.05, k_mm=0.2)
    # alternatively, with standard types from
    # https://pandapipes.readthedocs.io/en/develop/standard_types/std_types_in_pandapipes.html
    # ps.create_pipes(net, in_pipes['from_junction'], in_pipes['to_junction'],
    #                 std_type='50_PE_100_SDR_11',  length_km=in_pipes['length_km'])

    return net


def gas_net_example():
    input_dir = r'workshop/net_data_pipe'
    in_sinks = pd.read_csv(join(input_dir, 'example_net-sinks.CSV'))
    in_sources = pd.read_csv(join(input_dir, 'example_net-sources.CSV'))

    net = pipe_net_example()

    ps.create_sinks(net, in_sinks['junction'], mdot_kg_per_s=0.001)  # in_sinks['m_dot'])
    ps.create_sources(net, in_sources['junction'], mdot_kg_per_s=0.0005)  # in_sinks['m_dot'])
    ps.create_ext_grid(net, junction=0, p_bar=3, t_k=313.15)
    ps.create_ext_grid(net, junction=14, p_bar=16, t_k=283.15)
    ps.pipeflow(net)
    print_results(net)
    return net


def gas_net_example2():
    net = gas_net_example()
    ps.create_valves(net, [4, 9, 22], [16, 18, 28], 0.05, opened=False)

    ps.pipeflow(net, iter=2000)
    print_results(net)

    ps.create_pressure_control(net, 5, 9, 9, 2.7)

    ps.pipeflow(net, iter=2000)
    print_results(net)

    ps.create_pump(net, 9, 6, std_type='P1')

    ps.pipeflow(net, iter=2000)
    print_results(net)

    ps.create_fluid_from_lib(net, 'hydrogen', True)

    ps.pipeflow(net, iter=2000)
    print_results(net)
    return net

def heat_net_example():
    input_dir = r'workshop/net_data_pipe'
    in_ijs= pd.read_csv(join(input_dir, 'example_net-intermediate_junctions.CSV'))

    net = pipe_net_example()
    simple_plot(net)

    ps.create_fluid_from_lib(net, 'water')

    ps.create_circ_pump_const_pressure(net, 4, 9, 7, 3, t_flow_k=330)
    ps.create_circ_pump_const_mass_flow(net, 22, 28, 10, 0.1, t_flow_k=320)

    js = ps.create_junctions(net, 6, 1, 283.15, geodata=in_ijs[['lon', 'lat']].values)

    ps.create_flow_controls(net, [5, 11, 14, 18, 23, 30,], js, 2, 0.05)
    ps.create_heat_exchangers(net, js, [6, 13, 19, 29, 24, 27], 0.1, 2000)


    ps.pipeflow(net)
    print_results(net, True)
    ps.pipeflow(net, mode='all')
    print_results(net, TRue)
    return net

def print_results(net, heat=False):
    print('pressure:')
    print(net.res_junction.p_bar)
    if heat:
        print('temperature:')
        print(net.res_junction.t_k)
    print('velocity:')
    print(net.res_pipe.v_mean_m_per_s)



    print('min. pressure:')
    print(net.res_junction.p_bar.min())
    if heat:
        print('min temperature:')
        print(net.res_junction.t_k.min())
    print('max. velocity:')
    print(net.res_pipe.v_mean_m_per_s.abs().max())



if __name__ == '__main__':
    net1 = pipe_net_example()
    net2 = gas_net_example()
    net3 = gas_net_example2()
    net4 = heat_net_example()