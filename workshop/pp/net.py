import pandapower as pp
import pandas as pd


def power_net_example():
    net = pp.create_empty_network(name='minimal example')

    in_buses = pd.read_csv(r'workshop/net_data_power/example_net-buses.CSV')
    in_lines = pd.read_csv(r'workshop/net_data_power/example_net-lines.CSV')
    in_loads = pd.read_csv(r'workshop/net_data_power/example_net-loads.CSV')
    in_sgens = pd.read_csv(r'workshop/net_data_power/example_net-sgens.CSV')

    geodata = in_buses[['lon', 'lat']].values

    pp.create_buses(net, 33, vn_kv=[0.4] * 31 + [10, 10], name=in_buses['Name'], geodata=geodata)
    # pp.create_line(net, 0, 1, length_km=5, std_type='NAYY 4x150 SE')
    pp.create_lines(net, from_buses=in_lines['from_bus'], to_buses=in_lines['to_bus'], length_km=in_lines['length_km'],
                    std_type='NAYY 4x150 SE')

    pp.create_loads(net, buses=in_loads['bus'], p_mw=in_loads['P_kW'] / 1000, q_mvar=in_loads['Q_kVar'] / 1000)

    pp.create_sgens(net, buses=in_sgens['bus'], p_mw=in_sgens['P_kW'] / 1000)

    pp.create_ext_grid(net, 31)
    pp.create_ext_grid(net, 32)

    pp.create_transformer(net, 31, 0, std_type='0.25 MVA 10/0.4 kV', tap_pos=-2)
    pp.create_transformer(net, 32, 14, std_type='0.25 MVA 10/0.4 kV', tap_pos=-2)

    pp.create_switches(net, [9, 4, 22], [18, 16, 28], ['b', 'b', 'b'], closed=False)

    pp.runpp(net)
    return net


if __name__ == '__main__':
    net = power_net_example()
    pp.to_json(net, 'workshop/pp/net.json')
