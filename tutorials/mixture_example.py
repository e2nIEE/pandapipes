import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

import pandapipes as pp

color = sns.color_palette("colorblind")
from pandapipes.topology import calc_distance_to_junction
import numpy as np


def prepare_grid():
    net = pp.from_json(r'tutorials/files/sw_gas.json')
    pp.create_fluid_from_lib(net, 'hgas')
    net.ext_grid.fluid = 'hgas'
    # pp.create_sinks(net, net.junction.index.values, 1e-5)

    j1128 = pp.create_junction(net, 0.1, 283.15)

    pp.create_pipe_from_parameters(net, 1127, j1128, 0.087, 0.1, 1)

    pp.create_source(net, j1128, 0.013298101722575308 / 2, 'hydrogen')
    pp.create_source(net, 204, 0.013298101722575308 / 2, 'hydrogen')
    pp.create_source(net, 1032, 0.013298101722575308 / 2, 'hydrogen')

    net.source.in_service = False

    pp.pipeflow(net)
    return net


def set_in_service(net, source_id):
    net.source.loc[source_id, 'in_service'] = True
    pp.pipeflow(net)


def plotting(net, legend_name, color, ax=None, sort=False, save=True):
    rcParams.keys()
    rcParams.update({'axes.grid': True, 'axes.grid.axis': 'y', 'font.size': 14., 'axes.edgecolor': 'grey',
                     'mathtext.default': 'regular'})
    if sort is None:
        data = net.res_junction.w_hydrogen * 100
    elif sort is True:
        data = net.res_junction.sort_values('w_hydrogen').reset_index().w_hydrogen * 100
    else:
        data = net.res_junction.loc[sort, 'w_hydrogen'] * 100
    data.index = data.index.astype('str')
    data.name = legend_name
    if ax is None:
        ax = data.plot.line(color=color, legend=True)
    else:
        ax = data.plot.line(color=color, ax=ax, legend=True)
    #ax.set_xticks(np.linspace(0, len(data), len(data.index[::20])))
    #ax.set_xticklabels(data.index[::20])
    ax.set_ylim((-5, 105))
    ax.set_ylabel('Massenanteil H2 [%]')
    ax.set_xlabel('Knotenindex')
    ax.xaxis.grid(False)
    ax.get_legend().remove()
    ax.legend(loc='lower center', ncol=2, bbox_to_anchor=(0.5, 1.))
    if save:
        fig = plt.gcf()
        plt.tight_layout()
        fig.savefig(r'massenanteil_%s.pdf' % legend_name.replace('/', '_'))
    return ax, data


def determine_distance(net, bus):
    return calc_distance_to_junction(net, bus)


if __name__ == '__main__':
    net = prepare_grid()
    dist = determine_distance(net, 45)
    fig = plt.figure(figsize=(7, 4))
    ax, data1 = plotting(net, 'kein Elektrolyseur', color[0], None, dist.index)
    set_in_service(net, 1)
    ax, data2 = plotting(net, 'Elektrolyseur I', color[7], ax, dist.index)
    set_in_service(net, 2)
    ax, data3 = plotting(net, 'Elektrolyseur I/II', color[5], ax, dist.index)
    set_in_service(net, 0)
    ax, data4 = plotting(net, 'Elektrolyseur I/II/III', color[6], ax, dist.index)
    plt.show()