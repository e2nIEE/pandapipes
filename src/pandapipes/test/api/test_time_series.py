import numpy as np

import pandapipes as pps
from pandapipes import networks as ntw
from pandapipes.timeseries import run_timeseries


def test_person_run_fct_time_series():
    def person_run_fct(net, sol_vec=None, **kwargs):
        pps.pipeflow(net, sol_vec, **kwargs)
        net.res_junction.p_bar.values[:] = 15.

    net = ntw.water_strand_net_2pumps()
    max_iter_hyd = 3
    max_iter_therm = 1
    pps.pipeflow(net, mode='sequential', max_iter_hyd=max_iter_hyd, max_iter_therm=max_iter_therm)
    assert all(np.isclose(net.res_junction.p_bar.values,
                          [7., 25.02309542, 24.73038636, 22.11851835, 16.70728061, 16.23637491]))
    run_timeseries(net, time_steps=range(10), max_iter_hyd=max_iter_hyd, max_iter_therm=max_iter_therm,
                   mode='sequential', run=person_run_fct)
    assert all(net.output_writer.iat[0, 0].np_results['res_junction.p_bar'].flatten() == 15.)
