Change Log
=============

[0.2.0] - 28.08.2020
-------------------------------
- [ADDED] added the pipeflow option "reuse_internal_data" which allows to reuse the system matrix from one pipeflow to
the next in combination with "only_update_hydraulic_matrix" - useful for timeseries calculations
- [ADDED] hydrogen properties
- [ADDED] Swamee-Jain friction model
- [ADDED] test networks (water) for Swamee-Jain friction model
- [ADDED] further explanation in the documentation, e.g. on heating networks and time series / controller
- [ADDED] heating network and time series tutorials
- [CHANGED] property files for bi-atomic gases
- [CHANGED] make ppipe_hook serializable and inherit from pp_hook by using decorators
- [CHANGED] changed column "controller" in controller table to "object"
- [CHANGED] names of parameters for regression function in pump
- [CHANGED] pressure lift for pumps is now always >= 0
- [CHANGED] on reverse flow, the pressure lift for pumps is 0
- [CHANGED] add_fluid_to_net is now a private function (usually the wrapper create_fluid_from_lib should be used)
- [CHANGED] moving sum_by_group and sum_by_group_sorted from internal_toolbox to toolbox
- [FIXED] direction of pump in the water test network 'versatility' for OpenModelica
- [FIXED] accurate calculation of v in get_internal_results for pipes

[0.1.2] - 2020-06-05
-------------------------------
- [ADDED] allow pipeflow for empty net (with no results)
- [ADDED] tests for plotting
- [ADDED] new toolbox functions and tests
- [ADDED] get... methods for fluids
- [ADDED] tutorial, documentation and tests for heat networks
- [CHANGED] default column for controllers changed from controller to object
- [CHANGED] deepcopy (now in ADict) and repr of pandapipes net + fluid
- [CHANGED] improved plotting: respect in_service
- [CHANGED] for fluids, comp, molar_mass and der_comp are now read from .txt-files and are no longer hardcoded
- [CHANGED] pandapipes.toolbox renamed to pandapipes.internals_toolbox
- [FIXED] np.isclose comparison instead of 'p_from != p_to' in pipe_component to allow for computational inaccuracy

[0.1.1] - 2020-04-03
-------------------------------
- [ADDED] badges for pypi and versions
- [CHANGED] result table initialization now in most cases only contains one DF creation call #42
- [CHANGED] from pandapower tempdir to pytest tmp_path in test
- [CHANGED] default orientation of sink and source collections to avoid overlapping
- [FIXED] missing images and typos in documentation

[0.1.0] - 2020-03-18
-------------------------------
- first release of pandapipes