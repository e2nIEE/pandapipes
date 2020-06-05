Change Log
=============

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