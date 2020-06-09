.. _known_issues:

************
Known Issues
************

The following issues are known but have not been solved until the release of pandapipes:

- During a heat temperature calculation, pipes are not allowed to carry a fluid with zero
  velocity. This case leads to a zero line in the system matrix at the moment.
- When setting your own default options for the pipeflow (c.f. :ref:`user_options`), these values
  should still possibly be overridden during a pipeflow call. However, if a keyword argument you
  pass has the same value as the :ref:`pipeflow-function <pipeflow_function>` signature, this
  overriding will not be recognized. Once this issue is solved in pandapower, it will be changed
  in pandapipes as well.
- Please note that the heat transfer mode has not been tested as thoroughly as the hydraulics mode yet. It might be that
  some network configurations lead to unexpected behaviour. We assume that more tests will be
  added soon. In addition, we would be happy if you could report bugs on our github issue board.
