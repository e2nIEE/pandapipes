.. _run_pipeflow:

******************
Running a Pipeflow
******************

If you want to run a pipeflow with pandapipes, it might be the easiest to just try running the code

>>> pp.pipeflow(net)

However, this might not always suit your purpose. In order to control what is calculated and which
precise boundary conditions to use, some options exist that influence the internals. Changing the
default options can not only influence the boundary conditions, but also the execution time, the
convergence behavior or the output that is calculated.

It is recommended to know these options, which are explained more in detail in
:ref:`pipeflow_options`. If you want to learn more about how the pipeflow works internally, the
section :ref:`pipeflow_procedure` might give you a valuable insight. Also the internals of
different components in :ref:`components` might be helpful.


.. _pipeflow_function:

The Pipeflow Function
=====================

.. autofunction:: pandapipes.pipeflow.pipeflow
