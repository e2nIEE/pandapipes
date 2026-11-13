.. currentmodule:: pandapipes.pf.friction_factor_model

.. _friction_factor_models:

**********************
Friction Factor Models
**********************

The friction factor :math:`\lambda` appears in the Darcy‑Weisbach type
pressure drop equations used by pandapipes. The exact form of the pressure
drop equation depends on the fluid model (incompressible or compressible).

All friction factor models implement the :class:`~FrictionFactorModel`
protocol and provide both :math:`\lambda` and its derivative
:math:`\frac{\mathrm{d}\lambda}{\mathrm{d}Re}`,
which is essential for :ref:`constructing the Jacobian matrix <jacobian>`
in the Newton‑Raphson solver.


.. autoclass:: FrictionFactorModel
    :members:


.. _built_in_friction_factor_models:

Built-in Friction Factor Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are several models provided in pandapipes by default.

.. autoclass:: Nikuradse
    :members:

.. autoclass:: SwameeJain
    :members:

.. autoclass:: Colebrook
    :members:


.. _custom_friction_factor_models:

Custom Friction Factor Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can implement the :class:`~FrictionFactorModel` protocol and pass it to :ref:`pipeflow function<pipeflow>`:

.. code-block:: python

    pp.pipeflow(net, friction_model=MyFrictionFactorModel())
