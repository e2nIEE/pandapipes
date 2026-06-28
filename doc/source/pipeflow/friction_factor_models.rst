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
:math:`\frac{\mathrm{d}\lambda}{\mathrm{d}\dot{m}}`,
which is essential for :ref:`constructing the Jacobian matrix <jacobian>`
in the Newton‑Raphson solver.

.. note::
    Because the friction factor depends only on the magnitude of the mass flow
    (via :math:`Re = C|\dot{m}|`), :math:`\lambda(\dot{m})` must be an even
    function of :math:`\dot{m}`.

    Its derivative :math:`\frac{\mathrm{d}\lambda}{\mathrm{d}\dot{m}}`
    is therefore an odd function – it changes sign when the flow direction reverses.

    Since :math:`\lambda` decreases as the Reynolds number increases
    (as seen in the Moody chart), :math:`\frac{\mathrm{d}\lambda}{\mathrm{d}\dot{m}}`
    must be negative for positive mass flow.


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

.. autoclass:: RegimeAwareFrictionFactorModel
    :members:


.. _custom_friction_factor_models:

Custom Friction Factor Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can implement the :class:`~FrictionFactorModel` protocol and pass it to :ref:`pipeflow function<pipeflow>`:

.. code-block:: python

    pp.pipeflow(net, friction_model=MyFrictionFactorModel())
