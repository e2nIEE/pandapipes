.. _pipeflow_options:

****************
Pipeflow Options
****************

When running a pipeflow, you can choose between different calculation modes, turn on or off a lot
of different functionalities, or change input parameters and boundary conditions that shall be
considered. In this section you can find information on how the options can be set and how they
influence the pipeflow calculation.


.. _option_order:

The Option Order
================

The options in pandapipes are considered in a very specific order as there are different ways of how
to define your own options and how settings override one another depending on how they are defined.
There are four layers that define the final option setup. Those layers are:

#. The default options (found in the pipeflow_setup). There all necessary options are defined \
   with default values. This means that a pipeflow will not usually lead to an error of not having \
   defined a specific option.
#. Some default parameters in the :ref:`pipeflow call <pipeflow_function>` are already implemented \
   as explicit keyword arguments. The default options are overriden with those values.
#. It is possible to add user options (c.f. :ref:`user_options`). These user options will override \
   all default options defined before.
#. When calling the pipeflow function and explicitly overriding any arguments, this is identified \
   and even the user options are overriden again.

The available options and how they can be set are described in the following.


.. _initialize_options:

Initialize Option Function
==========================

The final option setup is generated and saved under net["_options"] by the following internal
function:

.. autofunction:: pandapipes.pf.pipeflow_setup.init_options


.. _user_options:

Setting User Options
====================

User options will be stored in a pandapipes network and will always override the default options,
unless specified otherwise during the :ref:`pipeflow call <pipeflow_function>`.

.. autofunction:: pandapipes.pf.pipeflow_setup.set_user_pf_options


Auxiliary Option Functions
==========================

With the help of the following functions options can be set or retrieved. These functions are
usually only used internally in the pipeflow.

.. autofunction:: pandapipes.pf.pipeflow_setup.get_net_option

.. autofunction:: pandapipes.pf.pipeflow_setup.get_net_options

.. note:: If you set options with the following function, they will still be overridden once you \
          run another pipeflow.

.. autofunction:: pandapipes.pf.pipeflow_setup.set_net_option
