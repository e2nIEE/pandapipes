# -*- coding: utf-8 -*-

# Copyright (c) 2016-2022 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from pandapower.auxiliary import _detect_read_write_flag, write_to_net
from pandapower.control.basic_controller import Controller

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


class LogicControl(Controller):
    """
    Class representing a generic time series logic controller for a specified element and variable.
    The input is designed to take a constant value, or be initialised from a designeated controller using
    the cascade solving order (Note: master controllers are solved first, the value is set within this
    logic controller).

    :param net: The pandapipes network in which the element is created
    :type net: pandapipesNet
    :param element: Output element name to control
    :type element: string
    :param variable: The elements parameter variable to control
    :type variable: string
    :param element_index: the PIP index of the element to control
    :type element_index: float, default 0
    :param input_1: The logic blocks input 1 - to implement a static value, set input 1 to desired constant
    :type input_1: float, default None
    :param input_2: The logic blocks input 1 - to implement a static value, set input 1 to desired constant
    :type input_2: float, default None
    :param logic_type: The type of logic applied inside the block
    :type logic_type: string, default None 'Low' selector
    :param scale_factor: Salce to apply to input or output ** Not implemented yet **
    :type scale_factor: float, default 1.0
    :Example:
            >>> LogicControl(net, element='dyn_circ_pump', variable='desired_mv', element_index=dyn_main_pmp_0, \\
                                             level=2, order=1, logic_type= 'low')
    """

    def __init__(self, net, element, variable, element_index, input_1=None, input_2=None, logic_type='Low',
                 scale_factor=1.0, in_service=True, recycle=True, order=-1, level=-1,
                 drop_same_existing_ctrl=False, matching_params=None, name=None,
                 initial_run=False, **kwargs):
        # just calling init of the parent
        if matching_params is None:
            matching_params = {"element": element, "variable": variable,
                               "element_index": element_index}
        super().__init__(net, in_service=in_service, recycle=recycle, order=order, level=level,
                         drop_same_existing_ctrl=drop_same_existing_ctrl,
                         matching_params=matching_params, initial_run=initial_run,
                         **kwargs)

        # Default inputs
        self.input_1 = input_1
        self.input_2 = input_2
        # Output Varaible : must be known!
        self.logic_type = logic_type
        self.element_index = element_index
        self.element = element
        self.values = None
        self.scale_factor = scale_factor
        self.applied = False
        self.write_flag, self.variable = _detect_read_write_flag(net, element, element_index, variable)
        self.set_recycle(net)
        self.name = name

    def time_step(self, net, time):
        """
        Get the values of the element from data source
        Write to pandapower net by calling write_to_net()
        If ConstControl is used without a data_source, it will reset the controlled values to the initial values,
        preserving the initial net state.
        """
        self.applied = False

        if self.input_1 is None:
            logger.warning("Logic input variable 1 is not initialised, or controller order is incorrect!")
        if self.input_2 is None:
            logger.warning("Logic input variable 2 is not initialised, or controller order is incorrect!")

        if isinstance(self.element_index, LogicControl):
            # element output is to a logic block controller, thus requires setting attribute of the controller
            if self.logic_type == 'min':
                self.element_index.__setattr__(self.variable, min(self.input_1, self.input_2))
            elif self.logic_type == 'max':
                self.element_index.__setattr__(self.variable, max(self.input_1, self.input_2))
            else:
                raise NotImplementedError("Sorry, logic type not implemented yet")
        else:
            # element index belongs to a standard control component - pump/valve etc.
            if self.logic_type == 'max':
                write_to_net(net, self.element, self.element_index, self.variable, max(self.input_1, self.input_2),
                         self.write_flag)
            elif self.logic_type == 'min':
                write_to_net(net, self.element, self.element_index, self.variable, min(self.input_1, self.input_2),
                         self.write_flag)
            else:
                raise NotImplementedError("Sorry, logic type not implemented yet")

    def is_converged(self, net):
        """
        Actual implementation of the convergence criteria: If controller is applied, it can stop
        """
        return self.applied

    def control_step(self, net):
        """
        Set applied to True, which means that the values set in time_step have been included in the load flow calculation.
        """
        self.applied = True

    def __str__(self):
        return super().__str__() + " [%s.%s]" % (self.element, self.variable)
