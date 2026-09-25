import numpy as np

from abc import ABC, abstractmethod
from typing import Generic, TypeVar
import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")
N = TypeVar("N")


class DiagnosticFunction(ABC, Generic[N, T]):
    """
    Base class for all pandapipes diagnostic checks.
    """

    def __init__(self):
        self.out = logger

    @abstractmethod
    def diagnostic(self, net, **kwargs):
        """
        Run diagnostic check on the network.

        Returns:
            result object or None
        """
        pass

    @abstractmethod
    def report(self, error, result):
        """
        Print diagnostic report.
        """
        pass


def check_number(element, element_index, column):
    try:
        nan_check = np.isnan(element[column])
        if nan_check or isinstance(element[column], bool):
            return element_index
    except TypeError:
        return element_index


def check_boolean(element, element_index, column):
    if element[column] not in [True, False, 0, 1, 0.0, 1.0]:
        return element_index


def check_greater_equal_zero(element, element_index, column):
    if check_number(element, element_index, column) is None:
        if element[column] < 0:
            return element_index
    else:
        return element_index


def check_greater_zero(element, element_index, column):
    if check_number(element, element_index, column) is None:
        if element[column] <= 0:
            return element_index
    else:
        return element_index


def check_pos_int(element, element_index, column):
    if check_number(element, element_index, column) is None:
        if not ((element[column] % 1 == 0) and element[column] > 0):
            return element_index
    else:
        return element_index


def check_existing_junction(element, element_index, column, junction_index):
    if element[column] not in junction_index:
        return element_index



def diagnostic(net, report=True, return_result_dict=True, **kwargs):
    from pandapipes.diagnostic.diagnostic import Diagnostic

    diag = Diagnostic()
    return diag.diagnose_network(
        net,
        report=report,
        return_result_dict=return_result_dict,
        **kwargs,
    )