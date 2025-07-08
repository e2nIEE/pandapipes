# Copyright (c) 2020-2025 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from pandapipes.component_models.abstract_models.branch_models import BranchComponent
import logging


def get_all_branch_component_models():
    """
    Get all models of available branch components

    :return: branch model
    :rtype: list
    """
    def get_all_subclasses(cls):
        all_subclasses = list()
        for subclass in cls.__subclasses__():
            all_subclasses.append(subclass)
            all_subclasses.extend(get_all_subclasses(subclass))
        return all_subclasses

    all_branch_components = get_all_subclasses(BranchComponent)
    filtered = list()
    for bc in all_branch_components:
        try:
            bc.table_name()
            filtered.append(bc)
        except Exception as e:
            logging.info(f"branch component {bc} has no table name {e}")
    return filtered


def get_all_branch_component_table_names():
    """
    Get all table names of available branch components

    :return: table names
    :rtype: list
    """
    cm = get_all_branch_component_models()
    return [c.table_name() for c in cm]
