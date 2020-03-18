# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import os
import pandas as pd
import numpy as np


def get_data(path, std_type_category):
    """
    get_data.

    :param path:
    :type path:
    :param std_type_category:
    :type std_type_category:
    :return:
    :rtype:
    """
    if std_type_category == 'pump':
        path = os.path.join(path)
        data = pd.read_csv(path, sep=';', dtype=np.float64)
    elif std_type_category == 'pipe':
        data = pd.read_csv(path, sep=';', index_col=0).T
    else:
        raise AttributeError('std_type_category %s not implemented yet' % std_type_category)
    return data


def get_p_v_values(path):
    """

    :param path:
    :type path:
    :return:
    :rtype:
    """
    data = get_data(path, 'pump')
    p_values = data.values[:, 0]
    v_values = data.values[:, 1]
    degree = data.values[0, 2]
    return p_values, v_values, degree


def regression_function(p_values, v_values, degree):
    """
    Regression function...

    :param p_values:
    :type p_values:
    :param v_values:
    :type v_values:
    :param degree:
    :type degree:
    :return:
    :rtype:
    """
    z = np.polyfit(v_values, p_values, degree)
    reg_par = z
    return reg_par
