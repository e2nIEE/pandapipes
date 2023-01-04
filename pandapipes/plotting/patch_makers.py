# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
from matplotlib.patches import RegularPolygon, Rectangle, Circle, PathPatch, FancyArrow
from matplotlib.path import Path
from pandapower.plotting.plotting_toolbox import get_color_list, _rotate_dim2, get_angle_list, \
    get_list


def get_filled_list(filled, number_entries, name_entries="valves"):
    return get_list(filled, number_entries, "filled", name_entries)


def valve_patches(coords, size, **kwargs):
    polys, lines = list(), list()
    edgecolor = kwargs.pop('patch_edgecolor')
    colors = get_color_list(edgecolor, len(coords))
    lw = kwargs.get("linewidths", 2.)
    ls = kwargs.get("linestyle", '-')
    filled = kwargs.pop("filled", np.full(len(coords), 0, dtype=bool))
    filled = get_filled_list(filled, len(coords))
    pos = kwargs.pop("valve_position", 2)
    for geodata, col, filled_ind in zip(coords, colors, filled):
        p1, p2 = np.array(geodata[0]), np.array(geodata[-1])
        diff = p2 - p1
        angle = np.arctan2(*diff)
        vec_size = _rotate_dim2(np.array([0, size]), angle)
        centroid_tri1 = p1 + diff / pos - vec_size
        centroid_tri2 = p1 + diff / pos + vec_size
        face_col = "w" if not filled_ind else col
        polys.append(RegularPolygon(centroid_tri1, numVertices=3, radius=size, orientation=-angle,
                                    ec=col, fc=face_col, lw=lw, ls=ls))
        polys.append(RegularPolygon(centroid_tri2, numVertices=3, radius=size,
                                    orientation=-angle + np.pi / 3, ec=col, fc=face_col, lw=lw, ls=ls))
        lines.append([p1, p1 + diff / pos - vec_size / 2 * 3])
        lines.append([p2, p1 + diff / pos + vec_size / 2 * 3])
    return lines, polys, {"filled"}


def flow_control_patches(coords, size, **kwargs):
    """Creates a valve-like patch with a T on it"""
    polys, lines = list(), list()
    edgecolor = kwargs.pop('patch_edgecolor')
    colors = get_color_list(edgecolor, len(coords))
    lw = kwargs.get("linewidths", 2.)
    filled = kwargs.pop("filled", np.full(len(coords), 0, dtype=bool))
    filled = get_filled_list(filled, len(coords))
    controlled = kwargs.pop("controlled", np.full(len(coords), 1, dtype=bool))
    controlled = get_filled_list(controlled, len(coords))
    for geodata, col, filled_ind, controlled_ind in zip(coords, colors, filled, controlled):
        p1, p2 = np.array(geodata[0]), np.array(geodata[-1])
        center = (0.5 * (p1[0] + p2[0]), 0.5 * (p1[1] + p2[1]))
        diff = p2 - p1
        angle = np.arctan2(*diff)
        vec_size = _rotate_dim2(np.array([0, size]), angle)
        centroid_tri1 = p1 + diff / 2 - vec_size
        centroid_tri2 = p1 + diff / 2 + vec_size
        face_col = "w" if not filled_ind else col
        polys.append(RegularPolygon(centroid_tri1, numVertices=3, radius=size, orientation=-angle,
                                    ec=col, fc=face_col, lw=lw))
        polys.append(RegularPolygon(centroid_tri2, numVertices=3, radius=size,
                                    orientation=-angle + np.pi / 3, ec=col, fc=face_col, lw=lw))
        lines.append([p1, p1 + diff / 2 - vec_size / 2 * 3])
        lines.append([p2, p1 + diff / 2 + vec_size / 2 * 3])
        dx = 1.5 * size * np.sin(angle - np.pi/2)
        dy = 1.5 * size * np.cos(angle - np.pi/2)

        polys.append(FancyArrow(center[0], center[1], dx, dy, head_length=0,
                                head_width=2*size*controlled_ind, color=col))
    return lines, polys, {"filled", "controlled"}


def heat_exchanger_patches(coords, size, **kwargs):
    polys, lines = list(), list()
    facecolor = kwargs.get("patch_facecolor", "w")
    edgecolor = kwargs.get("patch_edgecolor", "k")
    facecolors = get_color_list(facecolor, len(coords))
    edgecolors = get_color_list(edgecolor, len(coords))

    lw = kwargs.get("linewidths", 2.)
    for geodata, face_col, edge_col in zip(coords, facecolors, edgecolors):
        p1, p2 = np.array(geodata[0]), np.array(geodata[-1])
        diff = p2 - p1
        m = 3 * size / 4
        direc = diff / np.sqrt(diff[0] ** 2 + diff[1] ** 2)
        normal = np.array([-direc[1], direc[0]])
        path1 = (p1 + diff / 2 + direc * m / 2) + normal * (size * 9 / 8)
        path2 = p1 + diff / 2 + direc * m / 2
        path3 = p1 + diff / 2 + normal * size / 3
        path4 = p1 + diff / 2 - direc * m / 2
        path5 = (p1 + diff / 2 - direc * m / 2) + normal * (size * 9 / 8)
        path = [path1, path2, path3, path4, path5]
        radius = size  # np.sqrt(diff[0]**2+diff[1]**2)/15

        pa = Path(path)
        polys.append(Circle(p1 + diff / 2, radius=radius, edgecolor=edge_col, facecolor=face_col,
                            lw=lw))
        polys.append(PathPatch(pa, fill=False, lw=lw, edgecolor=edge_col))
        lines.append([p1, p1 + diff / 2 - direc * radius])
        lines.append([p2, p1 + diff / 2 + direc * radius])
    return lines, polys, {}


def source_patches(node_coords, size, angles, **kwargs):
    """
    Creation function of patches for sources.

    :param node_coords: coordinates of the nodes that the sources belong to.
    :type node_coords: iterable
    :param size: size of the patch
    :type size: float
    :param angles: angles by which to rotate the patches (in radians)
    :type angles: iterable(float), float
    :param kwargs: additional keyword arguments (might contain parameter "offset")
    :type kwargs:
    :return: Return values are: \
        - lines (list) - list of coordinates for lines leading to source patches and diagonals\
        - polys (list of RegularPolygon) - list containing the load patches\
        - keywords (set) - set of keywords removed from kwargs
    """
    offset = kwargs.get("offset", size * 2)
    all_angles = get_angle_list(angles, len(node_coords))
    facecolor = kwargs.get("patch_faceolor", "w")
    edgecolor = kwargs.get("patch_edgecolor", "k")
    facecolors = get_color_list(facecolor, len(node_coords))
    edgecolors = get_color_list(edgecolor, len(node_coords))
    polys, lines = list(), list()
    for i, node_geo in enumerate(node_coords):
        p2 = node_geo + _rotate_dim2(np.array([0, offset]), all_angles[i])
        p_edge_left = p2 + _rotate_dim2(np.array([size, size]), all_angles[i])
        p_edge_right = p2 + _rotate_dim2(np.array([- size, size]), all_angles[i])
        p_ll = p2 + _rotate_dim2(np.array([-size, 0]), all_angles[i])
        polys.append(Rectangle(p_ll, 2 * size, 2 * size, angle=(- all_angles[i] / np.pi * 180),
                               fc=facecolors[i], ec=edgecolors[i]))
        lines.append((node_geo, p2))
        lines.append((p2, p_edge_left))
        lines.append((p2, p_edge_right))
    return lines, polys, {"offset"}


def pump_patches(coords, size, **kwargs):
    polys, lines = list(), list()
    edgecolor = kwargs.pop('patch_edgecolor')
    colors = get_color_list(edgecolor, len(coords))
    lw = kwargs.get("linewidths", 2.)
    for geodata, col in zip(coords, colors):
        p1, p2 = np.array(geodata[0]), np.array(geodata[-1])
        diff = p2 - p1
        angle = np.arctan2(*diff)
        vec_size = _rotate_dim2(np.array([0, size]), angle)
        line1 = _rotate_dim2(np.array([0, size * np.sqrt(2)]), angle - np.pi / 4)
        line2 = _rotate_dim2(np.array([0, size * np.sqrt(2)]), angle + np.pi / 4)
        radius = size

        polys.append(Circle(p1 + diff / 2, radius=radius, edgecolor=col, facecolor='w', lw=lw))

        lines.append([p1 + diff / 2 + vec_size, p1 + diff / 2 - vec_size + line1])
        lines.append([p1 + diff / 2 + vec_size, p1 + diff / 2 - vec_size + line2])

        lines.append([p1, p1 + diff / 2 - vec_size])
        lines.append([p2, p1 + diff / 2 + vec_size])
    return lines, polys, {}


def compressor_patches(coords, size, **kwargs):
    polys, lines = list(), list()
    edgecolor = kwargs.pop('patch_edgecolor')
    colors = get_color_list(edgecolor, len(coords))
    lw = kwargs.get("linewidths", 2.)
    for geodata, col in zip(coords, colors):
        p1, p2 = np.array(geodata[0]), np.array(geodata[-1])
        diff = p2 - p1
        angle = np.arctan2(*diff)
        vec_size = _rotate_dim2(np.array([0, size]), angle)
        vec_size_or = _rotate_dim2(np.array([0, size * 1.4 / 2]), angle + np.pi / 2)
        pul = p1 + 0.5 * diff + vec_size + vec_size_or/3  # lower left
        pur = p1 + 0.5 * diff + vec_size - vec_size_or/3  # lower right
        pll = p1 + 0.6 * diff - vec_size + vec_size_or    # upper left
        plr = p1 + 0.6 * diff - vec_size - vec_size_or    # upper richt
        radius = size

        polys.append(Circle(p1 + diff / 2, radius=radius, edgecolor=col, facecolor='w', lw=lw))

        lines.append([p1, p1 + diff / 2 - vec_size])
        lines.append([p2, p1 + diff / 2 + vec_size])

        lines.append([pll, pul])
        lines.append([plr, pur])

    return lines, polys, {}


def pressure_control_patches(coords, size, **kwargs):
    polys, lines = list(), list()
    edgecolor = kwargs.pop('patch_edgecolor')
    colors = get_color_list(edgecolor, len(coords))
    lw = kwargs.get("linewidths", 2.)
    for geodata, col in zip(coords, colors):
        p1, p2 = np.array(geodata[0]), np.array(geodata[-1])
        diff = p2 - p1
        angle = np.arctan2(*diff)
        vec_size = _rotate_dim2(np.array([0, size]), angle)
        vec_size_or = _rotate_dim2(np.array([0, size * 1.2 / 2]), angle + np.pi / 2)
        pll = p1 + diff / 2 - vec_size + vec_size_or
        plr = p1 + diff / 2 - vec_size - vec_size_or
        pul = p1 + diff / 2 + vec_size + vec_size_or/3
        pur = p1 + diff / 2 + vec_size - vec_size_or/3

        polys.append(Rectangle(pll, 2 * size, size * 1.2, angle=np.rad2deg(-angle + np.pi/2),
                               edgecolor=col, facecolor='w', lw=lw))

        lines.append([p1, p1 + diff / 2 - vec_size])
        lines.append([p2, p1 + diff / 2 + vec_size])

        lines.append([pll, pul])
        lines.append([plr, pur])

    return lines, polys, {}
