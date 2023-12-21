# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import copy
import os

import numpy as np
from pandapipes.converter.stanet.preparing_steps import connection_pipe_section_table, get_pipe_geo


# Please note: The whole file is not being tested, please use any functions herein with care!


def sort_by_pos(group):
    group = group.sort_values("posL")
    group_characteristics(group)
    return group


def sort_by_flow(group):
    group["follower"] = group.FLUSSB.apply(lambda b: group.index[group.FLUSSA == b][0]
                                           if b in group.FLUSSA.values else np.NaN)
    ls = list(group.loc[~group.FLUSSA.isin(group.FLUSSB)].index)
    assert len(ls) == 1
    while len(ls) < len(group):
        ls.append(int(group.follower[ls[-1]]))
    group = group.loc[ls]
    group = group_characteristics(group)
    assert np.allclose(group.posL.values, np.sort(group.posL.values))

    return group


def group_characteristics(group):
    group["nodes_in_group"] = len(group)
    group["current_node"] = np.arange(1, len(group) + 1)
    pos_l_previous = np.zeros(len(group))
    pos_l_previous[1:] = group["posL"].values[:-1]
    group["len_previous"] = group["posL"] - pos_l_previous
    group["len_remaining"] = group["RORL"] - group["posL"]
    return group


def remove_connections_zero_length(stored_data, cutoff_length=1e-1, save_path=None,
                                   write_header=False, save_excel=False):
    connections = stored_data["house_connections"].copy()
    used_connections = np.isin(connections.STANETID, stored_data["house_pipes"].CLIENTID) \
        | np.isin(connections.STANETID, stored_data["house_pipes"].CLIENT2ID)
    unused_ids = connections.STANETID.loc[~used_connections].values

    pipe_geodata = get_pipe_geo(stored_data, modus="main")
    house_pipe_geodata = get_pipe_geo(stored_data, modus="houses")

    cons = connection_pipe_section_table(stored_data, pipe_geodata, house_pipe_geodata, True)
    cons2 = cons.groupby("SNUM").apply(sort_by_pos)

    hh_pipes = stored_data["house_pipes"].copy()
    node_data = stored_data["nodes"].copy()

    def replace_node(con_dat, repl, node_name):
        node_rec = node_data.loc[node_data.KNAM == node_name, "RECNO"]
        assert len(node_rec) == 1
        node_rec = node_rec.iloc[0]
        stid = con_dat.STANETID_connections
        hh_pipes.loc[hh_pipes.CLIENTID == stid, ["ANFNAM", "CLIENTTYP", "CLIENTID", "CLIENTNO"]] = \
            node_name, 1, "NODE " + node_name, node_rec
        hh_pipes.loc[hh_pipes.CLIENT2ID == stid,
                     ["ENDNAM", "CLIENT2TYP", "CLIENT2ID", "CLIENT2NO"]] = \
            node_name, 1, "NODE " + node_name, node_rec
        repl[stid] = {"type": "node", "id": node_name, "no": node_rec}

    def replace_con(con_dat, repl, con_id):
        con_rec = connections.loc[node_data.STANETID == con_id, "RECNO"]
        assert len(con_rec) == 1
        con_rec = con_rec.iloc[0]
        stid = con_dat.STANETID_connections
        hh_pipes.loc[hh_pipes.CLIENTID == stid, ["CLIENTTYP", "CLIENTID", "CLIENTNO"]] = \
            35, con_id, con_rec
        hh_pipes.loc[hh_pipes.CLIENT2ID == stid, ["CLIENT2TYP", "CLIENT2ID", "CLIENT2NO"]] = \
            35, con_id, con_rec
        repl[stid] = {"type": "connection", "id": con_id, "no": con_rec}

    replacements = dict()
    cons2["sorting"] = np.arange(len(cons2))
    previous_con = None
    for con_data in cons2.itertuples():
        if con_data.len_previous < cutoff_length:
            if con_data.current_node == 1:
                replace_node(con_data, replacements, con_data.ANFNAM_pipes)
            else:
                if previous_con in replacements:
                    if replacements[previous_con]["type"] == "node":
                        replace_node(con_data, replacements, replacements[previous_con]["id"])
                    else:
                        replace_con(con_data, replacements, replacements[previous_con]["id"])
        previous_con = copy.deepcopy(con_data.STANETID_connections)

    previous_con = None
    previous_remaining_len = 0.
    previous_snum = cons2.sort_values("sorting").SNUM.iloc[-1]
    for con_data in cons2.sort_values("sorting").itertuples():
        if (con_data.len_remaining - previous_remaining_len < cutoff_length
                and con_data.SNUM == previous_snum) or con_data.len_remaining < cutoff_length:
            if con_data.current_node == con_data.nodes_in_group:
                replace_node(con_data, replacements, con_data.ANFNAM_pipes)
            else:
                if previous_con in replacements:
                    if replacements[previous_con]["type"] == "node":
                        replace_node(con_data, replacements, replacements[previous_con]["id"])
                    else:
                        replace_con(con_data, replacements, replacements[previous_con]["id"])
        previous_con = copy.deepcopy(con_data.STANETID_connections)
        previous_snum = copy.deepcopy(con_data.SNUM)
        previous_remaining_len = copy.deepcopy(con_data.len_remaining)

    true_connections = connections.loc[~connections.STANETID.isin(set(replacements.keys())
                                                                  | set(unused_ids))]
    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        true_connections.to_csv(os.path.join(save_path, "hh_connection_nodes.csv"),
                                sep=";", decimal=",", index=False, header=write_header)
        hh_pipes.to_csv(os.path.join(save_path, "hh_pipes.csv"), sep=";", decimal=",",
                        index=False, header=write_header)
        if save_excel:
            true_connections.to_excel(os.path.join(save_path, "hh_connection_nodes.xlsx"))
            hh_pipes.to_excel(os.path.join(save_path, "hh_pipes.xlsx"))
    return true_connections, hh_pipes
