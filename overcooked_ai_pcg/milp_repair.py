import numpy as np
from docplex.mp.model import Context, Model
from overcooked_ai_py import read_layout_dict
import gc

from overcooked_ai_pcg.helper import lvl_number2str, lvl_str2number, obj_types


def read_in_fix_demo_layout():
    raw_lvl = read_layout_dict("milp_demo")
    raw_lvl = raw_lvl['grid'].split('\n')
    return lvl_str2number(raw_lvl)


# Adds constraints that ensure exactly one object is present in each cell
#
# mdl:                the milp model
# all_objects:        a list of all object variables [[W_i], [P_i], ...]
def add_object_placement(mdl, all_objects):
    # Transpose the given matrix and ensure exactly one object per graph node
    for cur_node in zip(*all_objects):
        mdl.add_constraint(sum(cur_node) == 1)


# Adds reachability constraints to milp
#
# mdl:                the milp model
# graph:              an adjacency list
# source_objects:     objects that must reach the sink objects [[P_i], ...]
# sink_objects:       objects that must be reached by the source objects [[K_i], [D_i], ...]
# blocking_objects:   a list of object types that impede movement [[W_i], ...]
# cnt:                integer to remember number of times add_reachability function is called
#                     needed because different calls should produce params with different names
#
# post condition: these constraints ensure that a path exists from some source
#                 object to all sink objects
def add_reachability(mdl, graph, source_objects, sink_objects,
                     blocking_objects, cnt):
    # Transpose the blocking objects matrix so all blocking objects for
    # a given node are easily accessible.
    blocking = list(zip(*blocking_objects))

    # Setup a flow network for each edge in the graph
    n_nodes = len(graph)
    # Add a flow variable for each edge in the graph
    # flow: the flow leaving node i
    # rev: flow edges entering node i
    flow = [[] for i in range(n_nodes)]
    rev = [[] for i in range(n_nodes)]
    for i, neighbors in enumerate(graph):
        for j in neighbors:
            f = mdl.integer_var(name='p_{}_{}-{}'.format(i, j, cnt),
                                lb=0,
                                ub=n_nodes)
            flow[i].append(f)
            rev[j].append(f)

    # Add supply and demand variables for the source and sink
    supplies = []
    demands = []
    for i in range(n_nodes):
        f = mdl.integer_var(name='p_s_{}-{}'.format(i, cnt), lb=0, ub=n_nodes)
        supplies.append(f)
        f = mdl.integer_var(name='p_{}_t-{}'.format(i, cnt), lb=0, ub=1)
        demands.append(f)
    # Add a flow conservation constraint for each node (outflow == inflow)
    for i in range(n_nodes):
        mdl.add_constraint(supplies[i] + sum(rev[i]) == demands[i] +
                           sum(flow[i]))

    # Add capacity constraints for each edge ensuring that no flow passes through a blocking object
    for i, neighbors in enumerate(flow):
        blocking_limits = [n_nodes * b for b in blocking[i]]
        for f in neighbors:
            mdl.add_constraint(f + sum(blocking_limits) <= n_nodes)

    # Place a demand at this object location if it contains a sink type object.
    sinks = list(zip(*sink_objects))
    for i in range(n_nodes):
        mdl.add_constraint(sum(sinks[i]) == demands[i])

    # Allow this node to have supply if it contains a source object
    sources = list(zip(*source_objects))
    for i in range(n_nodes):
        capacity = sum(n_nodes * x for x in sources[i])
        mdl.add_constraint(supplies[i] <= capacity)


# Adds edit distance cost function and constraints for fixing the level with minimal edits.
#
# graph:              an adjacency list denoting allowed movement
# objects:            a list [([(T_i, O_i)], Cm, Cc), ...] representing the cost of moving each
#                     object by one edge (Cm) and the cost of an add or delete (Cc).
#                     T_i represents the object variable at node i
#                     O_i is a boolean value denoting whether node i originally contained T_i.
def add_edit_distance(mdl, graph, objects, add_movement=True):
    costs = []
    if not add_movement:
        for objects_in_graph, cost_move, cost_change in objects:
            for cur_var, did_contain in objects_in_graph:
                if did_contain:
                    costs.append(cost_change * (1 - cur_var))
                else:
                    costs.append(cost_change * cur_var)

    else:
        for obj_id, (objects_in_graph, cost_move,
                     cost_change) in enumerate(objects):

            # Setup a flow network for each edge in the graph
            n_nodes = len(graph)
            # Add a flow variable for each edge in the graph
            # flow: the flow leaving node i
            # rev: flow edges entering node i
            flow = [[] for i in range(n_nodes)]
            rev = [[] for i in range(n_nodes)]
            for i, neighbors in enumerate(graph):
                for j in neighbors:
                    f = mdl.integer_var(name='edit({})_{}_{}'.format(
                        obj_id, i, j),
                                        lb=0,
                                        ub=n_nodes)
                    costs.append(cost_move * f)
                    flow[i].append(f)
                    rev[j].append(f)

            # Add a supply if the object was in the current location.
            # Demands go everywhere.
            demands = []
            waste = []
            num_supply = 0
            for i, (cur_var, did_contain) in enumerate(objects_in_graph):
                f = mdl.integer_var(name='edit({})_{}_t'.format(obj_id, i),
                                    lb=0,
                                    ub=1)
                demands.append(f)

                # Add a second sink that eats any flow that doesn't find a home.
                # The cost of this flow is deleting the object.
                f = mdl.integer_var(name='edit({})_{}_t2'.format(obj_id, i),
                                    lb=0,
                                    ub=n_nodes)
                costs.append(cost_change * f)
                waste.append(f)

                # Flow conservation constraint (inflow == outflow)
                if did_contain:
                    # If we had a piece of this type in the current node, match it to the outflow
                    mdl.add_constraint(1 + sum(rev[i]) == demands[i] +
                                       sum(flow[i]) + waste[i])
                    num_supply += 1
                else:
                    mdl.add_constraint(
                        sum(rev[i]) == demands[i] + sum(flow[i]) + waste[i])

            # Ensure we place a piece of this type to match it to the demand.
            for (cur_var,
                 did_contain), node_demand in zip(objects_in_graph, demands):
                mdl.add_constraint(node_demand <= cur_var)

            # Ensure that the source and sink have the same flow.
            mdl.add_constraint(num_supply == sum(demands) + sum(waste))

    mdl.minimize(mdl.sum(costs))


def add_reachability_helper(source_labels, sink_labels, blocking_labels, mdl,
                            adj, objs, cnt):
    source_objects = [objs[obj_types.index(label)] for label in source_labels]
    sink_objects = [objs[obj_types.index(label)] for label in sink_labels]
    blocking_objects = [
        objs[obj_types.index(label)] for label in blocking_labels
    ]
    add_reachability(mdl, adj, source_objects, sink_objects, blocking_objects,
                     cnt)


def repair_lvl(np_lvl):
    # from IPython import embed
    # embed()
    n, m = np_lvl.shape

    deltas = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    # Build an adjacency list for the dynamics of Overcooked
    n_nodes = n * m
    adj = [[] for i in range(n_nodes)]
    border_nodes = []
    for i in range(n_nodes):
        cur_row = i // m
        cur_col = i % m
        is_border = False
        for dr, dc in deltas:
            nxt_row = cur_row + dr
            nxt_col = cur_col + dc
            if 0 <= nxt_row and nxt_row < n and 0 <= nxt_col and nxt_col < m:
                j = nxt_row * m + nxt_col
                adj[i].append(j)
            else:
                is_border = True
        if is_border:
            border_nodes.append(i)

    context = Context.make_default_context()
    # context.cplex_parameters.threads = 1
    with Model(context=context) as mdl:

        objs = []
        for obj_label in obj_types:
            curr_type = [
                mdl.integer_var(name='obj_{}_{}'.format(obj_label, i),
                                lb=0,
                                ub=1) for i in range(n_nodes)
            ]
            objs.append(curr_type)

        # ensure one cell contains one obj_type
        add_object_placement(mdl, objs)

        # Ensure that all cells on the boundary are walls
        not_allowed_on_border = []
        for label in "12 ":
            i = obj_types.index(label)
            not_allowed_on_border += [objs[i][j] for j in border_nodes]
        mdl.add_constraint(sum(not_allowed_on_border) <= 0)

        # Player1 and 2 show up exactly once
        mdl.add_constraint(sum(objs[obj_types.index("1")]) == 1)
        mdl.add_constraint(sum(objs[obj_types.index("2")]) == 1)

        # At least one onion, dish plate, pot, and serve point
        mdl.add_constraint(sum(objs[obj_types.index("O")]) >= 1)
        mdl.add_constraint(sum(objs[obj_types.index("D")]) >= 1)
        mdl.add_constraint(sum(objs[obj_types.index("P")]) >= 1)
        mdl.add_constraint(sum(objs[obj_types.index("S")]) >= 1)

        # Upper bound number of onion, dish plate, pot, and serve point
        mdl.add_constraint(sum(objs[obj_types.index("O")]) <= 2)
        mdl.add_constraint(sum(objs[obj_types.index("D")]) <= 2)
        mdl.add_constraint(sum(objs[obj_types.index("P")]) <= 2)
        mdl.add_constraint(sum(objs[obj_types.index("S")]) <= 2)

        # Upper bound total number of onion, dish plate, pot, and serve point
        mdl.add_constraint(
            sum(objs[obj_types.index("O")]) + sum(objs[obj_types.index("D")]) +
            sum(objs[obj_types.index("P")]) +
            sum(objs[obj_types.index("S")]) <= 6)

        # reachability
        source_labels = "1"
        sink_labels = "ODPS2 "
        blocking_labels = "XODPS"
        add_reachability_helper(source_labels, sink_labels, blocking_labels,
                                mdl, adj, objs, 0)

        # add edit distance objective
        objects = []
        cost_move = 1
        cost_change = 20
        for cur_idx, cur_obj in enumerate(objs):
            objects_in_graph = []
            for r in range(n):
                for c in range(m):
                    i = r * m + c
                    objects_in_graph.append((cur_obj[i], cur_idx == np_lvl[r,
                                                                           c]))
            objects.append((objects_in_graph, cost_move, cost_change))

        add_edit_distance(mdl, adj, objects)

        solution = mdl.solve()

        def get_idx_from_variables(solution, node_id):
            for i, obj_var in enumerate(objs):
                if solution.get_value(obj_var[node_id]) == 1:
                    return i
            return -1

        # Extract the new level from the milp model
        new_lvl = np.zeros((n, m))
        for r in range(n):
            for c in range(m):
                i = r * m + c
                new_lvl[r, c] = get_idx_from_variables(solution, i)

        del solution
        gc.collect()

        return new_lvl.astype(np.uint8)


def main():
    np_lvl = read_in_fix_demo_layout()
    np_lvl = np_lvl.astype(np.uint8)
    before_fix = lvl_number2str(np_lvl)
    print(before_fix)
    after_fix = lvl_number2str(repair_lvl(np_lvl).astype(np.uint8))
    print(after_fix)


if __name__ == "__main__":
    main()
