from model import ipomdp_solver
from typing import Dict


def count_streaks(data):
    """
    Counts the streaks from time stamps.
    data should be a pandas Series which contains the
    time stamps of the events of interest.
    """
    if len(data) == 0:
        return dict()

    prev_t = data[0]
    streak = 1
    streaks = dict()
    for t in data[1:]:
        if t == prev_t + 1:
            # streak continues
            streak += 1
        else:
            # streak broken
            if streak not in streaks.keys():
                streaks[streak] = 1
            else:
                streaks[streak] += 1
            streak = 1
        prev_t = t

    # add final streak
    if streak not in streaks.keys():
        streaks[streak] = 1
    else:
        streaks[streak] += 1

    return streaks


def count_particles_by_depth(tree: ipomdp_solver.Tree):
    """
    At each depth found in the tree, find all nodes and count the number of
    particles they have.

    Returns:
    A dictionary with depth as key and list of particle counts as value.
    """

    # initialise result dictionary
    result = {}

    # crawl through all nodes
    for root_node in tree.root_nodes:
        _crawl_node(node=root_node, depth=0, result=result)

    return result


def _crawl_node(node: ipomdp_solver.Node, depth, result: Dict):
    # count number of particles in this node
    n_particles = len(node.particles)

    # add to result dictionary
    if depth not in result:
        result[depth] = []
    result[depth].append(n_particles)

    # recursively investigate child nodes
    for _, child_node in node.child_nodes.items():
        _crawl_node(node=child_node, depth=depth + 1, result=result)
