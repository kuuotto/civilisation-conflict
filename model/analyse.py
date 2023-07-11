import numpy as np
from scipy import stats
from model import ipomdp_solver, universe
from typing import Dict, Tuple


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


def count_particles_by_depth(model: universe.Universe) -> Dict:
    """
    For each tree found in the given model instance, find all nodes and count the
    number of particles they have.

    Keyword arguments:
    model: The model object to analyse.

    Returns:
    A dictionary with tree signatures as keys and dictionaries as values. Each
    value-dictionary has the depths (integer) as keys and lists as values where each
    list value represents the number of particles in a single node at that depth.
    """

    # initialise result dictionary
    result = dict()

    # go through all trees
    all_trees = (tree for ag in model.agents for tree in ag.forest.trees.values())
    for tree in all_trees:
        tree_result = dict()

        # crawl through all nodes in the tree
        for root_node in tree.root_nodes:
            _crawl_node(node=root_node, depth=0, result=tree_result)

        result[tree.signature] = tree_result

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


def prop_successful_lower_tree_queries(
    model: universe.Universe,
) -> Dict:
    """
    Calculates the proportion of lower tree queries that are successful in each
    level > 0 tree.

    Keyword arguments:
    model: the model to analyse

    Returns:
    A dictionary with tree signatures as keys and tuples as values. The values in the
    tuples are in the following order:
    1. Number of queries
    2. Proportion of successful queries
    3. Proportion of queries where node was missing
    4. Proportion of queries where belief diverged
    5. Proportion of queries where some actions were unexpanded
    """

    result = dict()

    all_trees = (
        tree_signature
        for ag in model.agents
        for tree_signature in ag.forest.trees.keys()
    )
    for tree_signature in all_trees:
        tree_events = [
            event
            for event in model.log
            if event.event_type in (10, 11, 12, 13)
            and event.event_data == tree_signature
        ]

        n_queries = len(tree_events)
        n_successful = len([event for event in tree_events if event.event_type == 10])
        n_missing_node = len([event for event in tree_events if event.event_type == 11])
        n_diverged_belief = len(
            [event for event in tree_events if event.event_type == 12]
        )
        n_some_actions_unexpanded = len(
            [event for event in tree_events if event.event_type == 13]
        )

        tree_result = (0, 0, 0, 0, 0)
        if n_queries > 0:
            tree_result = (
                n_queries,
                n_successful / n_queries,
                n_missing_node / n_queries,
                n_diverged_belief / n_queries,
                n_some_actions_unexpanded / n_queries,
            )

        result[tree_signature] = tree_result

    return result


def t_confidence_interval(sample, conf_level=0.95) -> Tuple[float, float]:
    """
    Assuming the sample is drawn from a population that is normally distributed, this
    returns the conf_level confidence interval for the mean of that population.

    Returns:
    The estimate of the mean and its error margin (half of the length of the C.I.)
    """
    n = len(sample)
    sample_mean = np.mean(sample)
    sample_sd = np.std(sample, ddof=1)
    quantile = stats.t.ppf(q=(1 - (1 - conf_level) / 2), df=n - 1)
    error_margin = quantile * sample_sd / np.sqrt(n)

    return sample_mean, error_margin
