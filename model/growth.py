import numpy as np
import math
from numba import njit


def influence_radius(tech_level):
    """
    Gives the radius of influence as a function of the technology level
    of the civilisation.

    TODO: define more rigorously
    """
    return 0.1 * np.tan(tech_level * (math.pi / 2))


def inv_influence_radius(inf_radius):
    """
    Gives the tech level corresponding to a given influence radius
    """
    return (2 / math.pi) * np.arctan(10 * inf_radius)


@njit
def sigmoid_growth(time, speed, takeoff_time):
    """
    Gives the current technology level for the agent. Assumes a
    sigmoid-shaped growing curve for the technological capabilities of
    civilisations.

    Arguments can be individual numbers of NumPy arrays of equal length.
    """
    exponent = -speed * (time - takeoff_time)
    return 1 / (1 + np.exp(exponent))


def tech_level(state, model):
    """
    Calculate the tech level(s) of the agent(s) in state.

    State can be an individual agent state (in which case a single tech level
    is returned), a model state (in which case n_agents tech levels are
    returned) or a higher-dimensional collection of model states.

    Keyword arguments:
    state - a NumPy array where the last dimension corresponds to an individual
            agent state
    model - a Universe with a corresponding agent growth function saved in
            its agent_growth attribute
    """
    if model.agent_growth == sigmoid_growth:
        return sigmoid_growth(
            time=state[..., 0], speed=state[..., 2], takeoff_time=state[..., 3]
        )

    raise NotImplementedError()
