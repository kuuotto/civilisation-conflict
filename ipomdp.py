# %%
import numpy as np
from model import Civilisation, sigmoid_growth

def transition(model, action):
    """
    Given a model in a given state and an action, this function produces all 
    the possible future states and the associated probabilities. This can be
    used as the transition function for all agents.

    Keyword arguments:
    model: a Universe object
    action: a dictionary, with keys 'actor' (a Civilisation) and 'type'
            (either a string ('hide' for hiding or '-' for no action) or a 
             Civilisation that was attacked)

    Returns: two tuples, the first for possible new states (represented as 
             NumPy arrays) and the second for the associated probabilities
    """
    # get current state as a n x k NumPy array
    state = model.get_state().copy()

    # always tick everyone's time by one
    state[:, 0] += 1
    
    if action['type'] == '-' or action['type'] == 'hide':

        if action['type'] == 'hide':
            # if actor hides, additionally update their visibility
            state[action['actor'].unique_id, 1] *= model.visibility_multiplier
            
        return (state,), (1,)

    elif isinstance(action['type'], Civilisation):
        # if there is an attack, see if it is successful or not
        target = action['type']

        # create states corresponding to the target surviving or dying
        # TODO: room for optimisation
        survived_state = state
        destroyed_state = state.copy()
        destroyed_state[target.unique_id, 0] = 0 # reset time
        destroyed_state[target.unique_id, 1] = 1 # visibility factor

        if action['actor'].tech_level > target.tech_level:
            # civilisation is destroyed
            return (destroyed_state,), (1,)
        elif action['actor'].tech_level == target.tech_level:
            # if the civilisations are evenly matched, the target is destroyed
            # with a 50% probability
            return (survived_state, destroyed_state), (0.5, 0.5)
        else:
            # target civilisation is not destroyed
            return (survived_state,), (1,)

    else:
        raise Exception(f"Action format was incorrect: {action}")


def reward(agent, action, 
           rewards={'destroyed': -1, 'hide': -0.01, 'attack': 0}):
    """
    Given an agent and an action, this function calculates the reward from 
    that action for the agent.

    Keyword arguments:
    agent: a Civilisation whose reward to calculate
    action: a dictionary, with keys 'actor' (a Civilisation) and 'type'
            (either a string ('hide' for hiding or '-' for no action) or a 
             Civilisation that was attacked)
    """
    # TODO: this function is probably called a lot with a skip action, could
    # be sped up in that case.

    actor = action['actor']

    if isinstance(action['type'], Civilisation):
        # action was an attack
        target = action['type']

        if target == agent and target.tech_level < actor.tech_level:
            return rewards['destroyed']
        elif target == agent and target.tech_level == actor.tech_level:
            return rewards['destroyed'] / 2
        elif actor == agent:
            return rewards['attack']

    elif action == 'hide' and actor == agent:
        return rewards['hide']

    # in all other cases the reward is 0
    return 0

def sample_init(n_samples, n_agents, level, rng, agent_growth, 
                **agent_growth_kwargs):
    """
    Samples the initial beliefs of an agent.

    If level = 0, agents only have beliefs about the environment states. 
    Each environment state is an array of size n_agents x k, where k is 
    the length of an individual agent state (typically k=4 for sigmoid growth).
    Therefore, the sample will be a NumPy array of size 
    n_samples x n_agents x k.

    If level = 1, agents have beliefs about the environment states and beliefs
    about others' beliefs about the environment states. So for every environment
    state (size n_agents x k) that we sample, there are also associated 
    belief distributions about what others believe about the environment. These 
    distributions are represented by samples (of size n_samples) from each of
    these distributions. There are n_agents - 1 of these distributions. The 
    resulting NumPy array will be of size
        n_samples x (1 + (n_agents - 1) * n_samples) x n_agents x k
    where in the outer parentheses the 1 represents the original agent's 
    environment sample, the (n_agents - 1) is the number of other agents, each 
    of which has a sample of size n_samples representing that agents belief
    about the environment.

    Keyword arguments:
    n_samples: number of samples to generate
    n_agents: number of agents in the model
    level: level of interactive beliefs of the agent. 0 or 1
    rng: random number generator (from the Universe object or elsewhere)
    agent_growth: growth function used
    **agent_growth_kwargs: arguments used for the growth function

    Returns:
    A NumPy array (see description above for the size)
    """

    # determine the number of values needed to describe an agent
    if agent_growth == sigmoid_growth:
        k = 4
    else:
        raise NotImplementedError()

    # initialise array of samples
    if level == 0:
        sample = np.zeros((n_samples, n_agents, k))
    elif level == 1:
        sample = np.zeros((n_samples, 1 + (n_agents - 1)*n_samples, n_agents, k))
    else:
        raise NotImplementedError("Levels above 1 are not supported")

    # determine the values or range of the growth parameters
    if agent_growth == sigmoid_growth:

        if ("speed" in agent_growth_kwargs and 
            "takeoff_time" in agent_growth_kwargs):
            # every agent has the same growth parameters
            speed_range = (agent_growth_kwargs["speed"], 
                           agent_growth_kwargs["speed"])
            takeoff_time_range = (agent_growth_kwargs["takeoff_time"],
                                  agent_growth_kwargs["takeoff_time"])
        elif ("speed_range" in agent_growth_kwargs and 
              "takeoff_time_range" in agent_growth_kwargs):
            # growth parameters are sampled from the given ranges
            speed_range = agent_growth_kwargs["speed_range"]
            takeoff_time_range = agent_growth_kwargs["takeoff_time_range"]
        else:
            raise Exception("Sigmoid growth parameters are incorrect")


    if level == 0:

        # in every sample and every agent, initial time is 0
        sample[:, :, 0] = 0
        # likewise, initially everyone has a visibility factor of 1
        sample[:, :, 1] = 1

        if agent_growth == sigmoid_growth:
            sample[:, :, 2] = rng.uniform(*speed_range, 
                                          size=(n_samples, n_agents))
            sample[:, :, 3] = rng.integers(*takeoff_time_range, 
                                           size=(n_samples, n_agents))

    elif level == 1:

        sample[:, :, :, 0] = 0
        sample[:, :, :, 1] = 1

        if agent_growth == sigmoid_growth:
            sample[:, :, :, 2] = rng.uniform(*speed_range,
                    size=(n_samples, 1 + (n_agents - 1) * n_samples, n_agents))
            sample[:, :, :, 3] = rng.integers(*takeoff_time_range,
                    size=(n_samples, 1 + (n_agents - 1) * n_samples, n_agents))

    return sample

