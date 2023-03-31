# %%
import numpy as np
from model.model import Civilisation, influence_radius, sigmoid_growth
from scipy.stats import multivariate_normal

#%%

def transition(state, action, model, agent_growth, in_place=False):
    """
    Given a model state and an action, this function samples from the
    distribution of all possible future states. In practice a (state, action) 
    combination only has one possible future state, except in the case of
    equally matched target and attacker, in which case the target is
    destoryed with a 0.5 probability (although note that that can only happen
    if tech levels are discrete, which they currently are not). This can be 
    used as the transition function for all agents, as they all have the same 
    transition model.

    Can also be supplied with multiple states and the same number of actions,
    in which case the states are all propagated forward with the corresponding
    actions.

    The random number generator of the model (the rng attribute) is used for
    random choices.
 
    Keyword arguments:
    state: representation of the system at time t-1. a NumPy array of size 
           (n_agents, k), where k is the length of an individual agent state
           representation (k=4 for sigmoid growth). If multiple states are 
           propagated simultaneously, should be of shape (n_samples, n_agents, 
           k).
    action: a dictionary, with keys 'actor' (a Civilisation) and 'type'
            (either a string ('hide' for hiding or '-' for no action) or a 
            Civilisation that was attacked). If multiple states are supplied, 
            this should be a list of length n_samples.
    model: a Universe
    agent_growth: growth function used
    in_place: boolean, whether to change the state object directly

    Returns: 
    A possible system state at time t. A NumPy array of the same shape as 
    the state argument.
    """
    # copy state so we don't change original
    if not in_place:
        state = state.copy()

    # if a single (state, action) combination is supplied, reshape
    if len(state.shape) == 2:
        state = state[np.newaxis]
    if not isinstance(action, list):
        action = [action]

    # make sure we have the same number of states and actions
    assert(len(state) == len(action))

    # always tick everyone's time by one
    state[:, :, 0] += 1

    for sample, act in enumerate(action):
    
        if act['type'] == 'hide':
            # if actor hides, additionally update their visibility
            state[sample, act['actor'].unique_id, 1] *= model.visibility_multiplier

        elif isinstance(act['type'], Civilisation):
            # if there is an attack, see if it is successful or not
            target_id = act['type'].unique_id
            actor_state = state[sample, act['actor'].unique_id, :]
            target_state = state[sample, target_id, :]

            if agent_growth == sigmoid_growth:
                actor_tech_level = sigmoid_growth(time=actor_state[0],
                                                  speed=actor_state[2],
                                                  takeoff_time=actor_state[3])
                target_tech_level = sigmoid_growth(time=target_state[0],
                                                   speed=target_state[2],
                                                   takeoff_time=target_state[3])
            else:
                raise NotImplementedError()

            if (actor_tech_level > target_tech_level or
                (actor_tech_level == target_tech_level and 
                 model.rng.random() > 0.5)):
                # civilisation is destroyed
                state[sample, target_id, 0] = 0 # reset time
                state[sample, target_id, 1] = 1 # visibility factor

    # return only a single state if a single state was supplied
    if state.shape[0] == 1:
        state = state[0]

    return state

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

def prob_observation(model, agent, state, action, observation, agent_growth, 
                     obs_noise_sd):
    """
    Returns the probability (density) of a given observation by “agent”, given 
    that the system is currently in state “state” and the previous action was 
    “action”.

    Keyword arguments:
    model: a Universe. Used for determining distances between civilisations.
    agent: the observing Civilisation
    state: a NumPy array of size n_agents x k, where k is the size of an
            individual agent state representation. For 
            agent_growth = sigmoid_growth, k = 4.
    action: a dictionary, with keys 'actor' (a Civilisation) and 'type'
            (either a string ('hide' for hiding or '-' for no action) or a 
            Civilisation that was attacked)
    observation: a NumPy array of length n_agents or n_agents + 1. The latter
                 corresponds to an observation where an attacker gets to know
                 the result of their attack last round (0 or 1). This binary
                 value is the last value in the array. A numpy.NaN 
                 denotes a civilisation that agent cannot observe yet, or the
                 agent itself.
    agent_growth: growth function used
    obs_noise_sd: standard deviation of observation noise (which is assumed to
                  follow a normal distribution centered around the true 
                  technosignature value)
    """
    agent_id = agent.unique_id
    n_agents = state.shape[0]

    # calculate tech levels
    if agent_growth == sigmoid_growth:
        agent_tech_levels = sigmoid_growth(time=state[:, 0], 
                                           speed=state[:, 2],
                                           takeoff_time=state[:, 3])
    else:
        raise NotImplementedError()

    # make sure that observation only contains observations on civilisations 
    # that agent can see
    nbr_ids = [nbr.unique_id
               for nbr in model.space.get_neighbors(
                   pos=agent.pos,
                   radius=influence_radius(agent_tech_levels[agent_id]),
                   include_center=False)]
    nbr_obs = observation[nbr_ids]
    unobserved_obs = np.delete(observation if observation.shape[0] == n_agents else observation[:-1], 
                               nbr_ids)

    if np.isnan(nbr_obs).any() or not np.isnan(unobserved_obs).all():
        #print("nans are wrong", np.isnan(nbr_obs).any(), not np.isnan(unobserved_obs).all(), nbr_ids, observation)
        return 0

    # check that if the agent attacked last round, the observation contains a 
    # bit indicating the success of the attack
    if (action['actor'] == agent and
        isinstance(action['type'], Civilisation) and
        (len(observation) != n_agents + 1 or
         observation[-1] not in (0, 1))):
        return 0

    # if there are no neighbours, then there is only one possible observation
    if len(nbr_ids) == 0:
        return 1

    # return density of multivariate normal. Observations from neighbours are
    # independent, centered around their respective technosignatures and
    # have a variance of obs_noise_sd^2
    # technosignature is a product of visibility factor and tech level
    nbr_technosignatures = agent_tech_levels[nbr_ids] * state[nbr_ids, 1]
    density = multivariate_normal.pdf(nbr_obs, 
                                      mean=nbr_technosignatures,
                                      cov=obs_noise_sd**2)

    return density

def sample_observation(state, action, agent, n_samples, model,
                       agent_growth, obs_noise_sd):
    """
    Returns n_samples of possible observations of “agent” when the system is
    currently in state “state” and the previous action was “action”.

    Model's random number generator (rng attribute) is used for sampling.

    Keyword arguments:
    state: a NumPy array of size n_agents x k, where k is the size of an
           individual agent state representation. For 
           agent_growth = sigmoid_growth, k = 4.
    action: a dictionary, with keys 'actor' (a Civilisation) and 'type'
            (either a string ('hide' for hiding or '-' for no action) or a 
            Civilisation that was attacked)
    agent: the observing Civilisation
    n_samples: number of observation samples to generate
    model: a Universe. Used for determining distances between civilisations and
           for random sampling.
    agent_growth: growth function used
    obs_noise_sd: standard deviation of observation noise (which is assumed to
                  follow a normal distribution centered around the true 
                  technosignature value)

    Returns:
    The observations. A NumPy array of size  n_samples x 
    (n_agents or n_agents + 1). The latter corresponds to an observation 
    where an attacker gets to know the result of their attack last round 
    (0 or 1). This binary value is the last value in the array. A numpy.NaN 
    denotes a civilisation that agent cannot observe yet, or the agent itself.
    If n_samples == 1, the NumPy array is squeezed into a 1d array.
    """
    agent_id = agent.unique_id
    n_agents = state.shape[0]
    obs_length = n_agents

    assert(len(state.shape) == 2)

    # if agent has attacked another last round, we need to include a result
    # bit in each observation
    if action['actor'] == agent and isinstance(action['type'], Civilisation):
        obs_length = n_agents + 1

        # determine if target was destroyed
        target_id = action['type'].unique_id
        target_destroyed = int(state[target_id, 0] == 0)

    # initialise array
    sample = np.full(shape=(n_samples, obs_length), fill_value=np.nan)

    # add success bit if necessary
    if obs_length == n_agents + 1:
        sample[:, -1] = target_destroyed

    # calculate tech levels
    if agent_growth == sigmoid_growth:
        agent_tech_levels = sigmoid_growth(time=state[:, 0], 
                                           speed=state[:, 2],
                                           takeoff_time=state[:, 3])
    else:
        raise NotImplementedError()

    # add observations from the civilisations the agent can see
    nbr_ids = [nbr.unique_id
               for nbr in model.space.get_neighbors(
                   pos=agent.pos,
                   radius=influence_radius(agent_tech_levels[agent_id]),
                   include_center=False)]

    if len(nbr_ids) > 0:
        nbr_technosignatures = agent_tech_levels[nbr_ids] * state[nbr_ids, 1]

        nbr_observations = model.rng.multivariate_normal(
                                        mean=nbr_technosignatures,
                                        cov=obs_noise_sd**2 * np.eye(len(nbr_ids)),
                                        size=n_samples)
        sample[:, nbr_ids] = nbr_observations

    if n_samples == 1:
        return sample[0]

    return sample

def sample_init(n_samples, n_agents, level, agent, rng, agent_growth, 
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

    Note that the agent always has correct beliefs about its own part of the
    environment state (although it's beliefs about others' beliefs of course
    don't have to be correct in general)

    Keyword arguments:
    n_samples: number of samples to generate
    n_agents: number of agents in the model
    level: level of interactive beliefs of the agent. 0 or 1
    agent: a Civilisation whose initial beliefs are sampled
    rng: random number generator (from the Universe object or elsewhere)
    agent_growth: growth function used
    **agent_growth_kwargs: arguments used for the growth function

    Returns:
    A NumPy array (see description above for the size)
    """
    agent_id = agent.unique_id

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

            # agent is certain about it's own state
            sample[:, agent_id, :] = agent.get_state()

    elif level == 1:

        sample[:, :, :, 0] = 0
        sample[:, :, :, 1] = 1

        if agent_growth == sigmoid_growth:
            sample[:, :, :, 2] = rng.uniform(*speed_range,
                    size=(n_samples, 1 + (n_agents - 1) * n_samples, n_agents))
            sample[:, :, :, 3] = rng.integers(*takeoff_time_range,
                    size=(n_samples, 1 + (n_agents - 1) * n_samples, n_agents))

        # agent is certain about it's own state in its own beliefs about the
        # environment
        sample[:, 0, agent_id, :] = agent.get_state()

    return sample

def update_beliefs_0(agent, belief, agent_action, agent_observation, 
                     action_dist_0, model, agent_growth, obs_noise_sd,
                     in_place=False, **kwargs):
    """
    Calculates agent's updated beliefs. This is done assuming that agent has 
    beliefs at t-1 represented by the particle set “belief”, agent takes the 
    given action and receives the given observation, and assumes a given 
    level 0 action distribution by the other agents.

    Model's random number generator (rng attribute) is used for sampling
    other's actions and for resampling.

    TODO: **kwargs is useless and is there for making this compatible with
    optimal_action

    Keyword arguments:
    agent: a Civilisation whose level 0 beliefs are in question. (In practice
           these beliefs are held by another civilisation about agent. This
           other civilisation attempts to simulate agent's belief update and
           thus update it's own beliefs about agent's beliefs.)
    belief: a sample of environment states. A NumPy array of size 
            (n_samples, n_agents, k) where k is the size of an individual 
            agent state representation. For sigmoid growth k = 4.
    agent_action: action taken by agent at t-1. Either a string ('hide' 
                  for hiding or '-' for no action) or a Civilisation that was 
                  attacked. If someone else acted, None.
    agent_observation: observation made by agent at time t following the 
                       action. A NumPy array of length n_agents or n_agents+1
                       if agent_action was an attack.
    action_dist_0: the distribution of actions that agent assumes others 
                   sampled their actions from at time t-1. "random" means the
                   others' action is chosen uniformly over the set of possible
                   choices. That is the only implemented option so far.
    model: a Universe
    agent_growth: growth function used
    obs_noise_sd: standard deviation of observation noise (which is assumed to
                  follow a normal distribution centered around the true 
                  technosignature value)
    in_place: whether intermediate operations can modify “belief” in-place.
              Note that even if this is True, the final result is only returned
              and not saved in-place.

    Returns:
    a sample of environment states representing the updated beliefs. A NumPy
    array of size (n_samples, n_agents, k)
    """
    n_samples = belief.shape[0]

    # calculate influence radii of civilisations in the different samples
    if agent_growth == sigmoid_growth:
        # this is of shape (n_samples, n_agents)
        radii = influence_radius(sigmoid_growth(time=belief[:, :, 0],
                                                speed=belief[:, :, 2],
                                                takeoff_time=belief[:, :, 3]))
    else:
        raise NotImplementedError()

    # sample others' actions, one for each sample
    if action_dist_0 == "random":

        if agent_action == None:
            # if agent didn't act, then one other civilisatin is randomly
            # chosen to act and one of their possible actions is randomly 
            # chosen
            other_agents = [ag for ag in model.schedule.agents if ag != agent]
            actors = model.rng.choice(other_agents, size=n_samples)
            
            actors_nbrs = [model.space.get_neighbors(
                    pos=actor.pos,
                    radius=radii[sample, actor.unique_id],
                    include_center=False)
                for sample, actor in enumerate(actors)]
            actor_actions = [model.rng.choice(['hide', '-'] + actor_nbrs)
                       for actor_nbrs in actors_nbrs]
        
            actions = [{'actor': actor, 'type': action}
                       for actor, action in zip(actors, actor_actions)]

        else:
            # if agent acted, then the others cannot do anything
            actions = n_samples * [{'actor': agent, 'type': agent_action}]

    else:
        raise NotImplementedError()


    # propagate all sample states forward using the sampled actions
    propagated_states = transition(state=belief, action=actions, model=model,
                                   agent_growth=agent_growth, in_place=in_place)

    # calculate weights of all propagated states, given by the observation
    # probabilities
    weights = np.array([prob_observation(model=model,
                                         agent=agent,
                                         state=p_state,
                                         action=action,
                                         observation=agent_observation,
                                         agent_growth=agent_growth,
                                         obs_noise_sd=obs_noise_sd)
                        for p_state, action in zip(propagated_states, actions)])

    # normalise weights
    weights = weights / weights.sum()

    # resample
    updated_belief = model.rng.choice(propagated_states, 
                                      size=n_samples, p=weights)

    return updated_belief

def update_beliefs_1(agent, belief, agent_action, agent_observation, 
                     action_dist_0, model, agent_growth, obs_noise_sd,
                     time_horizon, discount_factor):
    """
    Calculates agent's updated level 1 interactive beliefs. This is done 
    assuming that agent has level 1 interactive beliefs at t-1 represented by 
    the particle set “belief”, agent takes the given action and receives the 
    given observation, and assumes that the other agents assume a given level 0 
    action distribution for the other agents when the original agent updates 
    its level 0 beliefs about others' beliefs.

    NOTE: currently only a single observation of each of the other agents is
    sampled when updating the level 0 beliefs corresponding to a propagated
    environment state.

    Model's random number generator (rng attribute) is used for sampling
    other's actions and for resampling.

    Keyword arguments:
    agent: a Civilisation whose level 1 beliefs are in question
    belief: a sample of level 1 interactive states. A NumPy array of size 
            n_samples x (1 + (n_agents - 1) * n_samples) x n_agents x k
            where k is the size of an individual agent state representation. 
            For sigmoid growth k = 4. In a sample, the first state corresponds
            to an environment state, the following n_samples correspond to 
            beliefs about civilisation 0's beliefs, the next n_samples
            correspond to beliefs about civilisation 1's beliefs, and so on
            (skipping the agent's own id)
    agent_action: action taken by agent at t-1. Either a string ('hide' 
                  for hiding or '-' for no action) or a Civilisation that was 
                  attacked. If someone else acted, None.
    agent_observation: observation made by agent at time t following the 
                       action. A NumPy array of length n_agents or n_agents+1
                       if agent_action was an attack.
    action_dist_0: the distribution of actions that agent assumes others 
                   assume when agent updates its level 0 beliefs about them.
                   Passed to update_beliefs_0. "random" means the others' 
                   action is chosen uniformly over the set of possible
                   choices. That is the only implemented option so far.
    model: a Universe
    agent_growth: growth function used
    obs_noise_sd: standard deviation of observation noise (which is assumed to
                  follow a normal distribution centered around the true 
                  technosignature value)
    time_horizon: number of time steps to look ahed when determining what
                  others did
    discount_factor: how much future time steps are discounted when determining
                     the rational actions of other agents

    Returns:
    a sample of environment states representing the updated interactive 
    beliefs. A NumPy array of size 
    n_samples x (1 + (n_agents - 1) * n_samples) x n_agents x k
    """
    n_samples = belief.shape[0]

    # find other agents and represent with tuples of form 
    # (agent, array_index)
    other_agents = [(ag, ag.unique_id - (ag.unique_id > agent.unique_id))
                    for ag in model.schedule.agents
                    if ag != agent]

    # determine what others did last round, for each sample
    if agent_action == None:
        # if agent didn't act, then someone else did. Therefore we determine
        # the set of rational actions for each other civilisation and sample
        # one of these per civilisation. We do this for each sample.
        actions = [{'actor': ag,
                    'type': optimal_action(
                        belief=i_state[1 + n_samples*ind:
                                       1 + n_samples*(ind+1)],
                        agent=ag,
                        actor=ag,
                        time_horizon=time_horizon,
                        level=0,
                        discount_factor=discount_factor,
                        action_dist_0=action_dist_0,
                        model=model,
                        agent_growth=agent_growth,
                        obs_noise_sd=obs_noise_sd,
                        return_sample=True)}
                   for ag, ind in other_agents
                   for i_state in belief]

        n_actions = len(other_agents)

    else:
        # if agent acted, then the others cannot do anything
        actions = [{'actor': agent, 'type': agent_action} for i_state in belief]
        n_actions = 1

    # propagate environment states using the determined actions
    # repeat also creates a copy of belief so the original belief will not
    # be altered
    propagated_i_states = belief.repeat(repeats=n_actions, axis=0)
    transition(state=propagated_i_states[:, 0, :, :],
               action=actions,
               model=model,
               agent_growth=agent_growth,
               in_place=True)

    # calculate associated weights, which depend on how compatible the state-
    # action combination is with the given observation
    weights = np.array([prob_observation(model=model, 
                                         agent=agent, 
                                         state=p_p_i_state[0], 
                                         action=action,
                                         observation=agent_observation,
                                         agent_growth=agent_growth,
                                         obs_noise_sd=obs_noise_sd)
                        for p_p_i_state, action in zip(propagated_i_states, actions)])

    # normalise weights
    weights = weights / weights.sum()

    # resample
    samples = model.rng.choice(len(propagated_i_states), 
                               size=n_samples, p=weights)
    propagated_i_states = propagated_i_states[samples]
    actions = [actions[i] for i in samples]

    # we have to update the level 0 beliefs for each interactive state sampled
    for sample_i, (p_i_state, action) in enumerate(zip(propagated_i_states, actions)):

        for ag, ind in other_agents:
            # sample a single observations for each other agent
            observation = sample_observation(n_samples=1,
                                             model=model,
                                             agent=ag,
                                             state=p_i_state[0],
                                             action=action,
                                             agent_growth=agent_growth,
                                             obs_noise_sd=obs_noise_sd)

            ag_action = action['type'] if action['actor'] == ag else None

            # update agent's level 0 beliefs about ag
            propagated_i_states[sample_i,
                                1 + n_samples*ind:
                                1 + n_samples*(ind+1)] = (
                update_beliefs_0(agent=ag,
                                 belief=p_i_state[1 + n_samples*ind:
                                                  1 + n_samples*(ind+1)],
                                 agent_action=ag_action,
                                 agent_observation=observation,
                                 action_dist_0=action_dist_0,
                                 model=model,
                                 agent_growth=agent_growth,
                                 obs_noise_sd=obs_noise_sd,
                                 in_place=True))

    return propagated_i_states

def optimal_action(belief, agent, actor, time_horizon, level, discount_factor, 
                   action_dist_0, model, agent_growth, obs_noise_sd, 
                   return_sample=False):
    """
    Calculate the optimal action to take in belief state “belief”. Returns
    also the associated expected utility of taking that action.

    The consequences of actions are modelled time_horizon steps forward, and
    the action that has the highest expected utility (assuming agent and 
    everyone else acts optimally every time they get a turn). Utilities from 
    future timesteps are discounted by discount_factor. All possible future 
    actor sequences (of length time horizon) are tested. NOTE that this becomes 
    very prohibitive for long horizons and large numbers of actor, so this might
    have to change in the future.

    Keyword arguments:
    belief: beliefs (at level “level”) of agent. If level=1, this is a 
            NumPy array of size (n_samples, 1 + (n_agents-1) * n_samples, 
            n_agents, k) where k is the size of an individual agent's part in
            the state representation. For sigmoid_growth, k=4. If level=0, 
            this is a NumPy array of size (n_samples, n_agents, k).
    agent: the Civilisation whose utility is in question
    actor: the Civilisation that gets to move on the first round
    time_horizon: number of rounds to model forward
    level: level of beliefs. Either 0 or 1.
    discount_factor: a number between 0 and 1 indicating how much future 
                     rewards are discounted
    action_dist_0: distribution over others' actions that agent assumes
                   others assume (at level 0) when they decide how to act. 
                   "random" means a random choice over others' actions.
    model: a Universe. Used for determining neighbours and random sampling.
    agent_growth: the growth model used
    obs_noise_sd: standard deviation of observation noise
    return_sample: whether to return only one randomly chosen optimal action

    Returns:
    A list of optimal actions and the associated expected utility.
    """
    #print(f"Solving optimal actions for {agent.unique_id} when",
    #      f"{actor.unique_id} acts, horizon {time_horizon}")

    n_samples = belief.shape[0]
    n_agents = len(model.schedule.agents)
    
    assert(level in (0, 1))

    if actor == agent:
        # determine the actions available to agent. This involves finding
        # all the civilisations agent can attack

        # agent state will be correct no matter which sample we choose
        if level == 1:
            agent_state = belief[0, 0, agent.unique_id, :]
        elif level == 0:
            agent_state = belief[0, agent.unique_id, :]

        # calculate agent's influence radius
        if agent_growth == sigmoid_growth:
            radius = influence_radius(sigmoid_growth(time=agent_state[0],
                                                     speed=agent_state[2],
                                                     takeoff_time=agent_state[3]))
        else:
            raise NotImplementedError()

        # get all neighbours
        agent_nbrs = model.space.get_neighbors(pos=agent.pos,
                                               radius=radius,
                                               include_center=False)

        # create a list of actions
        agent_actions = [{'actor': agent, 'type': action}
                         for action in ['hide', '-'] + agent_nbrs]

    else:
        # agent doesn't act
        agent_actions = [None]

    # keep count of best actions found so far
    best_actions = []
    max_util = -np.inf

    # go through all actions to see which one is best
    for agent_action in agent_actions:

        # keep track of utility of this action
        action_utility_sum = 0

        # go through all samples in belief. These are interactive states if
        # level = 1 and states if level = 0
        for i_state in belief:

            # choose action to be executed
            if agent_action == None:
                # if agent is not moving this turn, then the actor chooses
                # its optimal action

                if level == 1:
                    # if we have beliefs about actor's beliefs (level=1), 
                    # we need to solve actor's optimal action
                    actor_ind = actor.unique_id - (actor.unique_id > agent.unique_id)
                    actor_belief = i_state[1 + actor_ind*n_samples: 
                                           1 + (actor_ind+1)*n_samples]

                    # if there are multiple equally good actions, pick one randomly
                    action, _ = optimal_action(belief=actor_belief,
                                            agent=actor,
                                            actor=actor,
                                            time_horizon=time_horizon,
                                            level=0,
                                            discount_factor=discount_factor,
                                            action_dist_0=action_dist_0,
                                            model=model,
                                            agent_growth=agent_growth,
                                            obs_noise_sd=obs_noise_sd,
                                            return_sample=True)

                elif level == 0:
                    # if we don't have beliefs about actor's beliefs, we simply
                    # assume that they choose their action according to
                    # action_dist_0

                    if action_dist_0 == "random":

                        actor_state = i_state[actor.unique_id]
                        if agent_growth == sigmoid_growth:
                            actor_influence_radius = influence_radius(
                                sigmoid_growth(time=actor_state[0], 
                                               speed=actor_state[2], 
                                               takeoff_time=actor_state[3]))
                        else:
                            raise NotImplementedError()

                        actor_nbrs = model.space.get_neighbors(
                                        pos=actor.pos,
                                        radius=actor_influence_radius,
                                        include_center=False)
                        
                        action = {'actor': actor, 
                                  'type': model.rng.choice(['hide', '-'] + 
                                                            actor_nbrs)}
                    
                    else:
                        raise NotImplementedError()

            else:
                action = agent_action

            if level == 1:
                state = i_state[0]
            elif level == 0:
                state = i_state

            # propagate environment state
            state = transition(state=state, action=action, model=model,
                               agent_growth=agent_growth)

            # sample an observation
            observation = sample_observation(state=state,
                                             action=action,
                                             agent=agent,
                                             n_samples=1,
                                             model=model,
                                             agent_growth=agent_growth,
                                             obs_noise_sd=obs_noise_sd)

            # calculate immediate reward
            action_utility_sum += reward(agent=agent, action=action)

            if time_horizon > 0:
                # update beliefs and compute expected utility at next time step

                updater = update_beliefs_1 if level == 1 else update_beliefs_0

                updated_belief = updater(
                    agent=agent,
                    belief=belief,
                    agent_action=action['type'] if agent == actor else None,
                    agent_observation=observation,
                    action_dist_0=action_dist_0,
                    model=model,
                    agent_growth=agent_growth,
                    obs_noise_sd=obs_noise_sd,
                    time_horizon=time_horizon,
                    discount_factor=discount_factor)

                next_utility_sum = 0

                for next_actor in model.schedule.agents:

                    _, next_utility = optimal_action(belief=updated_belief,
                                                     agent=agent,
                                                     actor=next_actor,
                                                     time_horizon=time_horizon-1,
                                                     level=level,
                                                     discount_factor=discount_factor,
                                                     action_dist_0=action_dist_0,
                                                     model=model,
                                                     agent_growth=agent_growth,
                                                     obs_noise_sd=obs_noise_sd)

                    next_utility_sum += next_utility

                action_utility_sum += discount_factor * next_utility_sum / n_agents

        action_utility = action_utility_sum / n_samples

        # update best action
        if action_utility > max_util:
            best_actions = [agent_action]
            max_util = action_utility
        elif action_utility_sum == max_util:
            best_actions.append(agent_action)

    if len(best_actions) == 1:
        best_actions = best_actions[0]
    elif len(best_actions) > 1 and return_sample:
        best_actions = model.rng.choice(best_actions)

    return best_actions, max_util
    