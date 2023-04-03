# %%
import numpy as np
from model.growth import influence_radius, sigmoid_growth
from scipy.stats import multivariate_normal

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
        return sigmoid_growth(time=state[..., 0],
                              speed=state[..., 2],
                              takeoff_time=state[..., 3])
    else:
        raise NotImplementedError()

def transition(state, action, model, in_place=False):
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
    action: a dictionary, with keys 'actor' (the id of the acting Civilisation) 
            and 'type' (either a string ('hide' for hiding or '-' for no 
            action) or the id of a Civilisation that was attacked). 
            If multiple states are supplied, this should be a list of length 
            n_samples.
    model: a Universe
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
            state[sample, act['actor'], 1] *= model.visibility_multiplier

        elif isinstance(act['type'], int):
            # if there is an attack, see if it is successful or not
            actor_state = state[sample, act['actor'], :]
            target_state = state[sample, act['type'], :]

            actor_tech_level = tech_level(state=actor_state, model=model)
            target_tech_level = tech_level(state=target_state, model=model)

            if (actor_tech_level > target_tech_level or
                (actor_tech_level == target_tech_level and 
                 model.rng.random() > 0.5)):
                # civilisation is destroyed
                state[sample, act['type'], 0] = 0 # reset time
                state[sample, act['type'], 1] = 1 # visibility factor

    # return only a single state if a single state was supplied
    if state.shape[0] == 1:
        state = state[0]

    return state

def reward(state, action, agent, model):
    """
    Given the current environment state and an action taken in that state, 
    this function calculates the reward for an agent.

    Keyword arguments:
    state: a NumPy array of size n_agents x k, where k is the size of an
           individual agent's portion of the environment state
    action: a dictionary, with keys 'actor' (the id of the acting Civilisation) 
            and 'type' (either a string ('hide' for hiding or '-' for no 
            action) or the id of a Civilisation that was attacked). 
    agent: the id of the Civilisation whose reward to calculate
    model: a Universe
    """
    actor = action['actor']

    if isinstance(action['type'], int):
        # action was an attack
        target = action['type']

        if target == agent:
            
            # calculate tech levels of actor and target
            actor_tech_level = tech_level(state=state[actor], model=model)
            target_tech_level = tech_level(state=state[target], model=model)

            if target_tech_level < actor_tech_level:
                return model.rewards['destroyed']
            elif target_tech_level == actor_tech_level:
                return model.rewards['destroyed'] / 2

        elif actor == agent:
            return model.rewards['attack']

    elif action['type'] == 'hide' and actor == agent:
        return model.rewards['hide']

    # in all other cases the reward is 0
    return 0

def prob_observation(observation, state, action, agent, model):
    """
    Returns the probability (density) of a given observation by “agent”, given 
    that the system is currently in state “state” and the previous action was 
    “action”.

    Technosignature observations from each agent are assumed to have Gaussian
    observation noise, which is saved in the model's obs_noise_sd attribute.

    Keyword arguments:
    observation: a NumPy array of length n_agents or n_agents + 1. The latter
                 corresponds to an observation where an attacker gets to know
                 the result of their attack last round (0 or 1). This binary
                 value is the last value in the array. A numpy.NaN 
                 denotes a civilisation that agent cannot observe yet, or the
                 agent itself.
    state: a NumPy array of size n_agents x k, where k is the size of an
            individual agent state representation
    action: a dictionary, with keys 'actor' (the id of the acting Civilisation) 
            and 'type' (either a string ('hide' for hiding or '-' for no 
            action) or the id of a Civilisation that was attacked). 
    agent: the id of the observing Civilisation
    model: a Universe. Used for determining distances between civilisations.
    """
    n_agents = state.shape[0]

    # calculate tech levels
    agent_tech_levels = tech_level(state=state, model=model)

    # make sure that observation only contains observations on civilisations 
    # that agent can see
    nbr_ids = [nbr.unique_id
               for nbr in model.space.get_neighbors(
                   pos=model.schedule.agents[agent].pos,
                   radius=influence_radius(agent_tech_levels[agent]),
                   include_center=False)]
    nbr_obs = observation[nbr_ids]
    unobserved_obs = np.delete((observation if observation.shape[0] == n_agents
                                            else observation[:-1]), 
                               nbr_ids)

    if np.isnan(nbr_obs).any() or not np.isnan(unobserved_obs).all():
        #print("nans are wrong", np.isnan(nbr_obs).any(), not np.isnan(unobserved_obs).all(), nbr_ids, observation)
        return 0

    # check that if the agent attacked last round, the observation contains a 
    # bit indicating the success of the attack
    if (action['actor'] == agent and
        isinstance(action['type'], int) and
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
                                      cov=model.obs_noise_sd**2)

    return density

def sample_observation(state, action, agent, model, n_samples):
    """
    Returns n_samples of possible observations of “agent” when the system is
    currently in state “state” and the previous action was “action”.

    Model's random number generator (rng attribute) is used for sampling.

    Technosignature observations from each agent are assumed to have Gaussian
    observation noise, which is saved in the model's obs_noise_sd attribute.

    Keyword arguments:
    state: a NumPy array of size n_agents x k, where k is the size of an
           individual agent state representation
    action: a dictionary, with keys 'actor' (the id of the acting Civilisation) 
            and 'type' (either a string ('hide' for hiding or '-' for no 
            action) or the id of a Civilisation that was attacked). 
    agent: the id of the observing Civilisation
    model: a Universe. Used for determining distances between civilisations and
           for random sampling.
    n_samples: number of observation samples to generate

    Returns:
    The observations. A NumPy array of size  n_samples x 
    (n_agents or n_agents + 1). The latter corresponds to an observation 
    where an attacker gets to know the result of their attack last round 
    (0 or 1). This binary value is the last value in the array. A numpy.NaN 
    denotes a civilisation that agent cannot observe yet, or the agent itself.
    If n_samples == 1, the NumPy array is squeezed into a 1d array.
    """
    assert(len(state.shape) == 2)

    n_agents = state.shape[0]
    obs_length = n_agents

    # if agent has attacked another last round, we need to include a result
    # bit in each observation
    if action['actor'] == agent and isinstance(action['type'], int):
        obs_length = n_agents + 1

        # determine if target was destroyed
        target_id = action['type']
        target_destroyed = int(state[target_id, 0] == 0)

    # initialise array
    sample = np.full(shape=(n_samples, obs_length), fill_value=np.nan)

    # add success bit if necessary
    if obs_length == n_agents + 1:
        sample[:, -1] = target_destroyed

    # calculate tech levels
    agent_tech_levels = tech_level(state=state, model=model)

    # add observations from the civilisations the agent can see
    nbr_ids = [nbr.unique_id
               for nbr in model.space.get_neighbors(
                   pos=model.schedule.agents[agent].pos,
                   radius=influence_radius(agent_tech_levels[agent]),
                   include_center=False)]

    if len(nbr_ids) > 0:
        nbr_technosignatures = agent_tech_levels[nbr_ids] * state[nbr_ids, 1]

        nbr_observations = model.rng.multivariate_normal(
                            mean=nbr_technosignatures,
                            cov=model.obs_noise_sd**2 * np.eye(len(nbr_ids)),
                            size=n_samples)
        sample[:, nbr_ids] = nbr_observations

    if n_samples == 1:
        return sample[0]

    return sample

def sample_init(n_samples, level, agent, agent_state, model):
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
    level: level of interactive beliefs of the agent. 0 or 1
    agent: the id of the Civilisation whose initial beliefs are sampled
    agent_state: the initial state of the agent (a NumPy array of length k)
    model: a Universe

    Returns:
    A NumPy array (see description above for the size)
    """
    n_agents = model.n_agents

    # determine the number of values needed to describe an agent
    if model.agent_growth == sigmoid_growth:
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
    if model.agent_growth == sigmoid_growth:

        if ("speed" in model.agent_growth_params and 
            "takeoff_time" in model.agent_growth_params):
            # every agent has the same growth parameters
            speed_range = (model.agent_growth_params["speed"], 
                           model.agent_growth_params["speed"])
            takeoff_time_range = (model.agent_growth_params["takeoff_time"],
                                  model.agent_growth_params["takeoff_time"])
        elif ("speed_range" in model.agent_growth_params and 
              "takeoff_time_range" in model.agent_growth_params):
            # growth parameters are sampled from the given ranges
            speed_range = model.agent_growth_params["speed_range"]
            takeoff_time_range = model.agent_growth_params["takeoff_time_range"]
        else:
            raise Exception("Sigmoid growth parameters are incorrect")

    if level == 0:

        # in every sample and every agent, initial time is 0
        sample[:, :, 0] = 0
        # likewise, initially everyone has a visibility factor of 1
        sample[:, :, 1] = 1

        if model.agent_growth == sigmoid_growth:
            sample[:, :, 2] = model.rng.uniform(*speed_range, 
                                                size=(n_samples, n_agents))
            sample[:, :, 3] = model.rng.integers(*takeoff_time_range, 
                                                 size=(n_samples, n_agents))

            # agent is certain about it's own state
            sample[:, agent, :] = agent_state

    elif level == 1:

        sample[:, :, :, 0] = 0
        sample[:, :, :, 1] = 1

        if model.agent_growth == sigmoid_growth:
            sample[:, :, :, 2] = model.rng.uniform(*speed_range,
                size=(n_samples, 1 + (n_agents - 1) * n_samples, n_agents))
            sample[:, :, :, 3] = model.rng.integers(*takeoff_time_range,
                size=(n_samples, 1 + (n_agents - 1) * n_samples, n_agents))

        # agent is certain about it's own state in its own beliefs about the
        # environment
        sample[:, 0, agent, :] = agent_state

    return sample

def update_beliefs_0(belief, agent_action, agent_observation, agent,
                     model, in_place=False):
    """
    Calculates agent's updated beliefs. This is done assuming that agent has 
    beliefs at t-1 represented by the particle set “belief”, agent takes the 
    given action and receives the given observation, and assumes a given 
    level 0 action distribution by the other agents (the model's action_dist_0
    attribute).

    Model's random number generator (rng attribute) is used for sampling
    other's actions and for resampling.

    Keyword arguments:
    belief: a sample of environment states. A NumPy array of size 
            (n_samples, n_agents, k) where k is the size of an individual 
            agent state representation. For sigmoid growth k = 4.
    agent_action: action taken by agent at t-1. Either a string ('hide' 
                  for hiding or '-' for no action) or the id of a Civilisation 
                  that was attacked. If someone else acted, None.
    agent_observation: observation made by agent at time t following the 
                       action. A NumPy array of length n_agents or n_agents+1
                       if agent_action was an attack.
    agent: the id of the Civilisation whose level 0 beliefs are in question. 
           (In practice these beliefs are held by another civilisation about 
           agent. This other civilisation attempts to simulate agent's belief
           update and thus update it's own beliefs about agent's beliefs.)
    model: a Universe
    in_place: whether intermediate operations can modify “belief” in-place.
              Note that even if this is True, the final result is only returned
              and not saved in-place.

    Returns:
    a sample of environment states representing the updated beliefs. A NumPy
    array of size (n_samples, n_agents, k)
    """
    n_samples = belief.shape[0]

    # calculate influence radii of civilisations in the different samples
    # this is of shape (n_samples, n_agents)
    radii = influence_radius(tech_level(state=belief, model=model))

    # sample others' actions, one for each sample
    if model.action_dist_0 == "random":

        if agent_action == None:
            # if agent didn't act, then one other civilisation is randomly
            # chosen to act and one of their possible actions is randomly 
            # chosen
            other_agents = [ag for ag in model.schedule.agents
                            if ag.unique_id != agent]
            actors = model.rng.choice(other_agents, size=n_samples)
            
            # list of lists
            actors_nbrs = [model.space.get_neighbors(
                    pos=actor.pos,
                    radius=radii[sample, actor.unique_id],
                    include_center=False)
                for sample, actor in enumerate(actors)]
            actor_actions = [model.rng.choice(['hide', '-'] +
                                              [nbr.unique_id for nbr in actor_nbrs])
                             for actor_nbrs in actors_nbrs]
        
            actions = [{'actor': actor.unique_id, 'type': action}
                       for actor, action in zip(actors, actor_actions)]

        else:
            # if agent acted, then the others cannot do anything
            actions = n_samples * [{'actor': agent, 'type': agent_action}]

    else:
        raise NotImplementedError()


    # propagate all sample states forward using the sampled actions
    propagated_states = transition(state=belief, action=actions, model=model,
                                   in_place=in_place)
 
    # calculate weights of all propagated states, given by the observation
    # probabilities
    weights = np.array([prob_observation(model=model,
                                         agent=agent,
                                         state=p_state,
                                         action=action,
                                         observation=agent_observation)
                        for p_state, action in zip(propagated_states, actions)])

    if weights.sum() == 0:
        # TODO: this is a problem. This means that none of the samples in the
        # level 0 beliefs capture the situation where the given observation
        # is possible. For example, the observation contains a technosignature
        # observation from a civilisation which none of the samples think is
        # possible for agent to observe.
        # Possible solution could be to randomly sample new belief states. But
        # how to ensure they have positive probability?
        raise Exception("No beliefs are compatible with this observation!")

    # normalise weights
    weights = weights / weights.sum()

    # resample
    updated_belief = model.rng.choice(propagated_states, 
                                      size=n_samples, p=weights)

    return updated_belief

def update_beliefs_1(belief, agent_action, agent_observation, agent, model):
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
    belief: a sample of level 1 interactive states. A NumPy array of size 
            n_samples x (1 + (n_agents - 1) * n_samples) x n_agents x k
            where k is the size of an individual agent state representation. 
            For sigmoid growth k = 4. In a sample, the first state corresponds
            to an environment state, the following n_samples correspond to 
            beliefs about civilisation 0's beliefs, the next n_samples
            correspond to beliefs about civilisation 1's beliefs, and so on
            (skipping the agent's own id)
    agent_action: action taken by agent at t-1. Either a string ('hide' 
                  for hiding or '-' for no action) or the id of a Civilisation 
                  that was attacked. If someone else acted, None.
    agent_observation: observation made by agent at time t following the 
                       action. A NumPy array of length n_agents or n_agents+1
                       if agent_action was an attack.
    agent: the id of the Civilisation whose level 1 beliefs are in question
    model: a Universe

    Returns:
    a sample of environment states representing the updated interactive 
    beliefs. A NumPy array of size 
    n_samples x (1 + (n_agents - 1) * n_samples) x n_agents x k
    """
    n_samples = belief.shape[0]

    # find other agents and represent with tuples of form 
    # (agent_id, array_index)
    other_agents = [(ag.unique_id, ag.unique_id - (ag.unique_id > agent))
                    for ag in model.schedule.agents
                    if ag.unique_id != agent]

    # determine what others did last round, for each sample
    if agent_action == None:
        # if agent didn't act, then someone else did. Therefore we determine
        # the set of rational actions for each other civilisation and sample
        # one of these per civilisation. We do this for each sample. So in
        # total actions will contain n_samples * (n_agents-1) actions
        actions = [optimal_action(
                        belief=i_state[1 + n_samples*ind:
                                       1 + n_samples*(ind+1)],
                        agent=ag_id,
                        actor=ag_id,
                        time_horizon=model.belief_update_time_horizon,
                        level=0,
                        model=model,
                        return_sample=True)[0]
                   for ag_id, ind in other_agents
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
               in_place=True)

    # calculate associated weights, which depend on how compatible the state-
    # action combination is with the given observation
    weights = np.array([prob_observation(model=model, 
                                         agent=agent, 
                                         state=p_p_i_state[0], 
                                         action=action,
                                         observation=agent_observation)
                        for p_p_i_state, action in zip(propagated_i_states, actions)])

    if weights.sum() == 0:
        # TODO: this is a problem. This means that none of the samples in the
        # level 1 beliefs capture the situation where the given observation
        # is possible. For example, the observation contains a technosignature
        # observation from a civilisation which none of the samples think is
        # possible for agent to observe.
        # Possible solution could be to randomly sample new belief states. But
        # how to ensure they have positive probability?
        raise Exception("No beliefs are compatible with this observation!")

    # normalise weights
    weights = weights / weights.sum()

    # resample
    samples = model.rng.choice(len(propagated_i_states), 
                               size=n_samples, p=weights)
    propagated_i_states = propagated_i_states[samples]
    actions = [actions[i] for i in samples]

    # we have to update the level 0 beliefs for each interactive state sampled
    for sample_i, (p_i_state, action) in enumerate(zip(propagated_i_states, actions)):

        for ag_id, ind in other_agents:
            # sample a single observations for each other agent
            observation = sample_observation(n_samples=1,
                                             model=model,
                                             agent=ag_id,
                                             state=p_i_state[0],
                                             action=action)

            ag_action = action['type'] if action['actor'] == ag_id else None

            # update agent's level 0 beliefs about ag
            propagated_i_states[sample_i,
                                1 + n_samples*ind:
                                1 + n_samples*(ind+1)] = (
                update_beliefs_0(agent=ag_id,
                                 belief=p_i_state[1 + n_samples*ind:
                                                  1 + n_samples*(ind+1)],
                                 agent_action=ag_action,
                                 agent_observation=observation,
                                 model=model,
                                 in_place=True))

    return propagated_i_states

def optimal_action(belief, agent, actor, time_horizon, level, model, 
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
    agent: the id of the Civilisation whose utility is in question
    actor: the id of the Civilisation that gets to move on the first round
    time_horizon: number of rounds to model forward
    level: level of beliefs. Either 0 or 1.
    model: a Universe. Used for determining neighbours and random sampling.
    return_sample: whether to return only one randomly chosen optimal action

    Returns:
    A list of optimal actions and the associated expected utility.
    """
    #print(f"Solving optimal actions for {agent.unique_id} when",
    #      f"{actor.unique_id} acts, horizon {time_horizon}")

    n_samples = belief.shape[0]
    n_agents = model.n_agents
    
    assert(level in (0, 1))

    if actor == agent:
        # determine the actions available to agent. This involves finding
        # all the civilisations agent can attack

        # agent state will be correct no matter which sample we choose
        if level == 1:
            agent_state = belief[0, 0, agent, :]
        elif level == 0:
            agent_state = belief[0, agent, :]

        # calculate agent's influence radius
        radius = influence_radius(tech_level(state=agent_state, model=model))

        # get all neighbours
        agent_nbrs = model.space.get_neighbors(pos=model.schedule.agents[agent].pos,
                                               radius=radius,
                                               include_center=False)

        # create a list of actions
        agent_actions = [{'actor': agent, 'type': action}
                         for action in ['hide', '-'] + 
                                       [nbr.unique_id for nbr in agent_nbrs]]

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
                    actor_ind = actor - (actor > agent)
                    actor_belief = i_state[1 + actor_ind*n_samples: 
                                           1 + (actor_ind+1)*n_samples]

                    # if there are multiple equally good actions, pick one randomly
                    action, _ = optimal_action(belief=actor_belief,
                                               agent=actor,
                                               actor=actor,
                                               time_horizon=time_horizon,
                                               level=0,
                                               model=model,
                                               return_sample=True)

                elif level == 0:
                    # if we don't have beliefs about actor's beliefs, we simply
                    # assume that they choose their action according to
                    # action_dist_0

                    if model.action_dist_0 == "random":

                        actor_state = i_state[actor]
                        actor_influence_radius = influence_radius(
                            tech_level(state=actor_state, model=model))

                        actor_nbrs = model.space.get_neighbors(
                                        pos=model.schedule.agents[actor].pos,
                                        radius=actor_influence_radius,
                                        include_center=False)
                        
                        action = {'actor': actor, 
                                  'type': model.rng.choice(['hide', '-'] + 
                                        [nbr.unique_id for nbr in actor_nbrs])}
                    
                    else:
                        raise NotImplementedError()

            else:
                action = agent_action

            state = i_state if level == 0 else i_state[0]

            # calculate immediate reward
            action_utility_sum += reward(state=state, action=action, 
                                         agent=agent, model=model)

            # propagate environment state
            state = transition(state=state, action=action, model=model)

            # sample an observation
            observation = sample_observation(state=state,
                                             action=action,
                                             agent=agent,
                                             model=model,
                                             n_samples=1)

            if time_horizon > 0:
                # update beliefs and compute expected utility at next time step

                updater = update_beliefs_1 if level == 1 else update_beliefs_0

                updated_belief = updater(
                    belief=belief,
                    agent_action=action['type'] if agent == actor else None,
                    agent_observation=observation,
                    agent=agent,
                    model=model)

                next_utility_sum = 0

                for next_actor in model.schedule.agents:

                    _, next_utility = optimal_action(belief=updated_belief,
                                                     agent=agent,
                                                     actor=next_actor.unique_id,
                                                     time_horizon=time_horizon-1,
                                                     level=level,
                                                     model=model)

                    next_utility_sum += next_utility

                action_utility_sum += (model.discount_factor * 
                                       next_utility_sum / n_agents)

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
    