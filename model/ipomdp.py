# %%

# avoids having to give type annotations as strings
from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    # avoid circular imports with type hints
    from model import universe, ipomdp_solver

import numpy as np
from model import growth, action, civilisation, ipomdp_solver

def _norm_pdf(x, mean, sd):
    """
    Calculate the pdf of the (univariate) normal distribution with given
    parameters.

    This is roughly 10x faster than using scipy.stats.norm.pdf

    Keyword arguments:
    x: where to calculate the density, can be a NumPy array
    mean: mean of the distribution
    sd: standard deviation of the distribution
    """
    return (1/(sd*np.sqrt(2*np.pi)))*np.exp(-(1/2)*((x-mean)/sd)**2)

def transition(state: ipomdp_solver.State, 
               action_: ipomdp_solver.Action, 
               model: universe.Universe, 
               in_place=False) -> ipomdp_solver.State:
    """
    Given a model state and an action, this function samples from the
    distribution of all possible future states. In practice a (state, action) 
    combination only has one possible future state, except in the case of
    equally matched target and attacker, in which case the target is
    destoryed with a 0.5 probability (although note that that can only happen
    if tech levels are discrete, which they currently are not). This can be 
    used as the transition function for all agents, as they all have the same 
    transition model.

    The random number generator of the model (the rng attribute) is used for
    random choices.
 
    Keyword arguments:
    state: representation of the system at time t-1. a NumPy array of size 
           (n_agents, k), where k is the length of an individual agent state
           representation (k=4 for sigmoid growth).
    action_: the action performed
    model: a Universe
    in_place: boolean, whether to change the state object directly

    Returns: 
    A possible system state at time t. A NumPy array of the same shape as 
    the state argument.
    """
    # copy state so we don't change original
    if not in_place:
        state = state.copy()

    # multiple actors are not yet supported
    if len(action_.items()) > 1:
        raise NotImplementedError("Multiple actors are not yet supported")
    actor, actor_action = next(iter(action_.items()))

    # always tick everyone's time by one
    state[:, 0] += 1

    if actor_action == action.HIDE:
        # if actor hides, additionally update their visibility
        state[actor.id, 1] *= model.visibility_multiplier

    elif isinstance(actor_action, civilisation.Civilisation):
        # if there is an attack, see if it is successful or not
        target = actor_action
        actor_state = state[actor.id, :]
        target_state = state[target.id, :]

        actor_tech_level = growth.tech_level(state=actor_state, model=model)
        target_tech_level = growth.tech_level(state=target_state, model=model)

        if (actor_tech_level > target_tech_level or
            (actor_tech_level == target_tech_level and 
                model.rng.random() > 0.5)):
            # civilisation is destroyed
            state[target.id, 0] = 0 # reset time
            state[target.id, 1] = 1 # visibility factor

    elif actor_action != action.NO_ACTION:
        print(action_, actor, actor_action)
        raise Exception("Incorrect action format")

    return state

def reward(state: ipomdp_solver.State, 
           action_: ipomdp_solver.Action, 
           agent: civilisation.Civilisation, 
           model: universe.Universe) -> float:
    """
    Given the current environment state and an action taken in that state, 
    this function calculates the reward for an agent.

    Keyword arguments:
    state: the current environment state
    action: the action taken in the state
    agent: the Civilisation whose reward we want to calculate
    model: a Universe
    """
    assert(len(action_) == 1)
    actor, actor_action = next(iter(action_.items()))

    if isinstance(actor_action, civilisation.Civilisation):
        # action was an attack
        target = actor_action

        if target == agent:
            
            # calculate tech levels of actor and target
            actor_tech_level = growth.tech_level(state=state[actor.id], 
                                                 model=model)
            target_tech_level = growth.tech_level(state=state[target.id], 
                                                  model=model)

            if target_tech_level < actor_tech_level:
                return model.rewards['destroyed']
            elif target_tech_level == actor_tech_level:
                return model.rewards['destroyed'] / 2

        elif actor == agent:
            return model.rewards['attack']

    elif actor_action == action.HIDE and actor == agent:
        return model.rewards['hide']

    # in all other cases the reward is 0
    return 0

def prob_observation(observation: ipomdp_solver.Observation, 
                     state: ipomdp_solver.State, 
                     action: ipomdp_solver.Action, 
                     agent: civilisation.Civilisation, 
                     model: universe.Universe) -> float:
    """
    Returns the probability (density) of a given observation by “agent”, given 
    that the system is currently in state “state” and the previous action was 
    “action”.

    Technosignature observations from each agent are assumed to have Gaussian
    observation noise, which is saved in the model's obs_noise_sd attribute.

    Keyword arguments:
    observation: the observation received
    state: the current (assumed) system state
    action: previous action
    agent: the observing Civilisation
    model: a Universe. Used for determining distances between civilisations.
    """
    n_agents = model.n_agents

    if model.agent_growth == growth.sigmoid_growth:
        k = 4
    else:
        raise NotImplementedError()

    # check that if the agent attacked last round, the observation contains a 
    # bit indicating the success of the attack
    # TODO: later this check can probably be removed
    if (agent in action and
        isinstance(action[agent], civilisation.Civilisation) and
        (observation[-1] not in (0, 1))):
        raise Exception("Erroneous observation")

    # check that the observation of the agent's own state matches the model state
    #if not (observation[n_agents : n_agents + k] == state[agent.id]).all():
    #    return 1 # TODO

    # calculate tech levels
    agent_tech_levels = growth.tech_level(state=state, model=model)

    # make sure that observation only contains observations on civilisations 
    # that agent can see
    # TODO: this can be optimised by keeping an array of distances in the
    # model object.
    nbr_ids = [nbr.id
               for nbr in model.space.get_neighbors(
                   pos=agent.pos,
                   radius=growth.influence_radius(agent_tech_levels[agent.id]),
                   include_center=False)]

    # if there are no neighbours, then there is only one possible observation
    if len(nbr_ids) == 0:
        return 1

    # check that agent got observations only from those it can reach
    for ag_id in range(n_agents):

        if ag_id == agent.id:
            continue

        is_neighbour = ag_id in nbr_ids

        if is_neighbour and np.isnan(observation[ag_id]):
            return 0

        if not is_neighbour and not np.isnan(observation[ag_id]):
            return 0


    # technosignature is a product of visibility factor and tech level
    nbr_technosignatures = agent_tech_levels[nbr_ids] * state[nbr_ids, 1]

    # return density of multivariate normal. Observations from neighbours are
    # independent, centered around their respective technosignatures and
    # have a variance of obs_noise_sd^2
    density = np.prod(_norm_pdf(x=observation[nbr_ids], 
                                mean=nbr_technosignatures, 
                                sd=model.obs_noise_sd))

    return density

def sample_observation(state: ipomdp_solver.State, 
                       action: ipomdp_solver.Action,
                       agent: civilisation.Civilisation, 
                       model: universe.Universe, 
                       n_samples: int) -> ipomdp_solver.Observation:
    """
    Returns n_samples of possible observations of “agent” when the system is
    currently in state “state” and the previous action was “action”. 
    Observations include technosignatures from all civilisations (n_agents
    values, where agent's own value is np.nan), the state of agent in “state”
    (k values) and, if the agent attacked someone last round, a bit indicating
    success.

    Model's random number generator (rng attribute) is used for sampling.

    Technosignature observations from each agent are assumed to have Gaussian
    observation noise, which is saved in the model's obs_noise_sd attribute.

    Keyword arguments:
    state: the current model state
    action: the previous action
    agent: the observing Civilisation
    model: a Universe. Used for determining distances between civilisations and
           for random sampling.
    n_samples: number of observation samples to generate

    Returns:
    The observations. A NumPy array of size n_samples x (n_agents + k + 1).
    The n_agents values correspond to technosignatures of civilisations,
    k corresponds to the agents own state (which it always knows), and the
    final bit indicates whether a possible attack by the agent was successful. 
    A numpy.NaN denotes a civilisation that agent cannot observe yet or the 
    agent itself. The last attack bit it also np.nan if the agent did not 
    attack.
    If n_samples == 1, the NumPy array is squeezed into a 1d array.
    """
    assert(len(state.shape) == 2)

    if model.agent_growth == growth.sigmoid_growth:
        k = 4
    else:
        raise NotImplementedError()

    n_agents = model.n_agents

    # initialise array
    sample = np.full(shape=(n_samples, n_agents + k + 1), fill_value=np.nan)

    # add success bit if necessary
    agent_attacked = ((agent in action) and 
                      (isinstance(action[agent], civilisation.Civilisation)))
    if agent_attacked:
        # determine if target was destroyed
        target_id = action[agent].id
        target_destroyed = int(state[target_id, 0] == 0)
        sample[:, -1] = target_destroyed

    # add agent's own state
    sample[:, n_agents : n_agents + k] = state[agent.id]

    # calculate tech levels
    agent_tech_levels = growth.tech_level(state=state, model=model)

    # add observations from the civilisations the agent can see
    nbr_ids = [nbr.id
               for nbr in model.space.get_neighbors(
                   pos=agent.pos,
                   radius=growth.influence_radius(agent_tech_levels[agent.id]),
                   include_center=False)]

    if len(nbr_ids) > 0:
        nbr_technosignatures = agent_tech_levels[nbr_ids] * state[nbr_ids, 1]
        noise = model.rng.normal(loc=0, scale=model.obs_noise_sd, 
                                 size=len(nbr_ids))
        sample[:, nbr_ids] = nbr_technosignatures + noise

    if n_samples == 1:
        return sample[0]

    return sample


def sample_init(n_samples: int, 
                model: universe.Universe, 
                agent : civilisation.Civilisation = None
                ) -> ipomdp_solver.Belief:
    """
    Generates n_samples samples of the initial belief. These samples are used
    to represent the initial belief distribution of agents.

    Keyword arguments:
    n_samples: the number of samples to return
    model: used to access the agent growth function (which determines the shape 
           of states) and other parameters 
    agent: if provided, the state of the agent in the states is replaced with
           the true state of the agent. This is used at the highest level tree, 
           where there is certainty about the true state of the agent.

    Returns:
    a NumPy array of shape (n_samples, n_agents, k), where k is the size of 
    an individual agent state representation.
    """
    n_agents = model.n_agents

    # determine the number of values needed to describe an agent
    if model.agent_growth == growth.sigmoid_growth:
        k = 4
    else:
        raise NotImplementedError()

    # initialise array of samples
    sample = np.zeros((n_samples, n_agents, k))
    size = (n_samples, n_agents)

    # initial age distribution
    sample[..., 0] = model.rng.integers(*model.init_age_belief_range,
                                        size=size, endpoint=True)

    # initial visibility distribution
    sample[..., 1] = model.rng.uniform(*model.init_visibility_belief_range,
                                       size=size)

    # determine the values or range of the growth parameters
    if model.agent_growth == growth.sigmoid_growth:

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

        # sample from the ranges
        sample[..., 2] = model.rng.uniform(*speed_range, size=size)
        sample[..., 3] = model.rng.integers(*takeoff_time_range, size=size, 
                                            endpoint=True)

    # if provided, agent is certain about its own state in all samples
    if agent is not None:
        sample[:, agent.id, :] = agent.get_state()


    return sample

def rollout(state: ipomdp_solver.State, 
            agent: civilisation.Civilisation,
            model: universe.Universe,
            depth: int = 0
            ) -> Tuple[float, ipomdp_solver.AgentAction]:
        """
        Starting from the given model state, use random actions to propagate
        the state forward until the discount horizon is reached. Returns
        the value and the first action taken by agent in the rollout process.

        NOTE: state is mutated in place for efficiency.

        Keyword arguments:
        state: the state to start the rollout in
        agent: the agent whose reward is of interest
        model: a Universe.
        depth: the number of steps we have rolled out so far
        """

        # if we have reached the discount horizon, stop the recursion
        if model.discount_factor ** depth < model.discount_epsilon:
            return 0, action.NO_ACTION

        # choose actor
        actor = model.rng.choice(model.agents)

        # choose action
        possible_actions = list(ipomdp_solver.possible_actions(model=model, 
                                                               agent=agent))
        actor_action = model.rng.choice(possible_actions)
        action_ = {actor: actor_action}

        # calculate value of taking action in state
        value = reward(state=state, action_=action_, agent=agent, model=model)
        
        # propagate state
        next_state = transition(state=state, action_=action_, model=model, 
                                in_place=True)
        
        # continue rollout from the propagated state
        next_value, _ = rollout(state=next_state, agent=agent, model=model, 
                                depth=depth+1)
        
        agent_action = actor_action if agent == actor else action.NO_ACTION
        return value + model.discount_factor * next_value, agent_action

def level0_opponent_policy(agent: civilisation.Civilisation,
                           model: universe.Universe
                           ) -> ipomdp_solver.AgentAction:
    """
    Choose an agent action for agent according to the level 0 default policy.
    """
    if model.action_dist_0 == "random":
        possible_actions = tuple(ipomdp_solver.possible_actions(model=model,
                                                                agent=agent))
        return model.rng.choice(possible_actions)

    raise NotImplementedError()
