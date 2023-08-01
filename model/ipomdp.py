# %%

# avoids having to give type annotations as strings
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # avoid circular imports with type hints
    from model import universe
    from numpy.typing import NDArray

import numpy as np
from numba import njit
from model import growth, action, civilisation, ipomdp_solver


@njit
def _norm_pdf(x: NDArray, mean: NDArray, sd: float):
    """
    Calculate the pdf of the (univariate) normal distribution with given
    parameters.

    This is roughly 10x faster than using scipy.stats.norm.pdf

    Keyword arguments:
    x: where to calculate the density, can be a NumPy array
    mean: mean of the distribution
    sd: standard deviation of the distribution
    """
    return (1 / (sd * np.sqrt(2 * np.pi))) * np.exp(-(1 / 2) * ((x - mean) / sd) ** 2)


def transition(
    state: ipomdp_solver.State,
    action_: ipomdp_solver.Action,
    model: universe.Universe,
    in_place=False,
) -> ipomdp_solver.State:
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

    actor, actor_action = action_

    if actor_action == action.HIDE:
        # if actor hides, additionally update their visibility
        state[actor.id, 1] *= model.visibility_multiplier

    elif isinstance(actor_action, civilisation.Civilisation):
        # if there is an attack, see if it is successful or not
        target = actor_action

        # calculate tech levels of actor and target
        tech_levels = growth.tech_level(state=state, model=model)

        # determine if actor is capable of attacking target
        actor_capable = model.is_neighbour(
            agent1=actor, agent2=target, tech_level=tech_levels[actor.id]
        )

        if actor_capable and tech_levels[actor.id] > tech_levels[target.id]:  # or (
            # actor_capable
            # and tech_levels[actor.id] == tech_levels[target.id]
            # and model.rng.random() > 0.5
            # ):
            # civilisation is destroyed
            state[target.id, 0] = -1  # reset time
            state[target.id, 1] = 1  # visibility factor

    elif actor_action != action.NO_ACTION:
        print(action_, actor, actor_action)
        raise Exception("Incorrect action format")

    # always tick everyone's time by one
    state[:, 0] += 1

    return state


def reward(
    state: ipomdp_solver.State,
    action_: ipomdp_solver.Action,
    agent: civilisation.Civilisation,
    model: universe.Universe,
) -> float:
    """
    Given the current environment state and an action taken in that state,
    this function calculates the reward for an agent.

    Keyword arguments:
    state: the current environment state
    action: the action taken in the state
    agent: the Civilisation whose reward we want to calculate
    model: a Universe
    """
    actor, actor_action = action_

    # if agent hides, return cost
    if actor_action == action.HIDE and actor == agent:
        return model.rewards["hide"]

    # if someone else hides or the action (by anyone) is no action, reward for
    # agent is 0
    if actor_action == action.HIDE or actor_action == action.NO_ACTION:
        return 0

    # now we know that the action is an attack
    assert isinstance(actor_action, civilisation.Civilisation)

    target = actor_action

    # if agent is not the target nor the actor, it is not interested
    if agent != target and agent != actor:
        return 0

    # calculate tech levels of actor and target
    tech_levels = growth.tech_level(state=state, model=model)

    # determine if actor is capable of attacking target
    actor_capable = model.is_neighbour(
        agent1=actor, agent2=target, tech_level=tech_levels[actor.id]
    )

    # if the actor is not capable of attacking, it's as if the attack never
    # takes place
    if not actor_capable:
        return 0

    # if agent attacks, return the cost
    if actor == agent:
        return model.rewards["attack"]

    # now we know that agent was attacked

    if tech_levels[target.id] < tech_levels[actor.id]:
        return model.rewards["destroyed"]
    elif tech_levels[target.id] == tech_levels[actor.id]:
        return model.rewards["destroyed"] / 2

    # the tech level of the actor was higher than ours, we are not destroyed
    return 0


def prob_observation(
    observation: ipomdp_solver.Observation,
    states: NDArray,  # shape (n_states, n_agents, k)
    prev_action_actor_ids: NDArray,  # shape (n_states,)
    prev_action_target_ids: NDArray,  # shape (n_states,)
    observer: civilisation.Civilisation,
    model: universe.Universe,
):
    """
    This is a wrapper for the Numba-accelerated function _prob_observation. See its
    documentation for more information.
    """
    # calculate current and previous tech levels
    # these are of shape (n_states, n_agents)
    tech_levels = growth.tech_level(state=states, model=model)
    states[..., 0] -= 1
    prev_tech_levels = growth.tech_level(state=states, model=model)
    states[..., 0] += 1

    # convert observation into an array
    observation = np.array(observation, dtype=np.float_)

    # assert model.obs_noise_sd == model.obs_self_noise_sd

    return _prob_observation(
        observation=observation,
        states=states,
        observer_id=observer.id,
        tech_levels=tech_levels,
        prev_tech_levels=prev_tech_levels,
        actor_ids=prev_action_actor_ids,
        attack_target_ids=prev_action_target_ids,
        distances_tech_level=model._distances_tech_level,
        n_agents=model.n_agents,
        obs_noise_sd=model.obs_noise_sd,
        obs_self_noise_sd=model.obs_self_noise_sd,
    )


@njit
def _prob_observation(
    observation: NDArray,  # shape (n_agents + 2,)
    states: NDArray,  # shape (n_particles, n_agents, k)
    observer_id: int,
    tech_levels: NDArray,  # shape (n_particles, n_agents)
    prev_tech_levels: NDArray,  # shape (n_particles, n_agents)
    actor_ids: NDArray,  # shape (n_particles,)
    attack_target_ids: NDArray,  # shape (n_particles,)
    distances_tech_level: NDArray,  # shape (n_agents, n_agents)
    n_agents: int,
    obs_noise_sd: float,
    obs_self_noise_sd: float,
) -> float:
    """
    Returns the probability density of a given observation by an observer \
    (represented by its id, "observer_id") given a set of current system states \
    ("states") and previous actions encoded by "actor_ids" and "attack_target_ids".

    Technosignature observations from each agent are assumed to have Gaussian \
    observation noise, which has standard deviation obs_noise_sd for observations of \
    other agents and obs_self_noise_sd for the agent's observation of itself.

    NOTE: While the code is not vectorised, this does not matter in practice as \
    Numba works pretty much equally well with loops.

    Keyword arguments:
    observation: the observation received, as a NumPy array
    states: the system states to evaluate the density at
    observer_id: the id of the observing agent
    tech_levels: the technology levels of all agents in the states supplied
    prev_tech_levels: the technology levels of all agents at the previous time steps, \
                      calculated using the states
    actor_ids: an array of the same length as states, each storing the corresponding \
               actor of the previous action performed
    attack_target_ids: an array of the same length as states, each storing the \
                       corresponding target of a possible previous attack action. If \
                       the previous action was not an attack, the corresponding value \
                       is np.nan.
    distances_tech_level: an array storing the distances between civilisations, \
                          expressed in terms of technology level. This is obtained \
                          from the Universe object.
    n_agents: the number of agents in the model
    obs_noise_sd: the standard deviation of Gaussian noise in technosignature \
                  observations
    obs_self_noise_sd: the standard deviation of Gaussian noise in the observation of \
                       the observer's own technology level
    """
    n_states = len(states)

    # initialise result
    probabilities = np.ones(shape=n_states)

    ### 1. determine if the success bits in the observation are correct

    # array of length n_states
    observer_attacked = (actor_ids == observer_id) & (attack_target_ids != -1)
    obs_observer_attacked = not np.isnan(observation[-2])

    # check if observer was actually capable of attacking last round
    for i in range(n_states):
        if not observer_attacked[i]:
            continue

        target = attack_target_ids[i]

        prev_observer_tech_level = prev_tech_levels[i, observer_id]

        observer_capable = (
            distances_tech_level[observer_id, target] < prev_observer_tech_level
        )

        observer_attacked[i] = observer_attacked[i] & observer_capable

    # observation should agree with the previous action about attacking
    probabilities *= observer_attacked == obs_observer_attacked

    # observation should agree with the state about the outcome of the attack
    obs_attack_successful = observation[-2]
    for i in range(n_states):
        if not observer_attacked[i]:
            continue

        target = attack_target_ids[i]
        attack_successful = states[i, target, 0] == 0

        probabilities[i] *= attack_successful == obs_attack_successful

    #######

    observer_targeted = attack_target_ids == observer_id
    obs_agent_targeted = not np.isnan(observation[-1])

    # check if actor was actually capable of attacking last round
    for i in range(n_states):
        if not observer_targeted[i]:
            continue

        actor = actor_ids[i]

        prev_actor_tech_level = prev_tech_levels[i, actor]

        actor_capable = distances_tech_level[observer_id, actor] < prev_actor_tech_level

        observer_targeted[i] = observer_targeted[i] & actor_capable

    # observation should agree with previous action about being targeted
    probabilities *= observer_targeted == obs_agent_targeted

    # observation should agree with the state about the outcome of the attack
    obs_attack_successful = observation[-1]
    for i in range(n_states):
        if not observer_targeted[i]:
            continue

        attack_successful = states[i, observer_id, 0] == 0
        probabilities[i] *= attack_successful == obs_attack_successful

    ### 2. Calculate the probability from the technosignatures

    observation = observation[:n_agents]
    # # determine which agents we should have observed in the current state
    # # array of shape (n_states, n_agents)
    observed_agents = np.zeros(shape=(n_states, n_agents), dtype=np.bool_)
    for i in range(n_states):
        observer_tech_level = tech_levels[i, observer_id]

        # array of length n_agents
        distances = distances_tech_level[observer_id, :]

        observed_agents[i] = distances < observer_tech_level

    # find expected observations from all agents
    # - technosignature from neighbours
    # - from non-neighbours, we “expect” the observation we actually received
    # - technology level from self
    # this is an array of shape (n_states, n_agents)
    expected_observation = tech_levels * states[:, :, 1]
    expected_observation = np.where(observed_agents, expected_observation, observation)
    expected_observation[:, observer_id] = tech_levels[:, observer_id]

    # find individual densities of observations
    # TODO: optimise
    densities = _norm_pdf(x=observation, mean=expected_observation, sd=obs_noise_sd)
    # densities[:, observer_id] = _norm_pdf(
    #     x=observation[observer_id],
    #     mean=expected_observation[:, observer_id],
    #     sd=obs_self_noise_sd,
    # )

    # multiply to get final probabilities
    # probabilities *= densities.prod(axis=1)
    for i in range(n_states):
        probabilities[i] *= densities[i, :].prod()

    return probabilities


def sample_observation(
    state: ipomdp_solver.State,
    action: ipomdp_solver.Action,
    agent: civilisation.Civilisation,
    model: universe.Universe,
) -> ipomdp_solver.Observation:
    """
    Returns a single possible observation of “agent” when the system is \
    currently in state “state” and the previous action was “action”. \
    Observations include technosignatures from all civilisations (n_agents \
    values, where agent's own value is its technology level) and two success \
    bits, the first indicating whether the agent successfully attacked someone \
    last round or not and the second indicating whether the agent itself was \
    successfully destroyed last round or not.

    Model's random number generator (rng attribute) is used for sampling.

    Technosignature observations from each agent are assumed to have Gaussian \
    observation noise, the standard deviation of which is defined by the model \
    (obs_noise_sd and obs_self_noise_sd attributes).

    Keyword arguments:
    state: the current model state
    action: the previous action
    agent: the observing Civilisation
    model: a Universe. Used for determining distances between civilisations and \
           for random sampling.

    Returns:
    The observation. A tuple of length n_agents + 2. 
    The n_agents values correspond to technosignatures of civilisations and the
    final two bits are success bits as described above.
    The first success bit is None if the agent did not attack.
    The second success bit is None if the agent was not attacked.
    """

    actor, actor_action = action

    ### determine success bits
    agent_attacked = agent == actor and isinstance(
        actor_action, civilisation.Civilisation
    )

    # check if agent was actually capable of attacking last round
    if agent_attacked:
        target = actor_action
        prev_agent_state = state[agent.id].copy()
        prev_agent_state[0] -= 1
        prev_agent_tech_level = growth.tech_level(state=prev_agent_state, model=model)
        agent_capable = model.is_neighbour(
            agent1=agent, agent2=target, tech_level=prev_agent_tech_level
        )
        agent_attacked = agent_attacked and agent_capable

    agent_targeted = agent == actor_action

    # check if actor was actually capable of attacking last round
    if agent_targeted:
        prev_actor_state = state[actor.id].copy()
        prev_actor_state[0] -= 1
        prev_actor_tech_level = growth.tech_level(state=prev_actor_state, model=model)
        actor_capable = model.is_neighbour(
            agent1=actor, agent2=agent, tech_level=prev_actor_tech_level
        )
        agent_targeted = agent_targeted and actor_capable

    if agent_attacked:
        # determine if target was destroyed
        target_destroyed = state[target.id, 0] == 0
    else:
        target_destroyed = None

    if agent_targeted:
        # determine if agent was destroyed
        agent_destroyed = state[agent.id, 0] == 0
    else:
        agent_destroyed = None

    # calculate tech levels
    tech_levels = growth.tech_level(state=state, model=model)

    # calculate technosignatures
    technosignatures = tech_levels * state[:, 1]

    # find neighbours
    nbrs = model.get_agent_neighbours(agent=agent, tech_level=tech_levels[agent.id])

    # add noise to agent's own tech level and others' technosignatures
    orig_tech_levels = tech_levels.copy()
    tech_levels[agent.id] += model.random.gauss(mu=0, sigma=model.obs_self_noise_sd)
    technosignatures += model.rng.normal(
        loc=0, scale=model.obs_noise_sd, size=model.n_agents
    )

    observation = tuple(
        t if ag == agent else (ts if ag in nbrs else model.random.random())
        for ag, t, ts in zip(model.agents, tech_levels, technosignatures)
    )

    # add tech levels of agents to observation to help with debugging. Note that
    # these values are ignored when calculating probabilities of observations.
    observation += tuple(orig_tech_levels)

    # add result bits
    observation += (target_destroyed, agent_destroyed)

    return observation


def sample_init(
    n_samples: int, model: universe.Universe, agent: civilisation.Civilisation = None
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

    # initialise array of samples
    sample = np.zeros((n_samples, n_agents, model.agent_state_size))
    size = (n_samples, n_agents)

    # initial age distribution
    sample[..., 0] = model.rng.integers(
        *model.init_age_belief_range, size=size, endpoint=True
    )

    # initial visibility distribution
    sample[..., 1] = model.rng.uniform(*model.init_visibility_belief_range, size=size)

    # determine the values or range of the growth parameters
    if model.agent_growth == growth.sigmoid_growth:
        if (
            "speed" in model.agent_growth_params
            and "takeoff_time" in model.agent_growth_params
        ):
            # every agent has the same growth parameters
            speed_range = (
                model.agent_growth_params["speed"],
                model.agent_growth_params["speed"],
            )
            takeoff_time_range = (
                model.agent_growth_params["takeoff_time"],
                model.agent_growth_params["takeoff_time"],
            )
        elif (
            "speed_range" in model.agent_growth_params
            and "takeoff_time_range" in model.agent_growth_params
        ):
            # growth parameters are sampled from the given ranges
            speed_range = model.agent_growth_params["speed_range"]
            takeoff_time_range = model.agent_growth_params["takeoff_time_range"]
        else:
            raise Exception("Sigmoid growth parameters are incorrect")

        # sample from the ranges
        sample[..., 2] = model.rng.uniform(*speed_range, size=size)
        sample[..., 3] = model.rng.integers(
            *takeoff_time_range, size=size, endpoint=True
        )

    # if provided, agent is certain about its own state in all samples
    if agent is not None:
        sample[:, agent.id, :] = agent.get_state()

    return sample


def rollout(
    state: ipomdp_solver.State,
    agent: civilisation.Civilisation,
    model: universe.Universe,
    depth: int = 0,
) -> float:
    """
    Starting from the given model state, use random actions to propagate
    the state forward until the discount horizon is reached. Returns
    the value.

    NOTE: state is mutated in place for efficiency.

    Keyword arguments:
    state: the state to start the rollout in
    agent: the agent whose reward is of interest
    model: a Universe.
    depth: the number of steps we have rolled out so far
    """

    # if we have reached the discount horizon, stop the recursion
    if model.discount_factor**depth < model.discount_epsilon:
        return 0

    # choose actor
    actor = model.random.choice(model.agents)

    # choose action (only ones that actor is capable of)
    actor_action = model.random.choice(actor.possible_actions(state=state))
    action = (actor, actor_action)

    # calculate value of taking action in state
    value = reward(state=state, action_=action, agent=agent, model=model)

    # propagate state
    next_state = transition(state=state, action_=action, model=model, in_place=True)

    # continue rollout from the propagated state
    next_value = rollout(state=next_state, agent=agent, model=model, depth=depth + 1)

    return value + model.discount_factor * next_value


def level0_opponent_policy(
    agent: civilisation.Civilisation, model: universe.Universe
) -> ipomdp_solver.AgentAction:
    """
    Choose an agent action for agent according to the level 0 default policy.
    """
    if model.action_dist_0 == "random":
        return model.random.choice(agent.possible_actions())
    elif model.action_dist_0 == "passive":
        return action.NO_ACTION
    elif model.action_dist_0 == "aggressive":
        possible_targets = [*model.agents]
        possible_targets.remove(agent)
        return model.random.choice(possible_targets)

    raise NotImplementedError()
