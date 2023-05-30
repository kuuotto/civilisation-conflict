# %%

# avoids having to give type annotations as strings
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # avoid circular imports with type hints
    from model import universe, ipomdp_solver

import numpy as np
import math
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
    return (1 / (sd * math.sqrt(2 * math.pi))) * math.exp(
        -(1 / 2) * ((x - mean) / sd) ** 2
    )


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

        # calculate tech levels of actor and target
        tech_levels = growth.tech_level(state=state, model=model)

        # determine if actor is capable of attacking target
        actor_influence_radius = growth.influence_radius(tech_levels[actor.id])
        actor_capable = model.is_neighbour(
            agent1=actor, agent2=target, radius=actor_influence_radius
        )

        if (actor_capable and tech_levels[actor.id] > tech_levels[target.id]) or (
            actor_capable
            and tech_levels[actor.id] == tech_levels[target.id]
            and model.rng.random() > 0.5
        ):
            # civilisation is destroyed
            state[target.id, 0] = 0  # reset time
            state[target.id, 1] = 1  # visibility factor

    elif actor_action != action.NO_ACTION:
        print(action_, actor, actor_action)
        raise Exception("Incorrect action format")

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
    assert len(action_) == 1
    actor, actor_action = next(iter(action_.items()))

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
    actor_influence_radius = growth.influence_radius(tech_levels[actor.id])
    actor_capable = model.is_neighbour(
        agent1=actor, agent2=target, radius=actor_influence_radius
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
    state: ipomdp_solver.State,
    action: ipomdp_solver.Action,
    agent: civilisation.Civilisation,
    model: universe.Universe,
    tech_levels=None,
    agent_influence_radius=None,
    prev_influence_radii=None,
) -> float:
    """
    Returns the probability (density) of a given observation by “agent”, given
    that the system is currently in state “state” and the previous action was
    “action”.

    Technosignature observations from each agent are assumed to have Gaussian
    observation noise, which is saved in the model's obs_noise_sd attribute.

    Keyword arguments:
    observation: the observation received
    state: the current (assumed) system state
    action_: previous action
    agent: the observing Civilisation
    model: a Universe. Used for determining distances between civilisations.
    tech_levels: an array of length n_agents where each value represents the
                 technology level of each agent in state. Can be supplied to increase
                 performance when calculating probabilities for multiple particles at
                 once.
    agent_influence_radius: the influence radius of agent in state. Can be supplied
                            to increase performance when calculating proabilities for
                            multiple particles at once.
    prev_influence_radii: an array of length n_agents where each value represents the
                          influence radius of each agent in the previous time step
                          (assuming just passage of time). Can be supplied to increase
                          performance when calculating probabilities for multiple
                          particles at once.
    """

    ### 1. determine if the success bits in the observation are correct

    agent_attacked = agent in action and isinstance(
        action[agent], civilisation.Civilisation
    )
    obs_agent_attacked = observation[-2] is not None

    # check if agent was actually capable of attacking last round
    if agent_attacked:
        target = action[agent]

        # determine previous influence radius of agent
        if prev_influence_radii is None:
            prev_agent_state = state[agent.id]
            prev_agent_state[0] -= 1
            prev_agent_tech_level = growth.tech_level(
                state=prev_agent_state, model=model
            )
            prev_agent_state[0] += 1  # revert changes
            prev_agent_influence_radius = growth.influence_radius(prev_agent_tech_level)
        else:
            prev_agent_influence_radius = prev_influence_radii[agent.id]

        agent_capable = model.is_neighbour(
            agent1=agent, agent2=target, radius=prev_agent_influence_radius
        )
        agent_attacked = agent_attacked and agent_capable

    # observation should agree with the previous action about attacking
    if agent_attacked != obs_agent_attacked:
        return 0

    # observation should agree with the state about the outcome of the attack
    if agent_attacked:
        target_id = action[agent].id
        attack_successful = state[target_id, 0] == 0
        obs_attack_successful = observation[-2]

        if attack_successful != obs_attack_successful:
            return 0

    agent_targeted = agent in action.values()
    obs_agent_targeted = observation[-1] is not None

    # check if actor was actually capable of attacking last round
    if agent_targeted:
        actor = next(iter(action.keys()))

        # determine previous influence radius of actor
        if prev_influence_radii is None:
            prev_actor_state = state[actor.id]
            prev_actor_state[0] -= 1
            prev_actor_tech_level = growth.tech_level(
                state=prev_actor_state, model=model
            )
            prev_actor_state[0] += 1  # revert changes
            prev_actor_influence_radius = growth.influence_radius(prev_actor_tech_level)
        else:
            prev_actor_influence_radius = prev_influence_radii[actor.id]

        actor_capable = model.is_neighbour(
            agent1=actor, agent2=agent, radius=prev_actor_influence_radius
        )
        agent_targeted = agent_targeted and actor_capable

    # observation should agree with previous action about being targeted
    if agent_targeted != obs_agent_targeted:
        return 0

    # observation should agree with the state about the outcome of the attack
    if agent_targeted:
        attack_successful = state[agent.id, 0] == 0
        obs_attack_successful = observation[-1]

        if attack_successful != obs_attack_successful:
            return 0

    ### 2. Calculate the probability from the technosignatures

    # calculate tech levels
    if tech_levels is None:
        tech_levels = growth.tech_level(state=state, model=model)

    # calculate agent influence radius
    if agent_influence_radius is None:
        agent_influence_radius = growth.influence_radius(tech_levels[agent.id])

    # determine which agents we should have observed in the current state
    observed_agents = model.get_agent_neighbours(
        agent=agent, radius=agent_influence_radius
    ) + (agent,)

    # find expected observations from all agents (technosignature from
    # neighbours, technology level from self)
    expected_observation = tech_levels * state[:, 1]
    expected_observation[agent.id] = tech_levels[agent.id]

    # find individual densities of observations and multiply
    density = math.prod(
        _norm_pdf(x=obs, mean=exp, sd=model.obs_noise_sd)
        for ag, obs, exp in zip(
            model.agents, observation[: model.n_agents], expected_observation
        )
        if ag in observed_agents
    )

    return density


def sample_observation(
    state: ipomdp_solver.State,
    action: ipomdp_solver.Action,
    agent: civilisation.Civilisation,
    model: universe.Universe,
) -> ipomdp_solver.Observation:
    """
    Returns a single possible observation of “agent” when the system is
    currently in state “state” and the previous action was “action”.
    Observations include technosignatures from all civilisations (n_agents
    values, where agent's own value is its technology level) and two success
    bits, the first indicating whether the agent successfully attacked someone
    last round or not and the second indicating whether the agent itself was
    successfully destroyed last round or not.

    Model's random number generator (rng attribute) is used for sampling.

    Technosignature observations from each agent are assumed to have Gaussian
    observation noise, which is saved in the model's obs_noise_sd attribute.

    Keyword arguments:
    state: the current model state
    action: the previous action
    agent: the observing Civilisation
    model: a Universe. Used for determining distances between civilisations and
           for random sampling.

    Returns:
    The observation. A NumPy array of size n_agents + 2.
    The n_agents values correspond to technosignatures of civilisations and the
    final two bits are success bits as described above.
    The first success bit is np.nan if the agent did not attack.
    The second success bit is np.nan if the agent was not attacked.
    """

    ### determine success bits
    agent_attacked = agent in action and isinstance(
        action[agent], civilisation.Civilisation
    )

    # check if agent was actually capable of attacking last round
    if agent_attacked:
        target = action[agent]
        prev_agent_state = state[agent.id].copy()
        prev_agent_state[0] -= 1
        prev_agent_tech_level = growth.tech_level(state=prev_agent_state, model=model)
        prev_agent_influence_radius = growth.influence_radius(prev_agent_tech_level)
        agent_capable = model.is_neighbour(
            agent1=agent, agent2=target, radius=prev_agent_influence_radius
        )
        agent_attacked = agent_attacked and agent_capable

    agent_targeted = agent in action.values()

    # check if actor was actually capable of attacking last round
    if agent_targeted:
        actor = next(iter(action.keys()))
        prev_actor_state = state[actor.id].copy()
        prev_actor_state[0] -= 1
        prev_actor_tech_level = growth.tech_level(state=prev_actor_state, model=model)
        prev_actor_influence_radius = growth.influence_radius(prev_actor_tech_level)
        actor_capable = model.is_neighbour(
            agent1=actor, agent2=agent, radius=prev_actor_influence_radius
        )
        agent_targeted = agent_targeted and actor_capable

    if agent_attacked:
        # determine if target was destroyed
        target_id = action[agent].id
        target_destroyed = state[target_id, 0] == 0
    else:
        target_destroyed = None

    if agent_targeted:
        # determine if agent was destroyed
        agent_destroyed = state[agent.id, 0] == 0
    else:
        agent_destroyed = None

    # calculate tech levels
    tech_levels = growth.tech_level(state=state, model=model)
    agent_tech_level = tech_levels[agent.id]

    # calculate technosignatures
    technosignatures = tech_levels * state[:, 1]

    # find neighbours
    nbrs = model.get_agent_neighbours(
        agent=agent, radius=growth.influence_radius(agent_tech_level)
    )

    # add noise to agent's own tech level and others' technosignatures
    agent_tech_level += model.random.gauss(mu=0, sigma=model.obs_noise_sd)
    technosignatures += model.rng.normal(
        loc=0, scale=model.obs_noise_sd, size=model.n_agents
    )

    observation = tuple(
        t if ag == agent else (ts if ag in nbrs else model.random.random())
        for ag, t, ts in zip(model.agents, tech_levels, technosignatures)
    )

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

    # determine the number of values needed to describe an agent
    if model.agent_growth == growth.sigmoid_growth:
        k = 4
    else:
        raise NotImplementedError()

    # initialise array of samples
    sample = np.zeros((n_samples, n_agents, k))
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
    action = {actor: actor_action}

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
