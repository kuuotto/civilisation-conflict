# %%

# avoids having to give type annotations as strings
from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

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
    frame: ipomdp_solver.Frame = None,
    in_place=False,
) -> Tuple[ipomdp_solver.State, Tuple[float, ...]]:
    """
    Given a model state and an action, this function samples from the
    distribution of all possible future states. It also returns the rewards for all
    agents. In practice a (state, action) combination only has one possible future state.

    This can be used as the transition function for all agents, as they all have the same
    transition model.

    NOTE: Assumes that everyone uses the same frame for their rewards.

    Keyword arguments:
    state: representation of the system at time t-1. a NumPy array of size
           (n_agents, k), where k is the length of an individual agent state
           representation.
    action_: the action performed
    model: a Universe
    frame: the frame used for reward calculation
    in_place: boolean, whether to change the state object directly

    Returns:
    The system state at time t and rewards for each agent
    """
    # copy state so we don't change original
    if not in_place:
        state = state.copy()

    # get the appropriate attack reward
    attack_reward = (
        model.rewards["destroyed"] if frame is None else frame["attack_reward"]
    )

    # tech levels before the action. This is only calculated if at least one of the
    # agent actions is an attack.
    prev_tech_levels = None

    # keep track of rewards (can also be used to check which agents have been destroyed)
    rewards = [0 for agent in model.agents]

    # copy over the current ages to the previous ages column
    state[:, 0] = state[:, 1]

    for agent, agent_action in zip(model.agents, action_, strict=True):
        if agent_action == action.HIDE:
            # if agent has already been destroyed
            agent_destroyed = rewards[agent.id] == model.rewards["destroyed"]

            # reward for hiding
            rewards[agent.id] += model.rewards["hide"]

            # if agent is already destroyed by someone else, hiding doesn't do anything
            if agent_destroyed:
                continue

            # if actor hides, additionally update their visibility
            state[agent.id, 2] *= model.visibility_multiplier

        elif isinstance(agent_action, civilisation.Civilisation):
            # calculate previous tech levels if they have not yet been calculated
            if prev_tech_levels is None:
                prev_tech_levels = growth.tech_level(
                    state=state, model=model, previous=True
                )

            ### See if the attack is successful
            target = agent_action

            # determine if actor is capable of attacking target
            actor_capable = model.is_neighbour(
                agent1=agent, agent2=target, tech_level=prev_tech_levels[agent.id]
            )

            if (
                actor_capable
                and prev_tech_levels[agent.id] > prev_tech_levels[target.id]
            ):
                # civilisation is destroyed
                state[target.id, 1] = -1  # reset age
                state[target.id, 2] = 1  # reset visibility factor
                rewards[target.id] += model.rewards["destroyed"]
                rewards[agent.id] += attack_reward

    # always tick everyone's time by one
    state[:, 1] += 1

    return state, rewards


def prob_observation(
    observation: ipomdp_solver.Observation,
    states: NDArray,  # shape (n_states, n_agents, k)
    attack_target_ids: NDArray,  # shape (n_states, n_agents)
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
    prev_tech_levels = growth.tech_level(state=states, model=model, previous=True)

    return _prob_observation(
        observation=observation,
        states=states,
        observer_id=observer.id,
        tech_levels=tech_levels,
        prev_tech_levels=prev_tech_levels,
        attack_target_ids=attack_target_ids,
        distances_tech_level=model._distances_tech_level,
        n_agents=model.n_agents,
        obs_noise_sd=model.obs_noise_sd,
        obs_self_noise_sd=model.obs_self_noise_sd,
    )


@njit
def _prob_observation(
    observation: ipomdp_solver.Observation,  # shape (n_agents + 2,)
    states: ipomdp_solver.State,  # shape (n_states, n_agents, k)
    observer_id: int,
    tech_levels: NDArray,  # shape (n_states, n_agents)
    prev_tech_levels: NDArray,  # shape (n_states, n_agents)
    attack_target_ids: NDArray,  # shape (n_states, n_agents)
    distances_tech_level: NDArray,  # shape (n_agents, n_agents)
    n_agents: int,
    obs_noise_sd: float,
    obs_self_noise_sd: float,
) -> float:
    """
    Returns the probability density of a given observation by an observer \
    (represented by its id, "observer_id") given a set of current system states \
    ("states") and previous attacks encoded by "attack_target_ids".

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
    attack_target_ids: an array of shape (n_states, n_agents), each row storing the \
                       ids of possible targets of attacks. If an agent's previous \
                       action was not an attack, the corresponding value is -1.
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
    ### observer's attack bit

    # observed bit
    obs_attack_successful = observation[-2]

    # calculates the correct observation bit for each state and compares to the
    # observed one
    for i in range(n_states):
        attack_successful = np.nan

        target_id = attack_target_ids[i, observer_id]

        if target_id != -1:
            # determine if the attack actually took place
            prev_observer_tech_level = prev_tech_levels[i, observer_id]

            observer_capable = (
                distances_tech_level[observer_id, target_id] < prev_observer_tech_level
            )

            if observer_capable:
                # observer was capable, let's see what the result is
                # note that observer was not necessarily the one who destroyed the
                # target
                attack_successful = states[i, target_id, 1] == 0

        probabilities[i] = attack_successful == obs_attack_successful

    ### observer targeted bit

    obs_attack_on_observer_successful = observation[-1]

    # calculates the correct bit for each state and compares to the observed one
    for i in range(n_states):
        attack_on_observer_successful = np.nan

        for other_agent_id in range(n_agents):
            if other_agent_id == observer_id:
                continue

            other_agent_attacked_observer = (
                attack_target_ids[i, other_agent_id] == observer_id
            )

            if other_agent_attacked_observer:
                # check if the other agent was actually capable of attacking
                prev_other_agent_tech_level = prev_tech_levels[i, other_agent_id]

                other_agent_capable = (
                    distances_tech_level[other_agent_id, observer_id]
                    < prev_other_agent_tech_level
                )

                if other_agent_capable:
                    # agent was capable, let's check what the result is
                    attack_on_agent_successful = states[i, observer_id, 1] == 0

                    if attack_on_agent_successful:
                        # if we have a successful attack, no need to check other agents.
                        # note that it might not have been other_agent who destroyed
                        # the observer
                        break

        probabilities[i] = (
            attack_on_observer_successful == obs_attack_on_observer_successful
        )

    ### 2. Calculate the probability from the technosignatures

    observation = observation[:n_agents]

    # find expected observations from all agents
    # - technosignature from neighbours
    # - from non-neighbours, we “expect” the observation we actually received
    # - technology level from self
    # this is an array of shape (n_states, n_agents)
    expected_observation = tech_levels * states[:, :, 2]

    for i in range(n_states):
        observer_tech_level = tech_levels[i, observer_id]

        # array of length n_agents
        non_observed_agents = distances_tech_level[observer_id, :] > observer_tech_level

        # we 'expect' the received observation from the agents we think we cannot observe,
        # since this observation is then just random noise
        expected_observation[i, non_observed_agents] = observation[non_observed_agents]

    # from the observer we expect to observe their tech level
    expected_observation[:, observer_id] = tech_levels[:, observer_id]

    # find individual densities of observations
    densities = _norm_pdf(x=observation, mean=expected_observation, sd=obs_noise_sd)
    densities[:, observer_id] = _norm_pdf(
        x=observation[observer_id],
        mean=expected_observation[:, observer_id],
        sd=obs_self_noise_sd,
    )

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
    values, where agent's own value is its noisy technology level) and two success \
    bits, the first indicating whether the agent successfully attacked someone \
    last round or not and the second indicating whether the agent itself was \
    destroyed last round or not.

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
    The observation. An array of length n_agents + 2. 
    The n_agents values correspond to technosignatures of civilisations and the
    final two bits are success bits as described above.
    The first success bit is np.nan if the agent did not attack.
    The second success bit is np.nan if the agent was not attacked.
    """
    agent_action = action[agent.id]

    # technology levels at the previous time step are only computed when they are
    # first needed
    prev_tech_levels = None

    ### determine success bits
    ## 1. Check if agent successfully attacked someone
    agent_attacked = isinstance(agent_action, civilisation.Civilisation)
    target_destroyed = np.nan

    # check if agent was actually capable of attacking last round
    if agent_attacked:
        target = agent_action

        # calculate previous tech levels if necessary
        if prev_tech_levels is None:
            prev_tech_levels = growth.tech_level(
                state=state, model=model, previous=True
            )

        agent_capable = model.is_neighbour(
            agent1=agent, agent2=target, tech_level=prev_tech_levels[agent.id]
        )

        if agent_capable:
            # agent was capable, let's see the result
            # note that it was not necessarily agent who destroyed the target
            target_destroyed = state[target.id, 1] == 0

    ## 2. Check if agent was destroyed by someone else
    agent_destroyed = np.nan
    for other_agent, other_agent_action in zip(model.agents, action, strict=True):
        if other_agent_action != agent:
            continue

        # calculate previous tech levels if necessary
        if prev_tech_levels is None:
            prev_tech_levels = growth.tech_level(
                state=state, model=model, previous=True
            )

        # other_agent tried to attack agent. Let's check the result
        other_agent_capable = model.is_neighbour(
            agent1=other_agent,
            agent2=agent,
            tech_level=prev_tech_levels[other_agent.id],
        )

        if other_agent_capable:
            # other agent was capable of attacking agent. Let's check the result.
            agent_destroyed = state[agent.id, 1] == 0

            if agent_destroyed:
                # if agent is destroyed, we can stop checking other agents.
                # note that it was not necessarily other_agent who destroyed agent
                break

    ### Determine technosignature observations

    # initialise observation as noise
    observation = model.rng.random(size=model.n_agents + 2)

    # store result bits
    observation[-2] = target_destroyed
    observation[-1] = agent_destroyed

    # calculate current tech levels
    tech_levels = growth.tech_level(state=state, model=model)

    # calculate technosignatures
    technosignatures = tech_levels * state[:, 2]

    # add noise to technosignatures
    technosignatures += model.rng.normal(
        loc=0, scale=model.obs_noise_sd, size=model.n_agents
    )

    # find neighbours
    nbrs = model.get_agent_neighbours(agent=agent, tech_level=tech_levels[agent.id])

    # add neighbours' noisy technosignatures to observation
    for nbr in nbrs:
        observation[nbr.id] = technosignatures[nbr.id]

    # add agent's own technology level with noise
    observation[agent.id] = tech_levels[agent.id] + model.random.gauss(
        mu=0, sigma=model.obs_self_noise_sd
    )

    return observation


def uniform_initial_belief(
    n_samples: int, model: universe.Universe, agent: civilisation.Civilisation = None
) -> ipomdp_solver.Belief:
    """
    Generates n_samples samples of the initial belief. Assumes a uniform distribution
    over possible ranges of variables, independently for each agent and variable.

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


def surpass_scenario_initial_belief(
    n_samples: int,
    level: int,
    model: universe.Universe,
) -> ipomdp_solver.Belief:
    """
    In this two-agent scenario, agent 0 is stronger than agent 1 (and can reach \
    agent 1).

    The samples will be generated such that:
    - 0 is stronger than 1
    - 0 can reach 1
    - 1 will surpass 0 in technology level within 'time_until_surpass' time steps \ 
      with probability prob_surpass_{level}

    The parameters time_until_surpass and prob_surpass_{0, 1} can be accessed through \
    the model.initial_belief_params dictionary
    
    NOTE: This algorithm is extremely inefficient, but should be okay compared to
    planning time.
    """
    assert model.n_agents == 2
    assert level in (0, 1)

    # initialise array of samples
    n_agents = 2
    samples = np.zeros((n_samples, n_agents, model.agent_state_size))

    # determine ranges for growth parameters
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
    else:
        raise NotImplementedError()

    prob_surpass = model.initial_belief_params[f"prob_surpass_{level}"]
    time_until_surpass = model.initial_belief_params["time_until_surpass"]

    # whether to generate a sample where a surpass happens next time. We decide what
    # kind of sample to generate before sampling and rejecting to avoid bias.
    generate_surpass = model.random.random() < prob_surpass

    # number of samples generated so far
    n_generated = 0

    while n_generated < n_samples:
        ### generate a random sample

        # age
        samples[n_generated, :, 0] = model.rng.integers(
            *model.init_age_belief_range, size=n_agents, endpoint=True
        )

        # visibility factor
        samples[n_generated, :, 1] = model.rng.uniform(
            *model.init_visibility_belief_range, size=n_agents
        )

        # growth speed
        samples[n_generated, :, 2] = model.rng.uniform(*speed_range, size=n_agents)

        # growth takeoff age
        samples[n_generated, :, 3] = model.rng.integers(
            *takeoff_time_range, size=n_agents, endpoint=True
        )

        ### Check that the conditions hold
        # 1. 0 is stronger than 1
        tech_levels = growth.tech_level(samples[n_generated], model=model)

        if tech_levels[0] < tech_levels[1]:
            # swap
            stronger_state = samples[n_generated, 1].copy()
            weaker_state = samples[n_generated, 0].copy()
            samples[n_generated, 0] = stronger_state
            samples[n_generated, 1] = weaker_state

        # 2. 0 can reach 1
        can_reach = model.is_neighbour(
            agent1=model.agents[0], agent2=model.agents[1], tech_level=tech_levels[0]
        )
        if not can_reach:
            continue

        # 3. 1 will surpass 0
        samples[n_generated, :, 0] += time_until_surpass
        future_tech_levels = growth.tech_level(samples[n_generated], model=model)

        if generate_surpass and future_tech_levels[1] < future_tech_levels[0]:
            continue
        elif not generate_surpass and future_tech_levels[1] > future_tech_levels[0]:
            continue

        # we accept the new sample. We also decide which kind of sample to generate next
        samples[n_generated, :, 0] -= time_until_surpass
        generate_surpass = model.random.random() < prob_surpass
        n_generated += 1

    return samples


def rollout(
    state: ipomdp_solver.State,
    agent: civilisation.Civilisation,
    frame: ipomdp_solver.Frame,
    model: universe.Universe,
    depth: int = 0,
) -> float:
    """
    Starting from the given model state, use random actions to propagate
    the state forward until the discount horizon is reached. Returns
    the value. Rewards are calculated using the supplied frame.

    NOTE: state is mutated in place for efficiency.

    Keyword arguments:
    state: the state to start the rollout in
    agent: the agent whose reward is of interest
    frame: the frame to use to calculate the rewards
    model: a Universe.
    depth: the number of steps we have rolled out so far
    """

    # if we have reached the discount horizon, stop the recursion
    if model.discount_factor**depth < model.discount_epsilon:
        return 0

    # choose actions
    action = []

    for actor in model.agents:
        # choose action (random if we choose, default policy for other actors)
        if (actor == agent) or (actor != agent and model.action_dist_0 == "random"):
            actor_action = model.random.choice(actor.possible_actions(state=state))
        else:
            actor_action = level0_opponent_policy(agent=actor, model=model)

        action.append(actor_action)

    # propagate state
    next_state, rewards = transition(
        state=state, action_=tuple(action), model=model, frame=frame, in_place=True
    )
    agent_reward = rewards[agent.id]

    # continue rollout from the propagated state
    next_value = rollout(
        state=next_state, agent=agent, frame=frame, model=model, depth=depth + 1
    )

    return agent_reward + model.discount_factor * next_value


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
