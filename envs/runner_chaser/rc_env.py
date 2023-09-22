import functools
from typing import Tuple, Dict, TypeAlias, Mapping

import gymnasium as gym
from gymnasium.spaces import Discrete

from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers

import numpy as np

from envs.runner_chaser.rc_posg import (
    RCPOSG,
    Agent,
    AgentType,
    Action,
    Reward,
    State,
)

# the environment uses a different type of action and observation than the POSG itself.
# This is the data type used by gym.spaces.Discrete
EnvAction: TypeAlias = np.int_ | int
EnvObservation: TypeAlias = np.int_ | int

# time until the process is truncated (terminated due to a time limit external to the
# POSG)
TIME_CUTOFF = 20


def env(render_mode=None, **kwargs):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode, **kwargs)
    # This wrapper is only for environments which print results to the terminal
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env


def raw_env(render_mode=None, **kwargs):
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    """
    env = parallel_env(render_mode=render_mode, **kwargs)
    env = parallel_to_aec(env)
    return env


class parallel_env(ParallelEnv[Agent, EnvObservation, EnvAction | None]):
    metadata = {"render_modes": ["human"], "name": "rc_v1"}

    def __init__(self, map_name: str, render_mode: str | None = None):
        """
        Initialises the environment. This includes defining the possible_agents and
        render_mode attributes (which should not be changed later on) and initialising
        the underlying POSG.
        """
        self.possible_agents = [
            Agent(type=AgentType.CHASER, id=0),
            Agent(type=AgentType.RUNNER, id=1),
        ]
        self.render_mode = render_mode

        # initialise the underlying POSG
        self._posg = RCPOSG(map_name=map_name)

        # track whether the process is done (terminated or truncated)
        self._done = True

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: Agent):
        """Returns the observation space of the given agent"""
        # there is only one possible observation (which conveys no information)
        return Discrete(1)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: Agent):
        """Returns the action space of the given agent"""
        return Discrete(4)

    def render(self):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling the render method without specifying a render mode."
            )
            return

        runner_location = self._state[0]
        chaser_location = self._state[1]

        print(
            f"At t={self.time}: Runner at {runner_location}, Chaser at {chaser_location}"
        )

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

    def reset(
        self, seed=None, options=None
    ) -> Tuple[Dict[Agent, EnvObservation], Dict[Agent, Dict]]:
        """
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.
        Here it initializes the `num_moves` variable which counts the number of
        hands that are played.
        Returns the observations for each agent
        """
        self.agents = self.possible_agents[:]

        # initialise step counter
        self.time = 0

        # initialise the state
        self._state: State = (
            self._posg._grid.runner_start_location,
            self._posg._grid.chaser_start_location,
        )

        # create initial observation. Here we could call the sample_observation method
        # from the POSG, but as there are no observations in this model, that can be
        # skipped
        observations: Dict[Agent, EnvObservation] = {agent: 0 for agent in self.agents}

        # info dictionary
        infos = {agent: {} for agent in self.agents}

        self._done = False

        return observations, infos

    def step(
        self, actions: Mapping[Agent, EnvAction]
    ) -> Tuple[
        Dict[Agent, EnvObservation],
        Dict[Agent, Reward],
        Dict[Agent, bool],
        Dict[Agent, bool],
        Dict[Agent, Dict],
    ]:
        """
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - terminations
        - truncations
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """
        if self._done:
            return {}, {}, {}, {}, {}

        # get the actions of both agents
        runner = self.agents[0]
        chaser = self.agents[1]
        if runner not in actions:
            raise Exception("Action not supplied for runner")
        if chaser not in actions:
            raise Exception("Action not supplied for chaser")
        # convert EnvActions (ints) into Actions
        runner_action = Action(actions[runner])
        chaser_action = Action(actions[chaser])
        joint_action = (runner_action, chaser_action)

        # get the next state and rewards with the transition function of the POSG
        next_state, rewards, terminated = self._posg.transition(
            self._state, joint_action=joint_action
        )

        # increase time
        self.time += 1

        # format rewards into a dictionary
        rewards = {runner: rewards[0], chaser: rewards[1]}

        # format termination information into a dictionary
        # all agents terminate at the same time
        assert terminated[0] == terminated[1]
        terminated = terminated[0]
        terminations = {runner: terminated, chaser: terminated}

        # determine time-based truncation
        truncated = self.is_truncated and not terminated
        truncations = {agent: truncated for agent in self.agents}

        # observations are always empty (we could call the sample_observation method of
        # the POSG but there is no real need as the observations are always empty)
        observations: Dict[Agent, EnvObservation] = {agent: 0 for agent in self.agents}

        # empty infos
        infos = {agent: {} for agent in self.agents}

        # update current state
        self._state = next_state

        # if process is truncated, remove agents to indicate this
        if truncated or terminated:
            self.agents = []
            self._done = True

        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminations, truncations, infos

    @property
    def is_truncated(self):
        return self.time >= TIME_CUTOFF

    def state(self) -> State:
        return self._state
