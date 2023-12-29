# avoids having to give type annotations as strings
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from random import Random
from typing import (
    Iterator,
    Mapping,
    Sequence,
    TypeAlias,
)

import numpy as np
from numpy.typing import NDArray

# agents should be represented by integers 0, 1, ...
Agent: TypeAlias = int

# states, actions and observations are arrays
State: TypeAlias = NDArray
Action: TypeAlias = NDArray
Observation: TypeAlias = NDArray

JointAction = Mapping[Agent, Action]
Reward: TypeAlias = float
# using Dict instead of Mapping because this is only used as a return type
JointReward: TypeAlias = dict[Agent, Reward]


class ActivationSchedule(Enum):
    JointActivation = 0  # all agents act each turn
    RandomActivation = 1  # a randomly chosen agent acts each turn


class Model(ABC):
    """
    An abstract class for a model. A model can be either intentional (I-POMDP / POMDP)
    or subintentional (e.g. a no-information model).
    """


class FixedPolicyModel(Model):
    """
    A model where an agent is assumed to choose an action randomly.
    """

    agent: Agent

    @abstractmethod
    def act(self) -> Action:
        """
        A fixed policy should return an action for agent.
        """


class IPOMDP(Model):
    """
    An abstract base class for I-POMDPs. This means that we define the methods
    an I-POMDP class must implement.
    """

    ## type hint some instance attributes
    owner: Agent
    # owner_reasoning_level: int
    activation_schedule: ActivationSchedule
    discount_factor: float
    agents: Sequence[Agent]

    # define the shapes of state and action arrays
    state_shape: tuple[int, ...]
    action_shape: tuple[int, ...]

    # define the data types of the state and action arrays
    state_dtype: np.dtype
    action_dtype: np.dtype

    # define which action is the "no turn" action
    no_turn_action: Action

    @abstractmethod
    def agent_models(self, agent: Agent) -> tuple[FixedPolicyModel, tuple[IPOMDP, ...]]:
        """
        Given an agent, return a default policy and a tuple (possibly empty) of rational
        models. For the owner of the owner of the I-POMDP there is only a default
        policy.
        """

    @abstractmethod
    def possible_actions(self, agent: Agent) -> tuple[Action, ...]:
        """
        Given an agent, returns a tuple of all possible actions of that agent.
        """

    @abstractmethod
    def utility_estimate(self, state: State) -> float:
        """
        Given a state, this method should return an estimate of the utility of that
        state (expected discounted sum of rewards assuming optimal action choices).
        This could be a random roll-out, for example, or domain-specific knowledge
        can be incorporated.
        """

    @abstractmethod
    def transition(
        self, state: State, joint_action: JointAction
    ) -> tuple[State, Reward]:
        """
        Given a joint action taken in a state, return a sample from the distribution
        of next states and the reward for the owner.
        """

    @abstractmethod
    def sample_observation(
        self, state: State, prev_joint_action: JointAction
    ) -> Observation:
        """
        Given the current state and the previous joint action taken, sample an
        observation for the owner from the distribution of possible observations.
        """

    @abstractmethod
    def prob_observation(
        self,
        observation: Observation,
        states: NDArray,
        prev_joint_actions: NDArray,
    ) -> NDArray:
        """
        Given an observation, an array of states and an array of previous joint
        actions, return the probability (density) of owner receiving the observation
        for each state-previous joint action combination.

        Keyword arguments:
        observation: the received observation
        states: an array (of shape (k, *self.state_shape)) containing the states where
                the probability should be calculated
        prev_joint_actions: an array (of shape (k, n_agents, *self.action_shape))
                            containing the previously performed actions corresponding
                            to each state

        Returns:
        An array of length k which contains the (possibly unnormalised) probabilities.
        """

    @abstractmethod
    def initial_belief(
        self,
        n_states: int,
    ) -> tuple[NDArray, tuple[dict[Agent, IPOMDP], ...]]:
        """
        Generate a given number of states from the initial belief distribution.
        A single sample consists of the state (a NumPy array) and a corresponding
        rational model, i.e. an I-POMDP, for each other agent. If the other agent is
        not modelled with a rational model, None should be returned instead.

        The initial belief of the owner about the beliefs of other agents are encoded
        by the initial_belief methods of these nested I-POMDPs. This means that these
        nested states are assigned uniform weights.

        Keyword arguments:
        n_states: the number of states to generate

        Returns:
        A 2-tuple. The first element is a numpy array of shape (n_states,
        *self.state_shape) and corresponds to the state samples. The second element is
        a tuple of length n_states. Each element in this tuple is a dictionary where
        Agents are keys and IPOMDPs values. The I-POMDP returned should correspond to
        one returned by agent_models.
        """

    @abstractmethod
    def add_noise_to_state(
        self,
        state: State,
        random: Random,
    ) -> State:
        """
        Add noise to the given state. This is used by the algorithm to create a unique
        new state after a joint action is chosen which has already been used to
        expand a particle.

        Keyword arguments:
        state: the state (a Numpy array of shape state_shape) to add noise to
        random: a random number generator object from the solver
        """

    @property
    def other_agents(self) -> Iterator[Agent]:
        yield from (ag for ag in self.agents if ag != self.owner)

    @property
    def n_agents(self) -> int:
        return len(self.agents)
