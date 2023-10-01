from abc import ABC, abstractmethod
from typing import (
    TypeVar,
    Generic,
    Mapping,
    TypeAlias,
    Callable,
    Optional,
    Sequence,
    Iterator,
)
from enum import Enum

# type annotations
Agent = TypeVar("Agent")
State = TypeVar("State")
Action = TypeVar("Action")
JointAction = Mapping[Agent, Action]
Observation = TypeVar("Observation")
Reward: TypeAlias = float
# using Dict instead of Mapping because this is only used as a return type
JointReward = dict[Agent, Reward]


class ActivationSchedule(Enum):
    JointActivation = 0  # all agents act each turn
    RandomActivation = 1  # a randomly chosen agent acts each turn


# class Frame:
#     """
#     Encapsulates important information related to decision-making. Specifically,
#     contains the transition/reward function and the observation sampling and
#     probability functions. A frame is always related to a specific agent.

#     Note that the frame implicitly defines the space of joint actions. The optimality
#     criterion is always assumed to be maximising the discounted expected future reward.
#     """

#     def __init__(
#         self,
#         agent: Agent,
#         transition_func: Callable[[State, JointAction], Tuple[State, JointReward]],
#         observation_sampling_func: Callable[[State, JointAction], Observation],
#         observation_probability_func: Callable[
#             [Observation, State, JointAction], float
#         ],
#         discount_factor: float,
#     ) -> None:
#         """
#         Initialises the frame.

#         Keyword arguments:
#         agent: the agent whose frame this is
#         transition_func: the transition function used by the agent
#         observation_sampling_func: the observation sampling function used by the agent
#         observation_probabiliity_func: the observation weighting function used by the agent
#         """
#         self.agent = agent
#         self.transition = transition_func
#         self.sample_observation = observation_sampling_func
#         self.prob_observation = observation_probability_func
#         self.discount_factor = discount_factor


class Model:
    """
    An abstract class for a model. A model can be either intentional (I-POMDP / POMDP)
    or subintentional (e.g. a no-information model).
    """


class FixedPolicyModel(ABC, Model, Generic[Agent, State, Action]):
    """
    A model where an agent is assumed to choose an action randomly.
    """

    agent: Agent

    @abstractmethod
    def act(self) -> Action:
        """
        A fixed policy should return an action for agent.
        """


class IPOMDP(ABC, Model, Generic[Agent, State, Action, Observation]):
    """
    Defines an abstract base class for I-POMDPs. This means that we define the methods
    an I-POMDP class must implement.

    Each instance should additionally have the following attributes:
    owner_reasoning_level: the reasoning level of the owner of the I-POMDP
    level_0_model_type: whether the model on level 0 is intentional or not.
    """

    # type hint some instance attributes
    owner: Agent
    owner_reasoning_level: int
    activation_schedule: ActivationSchedule
    discount_factor: float
    agents: Sequence[Agent]

    # def level_model_type(self, level: int) -> ModelType | None:
    #     """
    #     Given a level, this method should tell the kind of model used on this level.
    #     The supplied level should not be larger than owner_reasoning_level.
    #     """
    #     if level > self.owner_reasoning_level:
    #         return None
    #     elif level > 0:
    #         return ModelType.Intentional
    #     else:
    #         return self.level_0_model_type

    @abstractmethod
    def agent_models(self, agent: Agent) -> list[Model]:
        """
        Given an agent, return a list of possible models used for agent.
        """

    @property
    def other_agents(self) -> Iterator[Agent]:
        yield from (ag for ag in self.agents if ag != self.owner)

    # @abstractmethod
    # def possible_frames(
    #     self,
    #     agent: Agent,
    #     level: int,
    #     parent: Optional[Agent] = None,
    # ) -> List[Frame]:
    #     """
    #     Return a list of possible frames prescribed by parent for agent who is at the
    #     given reasoning level.

    #     Keyword arguments:
    #     agent: the agent whose frames are of interest
    #     level: the reasoning level of the agent
    #     parent: the agent who models "agent" using the frames
    #     """

    @abstractmethod
    def transition(
        self, state: State, joint_action: JointAction
    ) -> tuple[State, JointReward]:
        """
        Given a joint action taken in a state, return a sample from the distribution
        of next states and the rewards for each agent.
        """

    @abstractmethod
    def sample_observation(
        self, agent: Agent, state: State, prev_joint_action: JointAction
    ) -> Observation:
        """
        Given the current state and the previous joint action taken, sample an
        observation for agent from the distribution of possible observations.
        """

    @abstractmethod
    def prob_observation(
        self,
        observation: Observation,
        agent: Agent,
        state: State,
        prev_joint_action: JointAction,
    ) -> float:
        """
        Given an observation, the current state and the previous joint action, return
        the probability (density) of receiving the observation.
        """
