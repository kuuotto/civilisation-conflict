from typing import Tuple, List, TypeAlias
from enum import Enum
from envs.runner_chaser import grid
from envs.ipomdp import IPOMDP


# define the ids of the two agents
class AgentType(Enum):
    RUNNER = 0
    CHASER = 1


class Agent:
    def __init__(self, type: AgentType, id: int) -> None:
        self.type = type
        self.id = id


# define the possible actions
class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


# observations are empty
Observation: TypeAlias = None

# states contain the current location of the runner and the chaser
State = Tuple[grid.Location, grid.Location]

# joint actions contain an action for the runner and the chaser
JointAction = Tuple[Action, Action]

# rewards are floats, and joint rewards are 2-tuples of floats
Reward: TypeAlias = float
JointReward = Tuple[Reward, Reward]

# whether the agent has reached a terminal state
Terminated: TypeAlias = bool
JointTerminated = Tuple[Terminated, Terminated]

# Problem-based constants
ACTION_REWARD = -1
WIN_REWARD = 100
LOSS_REWARD = -100


class RCPOSG(IPOMDP[Agent, State, Action, Observation]):
    """A Partially Observable Stochastic Game for the Runner-Chaser problem"""

    def __init__(self, map_name: str) -> None:
        """
        Initialises the Runner-Chaser POSG.

        Keyword arguments:
        map_name: the name of the map used. Currently supports only "7x7"
        """
        self._grid = grid.Grid(map_name=map_name)

    def transition(
        self, state: State, joint_action: JointAction
    ) -> Tuple[State, JointReward, JointTerminated]:
        """
        Given an action from both agents taken in a state,
        return a sample of the next state. Also returns rewards for each agent and
        whether they have reached a terminal state.

        Keyword arguments:
        state: the current state of the system
        joint_action: the actions taken by both agents
        """
        # decompose the state into runner and chaser locations
        runner_loc, chaser_loc = state

        # decompose the action into runner and chaser actions
        runner_action, chaser_action = joint_action

        # check that the locations are valid
        if not (
            self._grid.is_valid_location(runner_loc)
            and self._grid.is_valid_location(chaser_loc)
        ):
            raise Exception(f"The starting state {state} is not valid.")

        # Check if the process is already in a terminated state
        if (
            self._grid.is_runner_goal(runner_loc)
            or chaser_loc == runner_loc
            or runner_loc in self._observed_tiles(chaser_loc)
        ):
            return (state, (0, 0), (True, True))

        # find next locations
        new_runner_loc = self._move(runner_loc, runner_action)
        new_chaser_loc = self._move(chaser_loc, chaser_action)
        new_state = (new_runner_loc, new_chaser_loc)

        # rewards; by default both get a (negative) reward ACTION_REWARD
        rewards = (ACTION_REWARD, ACTION_REWARD)
        terminated = (False, False)

        # check if the agents have reached a terminal state. This happens when the
        # runner reaches its goal tile or the chaser observes the runner.
        if self._grid.is_runner_goal(new_runner_loc):
            # runner wins
            rewards = (WIN_REWARD, LOSS_REWARD)
            terminated = (True, True)
        elif new_runner_loc == new_chaser_loc or new_runner_loc in self._observed_tiles(
            new_chaser_loc
        ):
            # chaser wins
            rewards = (LOSS_REWARD, WIN_REWARD)
            terminated = (True, True)

        return (new_state, rewards, terminated)

    def sample_observation(self, state: State, prev_joint_action: JointAction):
        """
        Given the current state and the previous joint action taken, sample a joint
        observation from the distribution of possible observations.

        NOTE: This problem has no observations, so they are empty.
        """
        return None

    def prob_observation(
        self, observation: Observation, state: State, prev_joint_action: JointAction
    ):
        """
        Given an observation, the current state and the previous joint action, return
        the probability (density) of receiving the observation.

        NOTE: This problem has no observations, so they are empty.
        """
        return 1

    def _move(self, location: grid.Location, action: Action) -> grid.Location:
        """
        Returns a new grid location as a result of the given movement (action).
        If the movement results in an invalid location (outside the bounds or inside
        a wall) the location does not change.

        Keyword arguments:
        location: the initial location
        action: the movement
        """
        if action == Action.UP:
            new_location = (location[0] - 1, location[1])
        elif action == Action.DOWN:
            new_location = (location[0] + 1, location[1])
        elif action == Action.LEFT:
            new_location = (location[0], location[1] - 1)
        elif action == Action.RIGHT:
            new_location = (location[0], location[1] + 1)

        # if movement results in an invalid location, there is no movement
        if not self._grid.is_empty(new_location):
            return location

        return new_location

    def _observed_tiles(self, agent_location: grid.Location) -> List[grid.Location]:
        """
        Determines the locations of the tiles that the agent can observe.
        Agents can observe the tiles immediately adjacent to them. There are between two
        and four such locations; tiles outside the boundaries of the grid are not
        observed.

        Keyword arguments:
        agent_location: location of the observer

        Returns:
        A list of the observed locations by the agent.
        """
        return self._grid.adjacent_tiles(location=agent_location)
