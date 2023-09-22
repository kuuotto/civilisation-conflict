from abc import ABC, abstractmethod


class POSG(ABC):
    """
    Defines an abstract base class for POSGs. This means that we define what methods
    a POSG class must implement.
    """

    @abstractmethod
    def transition(self, state, joint_action):
        """
        Given a joint action taken in a state, return a sample from the distribution
        of next states and the rewards for each agent.
        """

    @abstractmethod
    def sample_observation(self, state, prev_joint_action):
        """
        Given the current state and the previous joint action taken, sample a joint
        observation from the distribution of possible observations.
        """

    @abstractmethod
    def prob_observation(self, observation, state, prev_joint_action):
        """
        Given an observation, the current state and the previous joint action, return
        the probability (density) of receiving the observation.
        """
