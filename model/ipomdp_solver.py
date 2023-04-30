# %%

# avoids having to give type annotations as strings
from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, Tuple, Union, List
from numpy.typing import NDArray
from model.ipomdp import sample_init

if TYPE_CHECKING:
    # avoid circular imports with type hints
    from model.model import Civilisation

Action = Tuple[int, Union[str, int]]
AgentAction = Union[str, int, None]
State = NDArray
Observation = NDArray

class Scene(TypedDict):
    state: State
    action: Action
    observation: Observation
    reward: int

class BeliefForest:
    """
    This is a container for all the trees an agent uses to represent its 
    beliefs about the environment and others' beliefs.
    """

    def __init__(self, owner: Civilisation, agents: List[Civilisation]) -> None:
        """
        Keyword arguments:
        owner - the agent that uses the forest to reason
        agents - list of agents in the model
        """
        # create dictionary to hold trees
        self.trees = dict()
        self.owner = owner

        # create agent's own tree at the highest level
        self.trees[(owner,)] = Tree(agent=owner)

        # create trees for agents at lower levels
        self.create_child_trees(parent_tree_level=owner.level,
                                parent_tree_agent=owner,
                                parent_tree_signature=(owner,),
                                agents=agents)

    def create_child_trees(self, parent_tree_level: int, 
                           parent_tree_agent: Civilisation, 
                           parent_tree_signature: Tuple[Civilisation, ...], 
                           agents: List[Civilisation],
                           ) -> None:
        """
        Create trees for the opponents of agent who is at the given level.
        Works recursively.
        """
        # find agents to create trees for
        other_agents = (ag for ag in agents if ag != parent_tree_agent)
        
        # level of all child trees created
        tree_level = parent_tree_level - 1

        for other_agent in other_agents:

            # create tree
            tree_signature = parent_tree_signature + (other_agent,)
            self.trees[tree_signature] = Tree(agent=other_agent)

            # create child trees for this tree if applicable
            if tree_level > 0:
                self.create_child_trees(parent_tree_level=tree_level,
                                        parent_tree_agent=other_agent,
                                        parent_tree_signature=tree_signature,
                                        agents=agents)

class Tree:
    """
    Tree corresponds to a single agent. The tree is the child of another tree
    that corresponds to an agent at the level above.
    """

    def __init__(self, agent: Civilisation) -> None:
        """
        Initialise the tree.
        """
        pass
    