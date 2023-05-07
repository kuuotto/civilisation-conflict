# %%

# avoids having to give type annotations as strings
from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, Tuple, Union, List, Set
from numpy.typing import NDArray
from model.ipomdp import sample_init

if TYPE_CHECKING:
    # avoid circular imports with type hints
    import model.civilisation as civilisation

    AgentAction = Union[str, civilisation.Civilisation, None]
    Action = Set[Tuple[civilisation.Civilisation, AgentAction]]
    JointActionHistory = Tuple[Action, ...]
    State = NDArray
    Observation = NDArray
    Belief = NDArray

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

    def __init__(self, owner: civilisation.Civilisation, 
                 agents: List[civilisation.Civilisation]) -> None:
        """
        Keyword arguments:
        owner - the agent that uses the forest to reason
        agents - list of agents in the model
        """
        # create dictionary to hold trees
        self.trees = dict()
        self.owner = owner

        # create agent's own tree at the highest level
        self.trees[(owner,)] = Tree(agent=owner, top_level=True)

        # create trees for agents at lower levels
        self.create_child_trees(parent_tree_level=owner.level,
                                parent_tree_agent=owner,
                                parent_tree_signature=(owner,),
                                agents=agents)

    def create_child_trees(self, parent_tree_level: int, 
                           parent_tree_agent: civilisation.Civilisation, 
                           parent_tree_signature: Tuple[civilisation.Civilisation, ...], 
                           agents: List[civilisation.Civilisation],
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
    that corresponds to an agent at the level above (unless it is the top-level
    tree).
    """

    def __init__(self, agent: civilisation.Civilisation, top_level: bool = False) -> None:
        """
        Initialise the tree.

        Keyword argument:
        agent: the agent whose actions this tree simulates
        top_level: if the tree is the top-level tree, the agent's state is used
                   to constrain the initial belief as the agent is certain
                   about its own state.
        """
        model = agent.model

        # create root node corresponding to empty agent action history
        init_belief = sample_init(n_samples=model.n_root_belief_samples,
                                  model=model,
                                  agent=agent if top_level else None)
        self.root_nodes = {(): Node(initial_belief=init_belief)}

class Node:
    """
    A single node in a tree. Corresponds to a specific agent action history.
    """

    def __init__(self, initial_belief: Belief = None) -> None:
        """
        Initialise the node.

        Keyword arguments:
        initial_belief: set of (unweighted) states representing the initial 
                        belief in this tree. Applicable only if this node is a 
                        root node with empty action history
        """
        self.particles = set()

        if initial_belief is not None:
            # this is a new node corresponding to an empty action history

            # create particles representing this belief
            particles = {Particle(state=state, 
                                  joint_action_history=(), 
                                  weight=1) 
                         for state in initial_belief}
            self.particles.update(particles)

class Particle:
    """
    A particle consists of a model state, a joint action history, the next
    agent action (can be empty) and a weight to represent belief (can be empty).
    """

    def __init__(self, state: State, joint_action_history: JointActionHistory,
                 next_agent_action: AgentAction = None, weight: int = None) -> None:
        self.state = state
        self.joint_action_history = joint_action_history
        self.next_agent_action = next_agent_action # can be None
        self.weight = weight # can be None

    def __repr__(self):
        return (f"Particle(ages {self.state[:, 0]}, " + 
                f"{self.joint_action_history}, {self.next_agent_action}, " +
                f"{self.weight})")
