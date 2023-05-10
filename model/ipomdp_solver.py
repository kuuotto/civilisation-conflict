# %%

# avoids having to give type annotations as strings
from __future__ import annotations

from typing import TYPE_CHECKING, Tuple, Union, List, Dict
from numpy.typing import NDArray
from model import ipomdp, growth, action

if TYPE_CHECKING:
    # avoid circular imports with type hints
    from model import civilisation, universe

    AgentAction = Union[action.NO_ACTION, action.HIDE, civilisation.Civilisation]
    Action = Dict[civilisation.Civilisation, AgentAction]
    JointActionHistory = Tuple[Action, ...]
    AgentActionHistory = Tuple[AgentAction, ...]
    TreeSignature = Tuple[civilisation.Civilisation, ...]
    State = NDArray
    Observation = NDArray
    Belief = NDArray


def joint_to_agent_action_history(joint_history: JointActionHistory,
                                  agent: civilisation.Civilisation) -> AgentActionHistory:
    """
    Extracts the agent action history of the specified agent from a joint
    action history.

    Keyword arguments:
    joint_history: joint action history
    agent: agent whose action history to determine
    """
    return tuple(act[agent] if agent in act else action.NO_ACTION 
                 for act in joint_history)

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
        self.trees[(owner,)] = Tree(signature=(owner,), forest=self)

        # create trees for agents at lower levels
        self.create_child_trees(parent_tree_level=owner.level,
                                parent_tree_agent=owner,
                                parent_tree_signature=(owner,))

    def create_child_trees(self, parent_tree_level: int, 
                           parent_tree_agent: civilisation.Civilisation, 
                           parent_tree_signature: TreeSignature) -> None:
        """
        Create trees for the opponents of agent who is at the given level.
        Works recursively.
        """
        model = parent_tree_agent.model

        # find agents to create trees for
        other_agents = (ag for ag in model.agents if ag != parent_tree_agent)
        
        # level of all child trees created
        tree_level = parent_tree_level - 1

        for other_agent in other_agents:

            # create tree
            tree_signature = parent_tree_signature + (other_agent,)
            self.trees[tree_signature] = Tree(signature=tree_signature, 
                                              forest=self)

            # create child trees for this tree if applicable
            if tree_level > 0:
                self.create_child_trees(parent_tree_level=tree_level,
                                        parent_tree_agent=other_agent,
                                        parent_tree_signature=tree_signature)
                
    def plan(self):
        """
        Uses the MCTS (Monte Carlo Tree Search) based algorithm to simulate
        planning by the owner of the forest.

        Trees are expanded from the bottom up, starting at the lowest level.
        """
        for level in range(0, self.owner.level + 1):
            # find trees at this level
            trees = self.get_trees_at_level(level)

            # expand trees
            for tree in trees:
                tree.expand()

    def get_trees_at_level(self, level: int):
        """
        Returns a generator for all trees at a given level, in arbitrary order.
        """
        # length of tree signature at this level
        signature_length = self.owner.level - level + 1

        return (tree for signature, tree in self.trees.items() 
                if len(signature) == signature_length)
    
    def get_tree_level(self, tree: Tree):
        """
        Return the level of the given tree in the forest
        """
        return self.owner.level - len(tree.signature) + 1
    
    def sample_agent_action_history(self, target_tree: Tree):
        """
        Starting from the top-level tree, randomly sample particles and use
        the joint action history to choose a subtree at the level below. 
        This is repeated until the target tree is reached.

        Keyword arguments:
        target_tree: the tree from which we eventually want to choose an agent
                     action history.

        Returns:
        An agent action history from among the root nodes of the target tree.
        """
        model = self.owner.model
        target_signature = target_tree.signature

        # start from the unique agent action history of the top-level tree
        tree_signature = (self.owner,)
        tree = self.trees[tree_signature]
        assert(len(tree.root_nodes) == 1)
        node = next(iter(tree.root_nodes.values()))

        for next_actor in target_signature[1:]:

            if len(node.particles) == 0:
                raise Exception("No particles to sample from")

            # sample particle from current node
            particle = model.rng.choice(node.particles)

            # find agent action history of next actor
            next_actor_action_history = joint_to_agent_action_history(
                particle.joint_action_history, 
                next_actor)
            
            # find next node to sample from
            tree_signature = tree_signature + (next_actor,)
            tree = self.trees[tree_signature]
            node = tree.root_nodes[next_actor_action_history]
            
        return next_actor_action_history

    @property
    def top_level_tree(self):
        return self.trees[(self.owner,)]


class Tree:
    """
    Tree corresponds to a single agent. The tree is the child of another tree
    that corresponds to an agent at the level above (unless it is the top-level
    tree). The tree can contain several subtrees, the root of which each 
    correspond to a possible agent history for the tree agent.
    """

    def __init__(self, signature: TreeSignature, forest: BeliefForest) -> None:
        """
        Initialise the tree.

        If the tree is the top-level tree, the agent's state is used to 
        constrain the initial belief as the agent is certain about its own state.

        Keyword argument:
        signature: the sequence of agents representing the ownership of this tree
        forest: the belief forest that this tree is a part of
        """
        agent = signature[-1]
        model = agent.model
        self.forest = forest
        self.signature = signature
        self.level = forest.get_tree_level(self)

        # check whether tree is the top-level tree in the forest
        top_level = len(signature) == 1

        # create root node corresponding to empty agent action history
        init_belief = ipomdp.sample_init(n_samples=model.n_root_belief_samples,
                                         model=model,
                                         agent=agent if top_level else None)
        self.root_nodes = {(): Node(initial_belief=init_belief, tree=self)}

    def expand(self):
        """
        Expand the current tree multiple times by 
        1. Randomly choosing particles top-down to determine which subtree to
           expand
        2. Sampling a particle from the root belief of the chosen subtree
        3. Using a version of MCTS to traverse the tree, at each step randomly
           selecting who gets to act and choosing others' actions by simulating
           their beliefs at the level below.
        4. When reaching a node that has some untried agent actions, choosing
           one of these and creating a new corresponding tree node
        5. Determining the value of taking that action by performing a random
           rollout until the discounting depth is reached
        6. Propagating this value up the tree to all the new particles created
           (dicounting appropriately)
        """
        model = self.forest.owner.model

        # 1. Choose the root node to start sampling from
        if len(self.root_nodes) > 1:
            agent_action_history = self.forest.sample_agent_action_history(self)
            node = self.root_nodes[agent_action_history]
        else:
            node = next(iter(self.root_nodes.values))

        # 2. Sample a particle from the root belief
        assert(len(node.belief) > 0)
        state = model.rng.choice(node.belief)

        # 3. - 6.
        self.simulate_from(state)

    def simulate_from(self, state: State):
        """
        Simulate decision-making from the current state by
        1. choosing who gets to act
        2. choosing an action from the actors
        3. propagating the state with this action using the transition function
           of the I-POMDP
        4. generating an observation for the tree agent and for each tree agent
           on the level below and weighing each current particle to generate
           a new belief
        5. repeat
        """



class Node:
    """
    A single node in a tree. Corresponds to a specific agent action history.

    A node contains a set of particles. In addition, a node at the root of the 
    tree contains a set of model states which represent the belief of the
    tree agent about the current state of the world.
    """

    def __init__(self, tree: Tree, 
                 initial_belief: Belief = None) -> None:
        """
        Initialise the node.

        Keyword arguments:
        tree: the tree object this node belongs to.
        initial_belief: set of (unweighted) states representing the initial 
                        belief in this tree. Applicable only if this node is a 
                        root node with empty action history
        """
        self.tree = tree
        self.particles = []
        self.belief = initial_belief

    def __repr__(self) -> str:
        if self.belief is None:
            return f'Node({len(self.particles)} particles)'
        else:
            return (f'Node({len(self.particles)} particles, ' +
                    f'{len(self.belief)} belief states)')

class Particle:
    """
    A particle consists of 
    - a model state
    - a joint action history
    - the next agent action (can be empty)
    - a weight to represent belief (can be empty)
    - the value of choosing the next agent action and then continuing optimally
      afterwards (is empty if next agent action is empty)
    """

    def __init__(self, state: State, joint_action_history: JointActionHistory,
                 node: Node, next_agent_action: AgentAction = None, 
                 weight: int = None, value: float = None) -> None:
        self.state = state
        self.joint_action_history = joint_action_history
        self.next_agent_action = next_agent_action # can be None
        self.weight = weight # can be None
        self.value = value # can be None
        self.node = node

    def __repr__(self):
        model = self.node.tree.forest.owner.model
        levels = [round(growth.tech_level(state=agent_state, model=model), 2)
                  for agent_state in self.state]
        return (f"Particle(levels {levels}, " + 
                f"{self.joint_action_history}, {self.next_agent_action}, " +
                f"{self.weight}, {self.value})")
