# %%

# avoids having to give type annotations as strings
from __future__ import annotations

from typing import TYPE_CHECKING, Tuple, Union, List, Dict
from numpy.typing import NDArray
from model import ipomdp, growth, action
import numpy as np

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

def possible_actions(model: universe.Universe, agent: civilisation.Civilisation):
    """
    Generates all possible agent actions in model for agent.
    """
    yield from (ag for ag in model.agents if ag != agent)
    yield from (act for act in (action.NO_ACTION, action.HIDE))

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
    
    def child_trees(self, tree: Tree, include_parent: bool = False):
        """
        Generates all the child trees (trees at exactly one level lower) of the
        given tree.

        Keyword arguments:
        tree: the parent tree of interest
        include_parent: whether to include the parent tree in the generator
        """
        parent_signature = tree.signature
        agents = self.owner.model.agents

        if tree.level == 0:

            if include_parent:
                yield tree
            else:
                return

        else:
            yield tree
            yield from (parent_signature + (agent,) 
                        for agent in agents 
                        if agent != tree.agent)
    
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
        self.agent = signature[-1]
        self.forest = forest
        self.signature = signature
        self.level = forest.get_tree_level(self)
        model = self.agent.model

        # check whether tree is the top-level tree in the forest
        top_level = len(signature) == 1

        # create root node corresponding to empty agent action history
        init_belief = ipomdp.sample_init(n_samples=model.n_root_belief_samples,
                                         model=model,
                                         agent=self.agent if top_level else None)
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
        self.simulate_from(state=state, node=node)

    def simulate_from(self, state: State, node: Node):
        """
        Simulate decision-making from the current state at node by
        1. choosing who gets to act
        2. choosing the actor's action
        3. propagating the state with this action using the transition function
           of the I-POMDP
        4. generating an observation for the tree agent and for each tree agent
           on the level below and weighing each current particle to generate
           a new belief
        5. repeat
        """
        model = self.forest.owner.model

        ### 1. choose actor (this can be changed to support multiple actors)
        actor = model.rng.choice(model.agents)

        ### 2. choose an action
        if actor == self.agent:
            # use tree policy to choose action
            actor_action = self.tree_policy(node=node, exploration_term=True)
        elif self.level == 0:
            # use the default policy to choose the action of others
            if model.action_dist_0 == 'random':
                actor_action = model.rng.choice(
                    possible_actions(model=model, agent=actor))
            else:
                raise NotImplementedError()
        else:
            # use the tree below to choose the action
            tree = self.forest.trees[self.signature + (actor,)]
            actor_action = tree.tree_policy(node=node, exploration_term=False)

        action_ = {actor: actor_action}

        ### 3. Propagate state
        propagated_state = ipomdp.transition(state=state, 
                                             action=action_, 
                                             model=model)
        
        ### 4. Generate observations for the tree agent and all the other 
        ### agents one level below and weight particles in the next node

        for tree in self.forest.child_trees(tree=self, include_parent=True):

            tree_agent_action = (actor_action if tree.agent == actor 
                                else action.NO_ACTION)
            next_node = node.children[tree_agent_action]
            
            # generate observation
            tree_agent_obs = ipomdp.sample_observation(state=propagated_state,
                                                       action=action_,
                                                       agent=tree.agent,
                                                       n_samples=1)
            
            # assign weights to particles
            for particle in next_node.particles:
                particle.weight = (particle.previous_particle.weight * 
                                   ipomdp.prob_observation(
                                        observation=tree_agent_obs,
                                        state=particle.state,
                                        action=particle.joint_action_history[-1],
                                        agent=tree.agent,
                                        model=model))
                
        ### 5. Repeat
        tree_agent_action = (actor_action if self.agent == actor 
                             else action.NO_ACTION)
        current_node = node.children[tree_agent_action]
        self.simulate_from(state=propagated_state, node=current_node)
        

    def tree_policy(self, node: Node, exploration_term: bool) -> AgentAction:
        """
        Choose an action according to the MCTS (Monte Carlo Tree Search) 
        tree policy.

        Keyword arguments:
        node: node to choose the action in. All particles should have weights.
        """
        model = node.tree.forest.owner.model

        # check that there are particles
        assert(len(node.particles) > 0)

        # check that all particles have weights and actions
        assert(sum(p.weight is None for p in node.particles) == 0)
        assert(sum(p.next_agent_action is None for p in node.particles) == 0)

        # calculate necessary quantities
        N = sum(particle.weight > 0 for particle in node.particles)
        W = {act: sum(p.weight 
                      for p in node.particles 
                      if p.next_agent_action == act)   
             for act in possible_actions(model=model, agent=self.agent)}
        W_tilde = {action: weight*N for action, weight in W.items()}

        # if there are unexpanded actions under this belief, choose one of them
        if 0 in W.values():
            return next(action for action, weight in W.items() if weight == 0)

        # calculate values of different actions
        Q = {act: sum(p.weight * p.value 
                      for p in node.particles 
                      if p.next_agent_action == act) / W[act]
             for act in possible_actions(model=model, agent=self.agent)}
        
        if exploration_term:
            # add exploration bonuses
            Q = {act: (q + model.exploration_coef * 
                            np.sqrt(np.log(N) / W_tilde[act])) 
                for act, q in Q.items()}

        # return the action that maximises Q
        return max(Q, key=Q.get)

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
        self.children = dict()

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
    - the next agent action
    - a weight to represent belief (can be empty)
    - the value of choosing the next agent action and then continuing optimally
      afterwards
    - a reference to the previous particle (is empty for particles in a root node)
    """

    def __init__(self, state: State, joint_action_history: JointActionHistory,
                 node: Node, next_agent_action: AgentAction, value: float,
                 weight: int = None, previous_particle = None) -> None:
        self.state = state
        self.joint_action_history = joint_action_history
        self.node = node
        self.next_agent_action = next_agent_action
        self.value = value
        self.weight = weight # can be None
        self.previous_particle = previous_particle # can be None

    def __repr__(self):
        model = self.node.tree.forest.owner.model
        levels = [round(growth.tech_level(state=agent_state, model=model), 2)
                  for agent_state in self.state]
        return (f"Particle(levels {levels}, " + 
                f"{self.joint_action_history}, {self.next_agent_action}, " +
                f"{self.weight}, {self.value})")
