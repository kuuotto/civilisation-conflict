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

    def __init__(self, owner: civilisation.Civilisation) -> None:
        """
        Keyword arguments:
        owner - the agent that uses the forest to reason
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

                print(f"Expanding {tree.signature}")
                for _ in range(self.owner.model.n_tree_simulations):
                    tree.expand()

    def update_beliefs(self, 
                       owner_action: AgentAction, 
                       owner_observation: Observation) -> None:
        """
        Updates the beliefs in all trees after the owner takes the given
        action.

        Keyword arguments:
        owner_action: action taken by the owner of the forest
        owner_observation: observation received by the owner
        """

        ### 1. update the beliefs in the top-level tree of the owner
        
        # set correct node as the new root
        new_root_node = self.top_level_tree_root_node.children[owner_action]
        self.top_level_tree_root_node = new_root_node
        self.top_level_tree_root_node.root_node_weight = 1

        # remove references to previous particles
        for particle in self.top_level_tree_root_node.particles:
            particle.previous_particle = None

        # weight new root node particles according to their likelihood under
        # the given observation
        self.top_level_tree_root_node.weight_particles(owner_observation)

        # create a root node belief particle set
        self.top_level_tree_root_node.generate_root_belief()

        ### 2. update the beliefs in the child trees recursively

        for child_tree in self.child_trees(self.top_level_tree,
                                           include_parent=False):
            child_tree.update_beliefs()

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
    
    def get_parent_tree(self, tree: Tree):
        """
        Returns the parent of the tree.
        """
        parent_signature = tree.signature[:-1]
        return self.trees[parent_signature]
    
    def child_trees(self, parent_tree: Tree, include_parent: bool = False):
        """
        Generates all the child trees (trees at exactly one level lower) of the
        given tree.

        Keyword arguments:
        parent_tree: the parent tree of interest
        include_parent: whether to include the parent tree in the generator
        """
        parent_signature = parent_tree.signature
        agents = self.owner.model.agents

        if parent_tree.level == 0:

            if include_parent:
                yield parent_tree
            else:
                return

        else:
            if include_parent:
                yield parent_tree
            yield from (self.trees[parent_signature + (agent,)]
                        for agent in agents 
                        if agent != parent_tree.agent)
    
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
        tree = self.top_level_tree
        node = self.top_level_tree_root_node

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

    def optimal_action(self):
        """
        Return the optimal action of the owner of the forest.
        """
        return self.top_level_tree.tree_policy(
            node=self.top_level_tree_root_node, explore=False)[0]

    @property
    def top_level_tree(self) -> Tree:
        return self.trees[(self.owner,)]
    
    @property
    def top_level_tree_root_node(self) -> Node:
        return next(iter(self.top_level_tree.root_nodes.values()))
    
    @top_level_tree_root_node.setter
    def top_level_tree_root_node(self, new_node: Node):
        self.top_level_tree.root_nodes = {new_node.agent_action_history: new_node}


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
        self.root_nodes = {(): Node(tree=self, 
                                    agent_action_history=(), 
                                    initial_belief=init_belief)}

    def get_node(self, agent_action_history: AgentActionHistory) -> Node:
        """
        Find node in the tree matching the given agent action history.
        """

        # find the correct root node
        for root_agent_action_history in self.root_nodes.keys():
            root_length = len(root_agent_action_history)

            if agent_action_history[:root_length] == root_agent_action_history:
                # found a match
                current_node = self.root_nodes[root_agent_action_history]
                break
        else:
            raise Exception("Node not found")
    

        # traverse the tree until the desired node is found
        remaining_history = agent_action_history[root_length:]
        for agent_action in remaining_history:

            if agent_action not in current_node.children:
                raise Exception("Node not found")

            current_node = current_node.children[agent_action]

        return current_node

    def expand(self):
        """
        Expand the current tree by 
        1. Randomly choosing particles top-down to determine which subtree to
           expand
        2. Sampling a belief particle from the root belief of the chosen subtree
        3. Using a version of MCTS to traverse the tree, at each step randomly
           selecting who gets to act and choosing others' actions by simulating
           their beliefs at the level below.
        4. When reaching a node that has some untried agent actions, choosing
           one of these and creating a new corresponding tree node
        5. Determining the value of taking that action by performing a random
           rollout until the discounting horizon is reached
        6. Propagating this value up the tree to all the new particles created
           (dicounting appropriately)
        7. Removing the created temporary weights from the particles
        """
        model = self.forest.owner.model

        # 1. Choose the root node to start sampling from
        if len(self.root_nodes) > 1:
            agent_action_history = self.forest.sample_agent_action_history(self)
            node = self.root_nodes[agent_action_history]
        else:
            node = next(iter(self.root_nodes.values()))

        # 2. Sample a particle from the root belief
        assert(len(node.belief) > 0)
        belief_par = model.rng.choice(node.belief)

        # 3. - 6.

        # initialise new particle
        particle = Particle(state=belief_par.state,
                            joint_action_history=belief_par.joint_action_history,
                            node=belief_par.node,
                            weight=belief_par.weight)
        
        # simulate starting from particle
        self.simulate_from(particle=particle, node=node)
        
        # 7. Clear particle weights from self and trees one level below
        # (ignoring root node weights in both cases)
        for tree in self.forest.child_trees(self, include_parent=True):
            for root_node in tree.root_nodes.values():
                root_node.clear_child_particle_weights()
            
    def simulate_from(self, particle: Particle,
                      node: Node,
                      do_rollout: bool = False):
        """
        Simulate decision-making from the current particle at node by
        1. choosing who gets to act
        2. choosing the actor's action
        3. propagating the particle with this action using the transition 
           function of the I-POMDP
        4. generating an observation for the tree agent and for each tree agent
           on the level below and weighing each current particle to generate
           a new belief
        5. repeat
        """
        model = self.forest.owner.model

        # if we used an unexpanded action last time, 
        # perform a rollout and end recursion
        if do_rollout:
            # make a copy because rollout changes state in place
            start_state = particle.state.copy()
            value, next_agent_action = ipomdp.rollout(state=start_state, 
                                                      agent=self.agent,
                                                      model=model)
            
            # save value and next action to new particle
            particle.value = value
            particle.next_agent_action = next_agent_action
            
            # end recursion
            return value

        ### 1. choose actor
        actor = model.rng.choice(model.agents)

        ### 2. choose an action
        action_unvisited = False

        if actor == self.agent:
            # use tree policy to choose action
            actor_action, action_unvisited = self.tree_policy(
                node=node, explore=True)

        elif self.level == 0:
            # use the default policy to choose the action of others
            actor_action = ipomdp.level0_opponent_policy(agent=actor, 
                                                         model=model)
            
            # if there are no particles to choose from in the tree agent's next
            # node, we need to do a rollout in the next node
            if action.NO_ACTION not in node.children:
                action_unvisited = True

        else:
            # use the tree below to choose the action
            child_tree = self.forest.trees[self.signature + (actor,)]

            # find correct node in the child tree
            child_tree_agent_history = joint_to_agent_action_history(
                joint_history=particle.joint_action_history,
                agent=actor)
            
            try:
                child_tree_node = child_tree.get_node(child_tree_agent_history)
            except Exception:
                # if the tree below doesn't have a node corresponding to the
                # agent action history, we have no choice but to choose 
                # according to the level 0 default policy
                actor_action = ipomdp.level0_opponent_policy(agent=actor,
                                                             model=model)
            else:
                # determine action
                actor_action, _ = child_tree.tree_policy(node=child_tree_node, 
                                                        explore=False)

            # if there are no particles to choose from in the tree agent's next
            # node, we need to do a rollout in the next node
            if action.NO_ACTION not in node.children:
                action_unvisited = True

        # package action
        action_ = {actor: actor_action}
        agent_action = (actor_action if self.agent == actor 
                        else action.NO_ACTION)
        
        ### 3. Propagate state
        propagated_state = ipomdp.transition(state=particle.state, 
                                             action_=action_, 
                                             model=model)
        
        # calculate value of taking action
        agent_action_value = ipomdp.reward(state=particle.state,
                                           action_=action_,
                                           agent=self.agent,
                                           model=model)
        
        # create a new node if necessary
        if agent_action not in node.children:
            assert(action_unvisited)
            new_agent_action_history = node.agent_action_history + (agent_action,)
            new_node = Node(tree=self, 
                            agent_action_history=new_agent_action_history)
            node.children[agent_action] = new_node
        next_node = node.children[agent_action]

        # partially initialise a new particle
        new_joint_action_history = particle.joint_action_history + (action_,)
        new_particle = Particle(state=propagated_state,
                                joint_action_history=new_joint_action_history,
                                node=next_node,
                                previous_particle=particle)

        ### 4. Generate observations for the tree agent and all the other 
        ### agents one level below and weight particles in the next node

        if not action_unvisited:
            for tree in self.forest.child_trees(parent_tree=self, include_parent=True):

                # find correct node
                tree_agent_action_history = joint_to_agent_action_history(
                    new_joint_action_history, 
                    agent=tree.agent)
                
                try:
                    tree_agent_node = tree.get_node(tree_agent_action_history)
                except Exception:
                    # there is no node, so we cannot create beliefs
                    continue
                
                # generate observation
                tree_agent_obs = ipomdp.sample_observation(
                    state=propagated_state,
                    action=action_,
                    agent=tree.agent,
                    model=model,
                    n_samples=1)
                
                # assign weights to particles
                tree_agent_node.weight_particles(tree_agent_obs)
                
        ### 5. Repeat
        
        future_value = self.simulate_from(particle=new_particle, 
                                          node=next_node,
                                          do_rollout=action_unvisited)

        # save value and next agent action to particle
        value = agent_action_value + model.discount_factor * future_value
        particle.value = value
        particle.next_agent_action = agent_action
        
        # add particle to node
        node.particles.append(particle)

        return value
    
    def update_beliefs(self):
        """
        Update the beliefs of this tree using the parent tree.
        """
        model = self.forest.owner.model
        parent_tree = self.forest.get_parent_tree(self)
        parent_root_nodes = tuple(parent_tree.root_nodes.values())
        parent_root_node_weights = tuple(n.root_node_weight 
                                         for n in parent_root_nodes)

        ### 1. Swap in new root nodes
        root_nodes = [child_node 
                      for node in self.root_nodes.values()
                      for child_node in node.children.values()]
        
        # remove references to previous particles
        for root_node in root_nodes:
            for particle in root_node.particles:
                particle.previous_particle = None

        # determine the weight of each node
        # TODO: clean up
        for root_node in root_nodes:
            node_agent_action_history = root_node.agent_action_history

            weight = 0

            for parent_root_node in parent_root_nodes:

                n_matching_particles = 0
                n_particles = len(parent_root_node.particles)

                # count number of particles in parent root node that have an
                # agent action history matching the target node in the current
                # tree
                for par in parent_root_node.particles:

                    par_agent_action_history = joint_to_agent_action_history(
                        joint_history = par.joint_action_history,
                        agent=self.agent)
                    
                    if par_agent_action_history == node_agent_action_history:
                        n_matching_particles += 1
                    
                frac_matching_particles = n_matching_particles / n_particles

                weight += (parent_root_node.root_node_weight * 
                           frac_matching_particles)
                
            root_node.root_node_weight = weight

        # add all nodes that have a positive weight
        self.root_nodes = {node.agent_action_history: node 
                           for node in root_nodes
                           if node.root_node_weight > 0}

        ### 2. Create a belief for each new root node

        for root_node in self.root_nodes.values():

            belief = {p: 0 for p in root_node.particles}

            for _ in range(model.n_belief_update_samples):
                # choose a root node from parent tree
                parent_root_node = model.rng.choice(parent_root_nodes, 
                                                    p=parent_root_node_weights)
                
                # choose a particle from the chosen parent root node
                matching_particles = tuple(
                    p for p in parent_root_node.particles
                    if (p.agent_action_history(self.agent) == 
                        root_node.agent_action_history))
                sum_weights = sum(p.weight for p in matching_particles)
                weights = tuple(p.weight / sum_weights 
                                for p in matching_particles)

                particle = model.rng.choice(matching_particles, p=weights)

                # simulate an observation for the current tree agent
                obs = ipomdp.sample_observation(state=particle.state,
                                                action=particle.joint_action_history[-1],
                                                agent=self.agent,
                                                model=model,
                                                n_samples=1)
                
                # weight particles in current root node
                root_node.weight_particles(obs)

                # sample particle
                particle = model.rng.choice(
                    root_node.particles, 
                    p=tuple(p.weight for p in root_node.particles))
                
                # add to the counts
                belief[particle] += 1

            # normalise and save beliefs
            for particle in root_node.particles:
                particle.weight = belief[particle] / model.n_belief_update_samples

            # generate root belief nodes
            root_node.generate_root_belief()

        # recurse
        for child_tree in self.forest.child_trees(parent_tree=self, 
                                                  include_parent=False):
            child_tree.update_beliefs()

    def tree_policy(self, node: Node, explore: bool) -> AgentAction:
        """
        Choose an action according to the MCTS (Monte Carlo Tree Search) 
        tree policy.

        Keyword arguments:
        node: node to choose the action in. All particles should have weights.
        explore: whether to include the exploration term when choosing the 
                 action to take, and whether to choose an unvisited action if 
                 there is one. This is set to false when this tree is used to 
                 choose an action when expanding the parent tree (at which 
                 point this tree is already fully expanded).

        Return value:
        A (next_action, action_unvisited) tuple, where action_unvisited 
        indicates whether the given action is unvisited and thus a rollout 
        policy should be used.
        """
        model = node.tree.forest.owner.model

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

        if explore and 0 in W.values():
            # if there are unexpanded actions under this belief, choose one
            return (next(action for action, weight in W.items() if weight == 0), 
                    True)
        elif not explore and N == 0:
            # if none of the actions are expanded, choose action according to 
            # level 0 default policy
            return (ipomdp.level0_opponent_policy(agent=self.agent, 
                                                  model=model), False)
        elif not explore and 0 in W.values():
            # if some of the actions are not expanded and we are not exploring,
            # ignore these unexpanded actions
            W = {act: weight for act, weight in W.items() if weight > 0}

        # calculate values of different actions
        Q = {act: sum(p.weight * p.value 
                      for p in node.particles 
                      if p.next_agent_action == act) / total_weight
             for act, total_weight in W.items()}
        
        if explore:
            # add exploration bonuses
            Q = {act: (q + model.exploration_coef * 
                            np.sqrt(np.log(N) / W_tilde[act])) 
                for act, q in Q.items()}

        # return the action that maximises Q
        return max(Q, key=Q.get), False

class Node:
    """
    A single node in a tree. Corresponds to a specific agent action history.

    A node contains a set of particles. In addition, a node at the root of the 
    tree contains a set of belief particles which represent the belief of the
    tree agent about the current state of the world.

    A root node of a tree can also have a root_node_weight attribute, which
    denotes its likelihood in comparison to other root nodes.
    """

    def __init__(self, 
                 tree: Tree, 
                 agent_action_history: AgentActionHistory,
                 initial_belief: Belief = None) -> None:
        """
        Initialise the node.

        Keyword arguments:
        tree: the tree object this node belongs to.
        agent_action_history: the agent action history this node corresponds \
                              to
        initial_belief: set of (unweighted) states representing the initial \
                        belief in this tree. Applicable only if this node is \
                        a root node with empty action history.
        """
        self.tree = tree
        self.agent_action_history = agent_action_history

        self.particles = []
        self.children = dict()
        self.belief = None

        if initial_belief is not None:
            self.belief = [BeliefParticle(state=state, 
                                          joint_action_history=(),
                                          weight=1,
                                          node=self) 
                            for state in initial_belief]
            self.root_node_weight = 1

    def __repr__(self) -> str:
        if self.belief is None:
            return (f'Node({self.agent_action_history}, ' + 
                    f'{len(self.particles)} particles)')
        else:
            return (f'Node({self.agent_action_history}, ' +
                    f'{len(self.particles)} particles, ' +
                    f'{len(self.belief)} belief particles)')
        
    def clear_child_particle_weights(self):
        """
        Clears weights of particles from all child nodes of node
        """
        for child_node in self.children.values():
            # clear weights
            for particle in child_node.particles:
                particle.weight = None

            # recurse
            child_node.clear_child_particle_weights()

    def weight_particles(self, observation: Observation):
        """
        Add weights to each particle in self according to how likely they 
        are under the given observation
        """
        model = self.tree.forest.owner.model
        agent = self.tree.agent

        weights = {p: ipomdp.prob_observation(observation=observation,
                                              state=p.state,
                                              action=p.joint_action_history[-1],
                                              agent=agent,
                                              model=model) 
                   for p in self.particles}
        
        # normalise
        weight_sum = sum(weights.values())
        if weight_sum > 0:
            weights = {p: weight / weight_sum for p, weight in weights.items()}

        # save
        for particle in self.particles:
            particle.weight = weights[particle]

    def generate_root_belief(self):
        """
        Based on the weights of the particles, generate a set of belief
        particles for this node.
        """
        model = self.tree.forest.owner.model
        assert(hasattr(self, 'root_node_weight'))

        # resample
        sample = model.rng.choice(self.particles, 
                                  size=model.n_root_belief_samples,
                                  p=tuple(p.weight for p in self.particles))
        
        # reinvigorate
        n_new_particles = round(self.root_node_weight * 
                                model.n_root_belief_samples)
        reinvig_sample = model.rng.choice(self.particles,
                                          size=n_new_particles,
                                          p=tuple(p.weight 
                                                  for p in self.particles))
        # TODO: figure a good way to reinvigorate particles, currently nothing
        # is done to them

        belief_particles = [BeliefParticle(
                                state=part.state,
                                joint_action_history=part.joint_action_history,
                                weight=part.weight,
                                node=part.node) 
                            for part in list(sample) + list(reinvig_sample)]
        
        # save
        self.belief = belief_particles


class Particle:
    """
    A particle consists of 
    - a model state
    - a joint action history
    - the next agent action
    - a weight to represent belief (changes and can be empty)
    - the value of choosing the next agent action and then continuing optimally
      afterwards
    - a reference to the previous particle (is empty for particles in a root node)
    """

    def __init__(self, 
                 state: State, 
                 joint_action_history: JointActionHistory,
                 node: Node, 
                 next_agent_action: AgentAction | None = None,
                 value: float | None = None,
                 weight: int | None = None, 
                 previous_particle: Particle | None = None
                 ) -> None:
        self.state = state
        self.joint_action_history = joint_action_history
        self.node = node
        self.next_agent_action = next_agent_action
        self.value = value
        self.weight = weight # can be None
        self.previous_particle = previous_particle # can be None

    def agent_action_history(self, agent: civilisation.Civilisation
                             ) -> AgentActionHistory:
        """
        Returns the agent action history of the given agent, extracted from the
        joint action history stored in the particle.
        """
        return joint_to_agent_action_history(self.joint_action_history, 
                                             agent=agent)

    def __repr__(self) -> str:
        model = self.node.tree.forest.owner.model
        levels = [round(growth.tech_level(state=agent_state, model=model), 2)
                  for agent_state in self.state]
        return (f"Particle(levels {levels}, " + 
                f"{self.joint_action_history}, {self.next_agent_action}, " +
                f"weight {self.weight}, value {self.value})")
    
class BeliefParticle:
    """
    A set of belief particles represents a belief about the model state and the
    actions that have taken place so far.

    Each particle consists of:
    - a model state
    - a joint action history
    - a weight
    """

    def __init__(self, 
                 state: State, 
                 joint_action_history: JointActionHistory,
                 weight: float,
                 node: Node) -> None:
        self.state = state
        self.joint_action_history = joint_action_history
        self.weight = weight
        self.node = node

    def __repr__(self) -> str:
        model = self.node.tree.forest.owner.model
        levels = [round(growth.tech_level(state=agent_state, model=model), 2)
                  for agent_state in self.state]
        return (f"BeliefParticle(levels {levels}, " +
                f"{self.joint_action_history}, " + 
                f"weight {self.weight})")
