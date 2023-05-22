# %%

# avoids having to give type annotations as strings
from __future__ import annotations

from typing import TYPE_CHECKING, Tuple, List, Union, Dict, Generator, Iterable
from numpy.typing import NDArray
from model import ipomdp, growth, action
import numpy as np

if TYPE_CHECKING:
    # avoid circular imports with type hints
    from model import civilisation, universe

    AgentAction = Union[
        action.NO_TURN, action.NO_ACTION, action.HIDE, civilisation.Civilisation
    ]
    Action = Dict[civilisation.Civilisation, AgentAction]
    JointActionHistory = Tuple[Action, ...]
    AgentActionHistory = Tuple[AgentAction, ...]
    TreeSignature = Tuple[civilisation.Civilisation, ...]
    State = NDArray
    Observation = NDArray
    Belief = NDArray


def joint_to_agent_action_history(
    joint_action_history: JointActionHistory, agent: civilisation.Civilisation
) -> AgentActionHistory:
    """
    Extracts the agent action history of the specified agent from a joint
    action history.

    Keyword arguments:
    joint_history: joint action history
    agent: agent whose action history to determine
    """
    return tuple(
        act[agent] if agent in act else action.NO_TURN for act in joint_action_history
    )


# def sample_joint_action_histories(agent_action_history: AgentActionHistory,
#                                   agent: civilisation.Civilisation,
#                                   model: universe.Universe,
#                                   n_samples: int
#                                   ) -> Generator[JointActionHistory, None, None]:
#     """
#     Return samples of joint action histories where the given agent's actions
#     are unchanged but others' actions are decided based on the level 0
#     default policy.

#     Keyword arguments:
#     agent_action_history: agent action history to base the samples on
#     agent: the agent whose actions to keep intact in the joint action histories
#     model: a Universe
#     n_samples: number of samples to generate
#     """
#     for _ in range(n_samples):

#         sample = []

#         for agent_action in agent_action_history:

#             # don't change the joint action if agent acted
#             if agent_action != action.NO_TURN:
#                 sample.append({agent: agent_action})
#                 continue

#             # choose actor
#             actor = model.rng.choice(model.agents)

#             # choose action
#             actor_action = ipomdp.level0_opponent_policy(agent=actor,
#                                                          model=model)

#             sample.append({actor: actor_action})

#         yield tuple(sample)


def weight_particles(
    particles: Tuple[Particle, ...],
    observation: Observation,
    agent: civilisation.Civilisation,
    model: universe.Universe,
):
    """
    Add weights to each particle according to how likely they are under the
    given observation.

    Keyword arguments:
    particles: particles to weight
    observation: the observation to use for weighting the particles
    agent: the observer
    model: a Universe
    """
    # calculate weights
    weights = tuple(
        p.previous_particle.weight
        * ipomdp.prob_observation(
            observation=observation,
            state=p.state,
            action=p.joint_action_history[-1],
            agent=agent,
            model=model,
        )
        for p in particles
    )

    # normalise
    weight_sum = sum(weights)
    if weight_sum > 0:
        weights = tuple(weight / weight_sum for weight in weights)
    # else:
    #     print("When weighting particles, all got weight 0")

    # save
    for particle, weight in zip(particles, weights):
        particle.weight = weight


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
        self.create_child_trees(
            parent_tree_level=owner.level,
            parent_tree_agent=owner,
            parent_tree_signature=(owner,),
        )

    def create_child_trees(
        self,
        parent_tree_level: int,
        parent_tree_agent: civilisation.Civilisation,
        parent_tree_signature: TreeSignature,
    ) -> None:
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
            self.trees[tree_signature] = Tree(signature=tree_signature, forest=self)

            # create child trees for this tree if applicable
            if tree_level > 0:
                self.create_child_trees(
                    parent_tree_level=tree_level,
                    parent_tree_agent=other_agent,
                    parent_tree_signature=tree_signature,
                )

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
                for _ in range(self.owner.model.n_tree_simulations):
                    tree.expand()

    def update_beliefs(
        self, owner_action: AgentAction, owner_observation: Observation
    ) -> None:
        """
        Updates the beliefs in all trees after the owner takes the given
        action.

        Keyword arguments:
        owner_action: action taken by the owner of the forest
        owner_observation: observation received by the owner
        """

        ### 1. update the beliefs in the top-level tree of the owner

        # set correct node as the new root
        old_root_node = self.top_level_tree_root_node
        new_root_node = old_root_node.child_nodes[owner_action]
        self.top_level_tree_root_node = new_root_node
        new_root_node.root_node_weight = 1

        # create reinvigoration particles
        reinvig_particles = new_root_node.generate_reinvigorated_particles()

        # weight current particles and created particles together
        weight_particles(
            new_root_node.particles + reinvig_particles,
            observation=owner_observation,
            agent=self.owner,
            model=self.owner.model,
        )

        # create a root node belief particle set
        new_root_node.generate_root_belief(
            particles=new_root_node.particles + reinvig_particles
        )

        # check that the weights do not sum to 0
        if sum(p.weight for p in self.top_level_tree_root_node.belief) == 0:
            print(
                "The weights of the particles in the top-level tree root node are all 0"
            )

        # remove references to previous particles
        for particle in new_root_node.particles:
            particle.previous_particle = None

        print(
            f"{self.top_level_tree.signature}:",
            f"({new_root_node.agent_action_history}, {new_root_node.root_node_weight:.3f})",
        )

        ### 2. update the beliefs in the child trees recursively

        for child_tree in self.child_trees(self.top_level_tree, include_parent=False):
            child_tree.update_beliefs()

    def get_trees_at_level(self, level: int) -> Generator[Tree, None, None]:
        """
        Returns a generator for all trees at a given level, in arbitrary order.
        """
        # length of tree signature at this level
        signature_length = self.owner.level - level + 1

        return (
            tree
            for signature, tree in self.trees.items()
            if len(signature) == signature_length
        )

    def get_tree_level(self, tree: Tree) -> int:
        """
        Return the level of the given tree in the forest
        """
        return self.owner.level - len(tree.signature) + 1

    def get_parent_tree(self, tree: Tree) -> Tree:
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
            yield from (
                self.trees[parent_signature + (agent,)]
                for agent in agents
                if agent != parent_tree.agent
            )

    def optimal_action(self):
        """
        Return the optimal action of the owner of the forest.
        """
        return self.top_level_tree.tree_policy(
            node=self.top_level_tree_root_node, explore=False
        )[0]

    @property
    def top_level_tree(self) -> Tree:
        return self.trees[(self.owner,)]

    @property
    def top_level_tree_root_node(self) -> Node:
        return self.top_level_tree.root_nodes[0]

    @top_level_tree_root_node.setter
    def top_level_tree_root_node(self, new_node) -> None:
        self.top_level_tree.root_nodes[0] = new_node


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
        init_belief = ipomdp.sample_init(
            n_samples=model.n_root_belief_samples,
            model=model,
            agent=self.agent if top_level else None,
        )
        self.root_nodes = [
            Node(
                tree=self,
                agent_action_history=(),
                initial_belief=init_belief,
                parent_node=None,
            )
        ]

    def get_node(self, agent_action_history: AgentActionHistory) -> Node:
        """
        Find node in the tree matching the given agent action history.
        """

        # find the correct root node
        for root_node in self.root_nodes:
            root_length = len(root_node.agent_action_history)

            if agent_action_history[:root_length] == root_node.agent_action_history:
                # found a match
                current_node = root_node
                break
        else:
            raise Exception("Node not found")

        # traverse the tree until the desired node is found
        for agent_action in agent_action_history[root_length:]:
            if agent_action not in current_node.child_nodes:
                raise Exception("Node not found")

            current_node = current_node.child_nodes[agent_action]

        assert current_node.agent_action_history == agent_action_history
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
            root_node_weights = tuple(n.root_node_weight for n in self.root_nodes)
            node = model.random.choices(self.root_nodes, weights=root_node_weights)[0]
        else:
            node = self.root_nodes[0]

        # 2. Sample a particle from the root belief
        belief_par = model.random.choices(
            node.belief, weights=tuple(p.weight for p in node.belief)
        )[0]

        # initialise new particle
        particle = Particle(
            state=belief_par.state,
            joint_action_history=belief_par.joint_action_history,
            node=belief_par.node,
            weight=belief_par.weight,
        )

        # 3. - 6. simulate starting from particle
        self.simulate_from(particle=particle)

        # 7. Clear particle weights from self and trees one level below
        # (ignoring root node weights in both cases)
        # for tree in self.forest.child_trees(self, include_parent=True):
        #     for root_node in tree.root_nodes:
        #         root_node.clear_child_particle_weights()

    def simulate_from(
        self, particle: Particle, do_rollout: bool = False, depth: int = 0
    ):
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
        node = particle.node

        # don't simulate farther than the discount horizon
        if model.discount_factor**depth < model.discount_epsilon:
            return 0

        # if we used an unexpanded action last time,
        # perform a rollout and end recursion
        if do_rollout:
            # make a copy because rollout changes state in place
            start_state = particle.state.copy()
            value = ipomdp.rollout(state=start_state, agent=self.agent, model=model)

            # end recursion
            return value

        ### 1. choose actor
        # the weights are used to make sampling of the child nodes in this
        # tree more equal
        # n_actions = len(tuple(possible_actions(model=model, agent=self.agent)))
        # agent_weights = tuple(n_actions / (n_actions + 1)
        #                       if ag == self.agent
        #                       else 1/((len(model.agents)-1)*(n_actions + 1))
        #                       for ag in model.agents)
        # actor = model.rng.choice(model.agents, p=agent_weights)
        actor = model.random.choice(model.agents)

        ### 2. choose an action
        action_unvisited = False

        if actor == self.agent:
            # use tree policy to choose action
            actor_action, action_unvisited = self.tree_policy(node=node, explore=True)

        elif self.level == 0:
            # use the default policy to choose the action of others
            actor_action = ipomdp.level0_opponent_policy(agent=actor, model=model)

        else:
            # use the tree below to choose the action
            child_tree = self.forest.trees[self.signature + (actor,)]

            # find correct node in the child tree
            try:
                child_tree_node = child_tree.get_node(
                    particle.agent_action_history(actor)
                )
            except Exception:
                # if the tree below doesn't have a node corresponding to the
                # agent action history, we have no choice but to choose
                # according to the level 0 default policy
                # print("Other agent acted but had not planned for this")
                actor_action = ipomdp.level0_opponent_policy(agent=actor, model=model)
            else:
                # determine action
                actor_action, _ = child_tree.tree_policy(
                    node=child_tree_node, explore=False, softargmax=True
                )

        # if we don't act and ended up in an unvisited node, do a rollout
        if actor != self.agent:
            # check if we have visited NO_TURN before under this belief
            n_visitations = sum(
                n.weight > 0
                for n in node.particles
                if n.next_agent_action == action.NO_TURN
            )

            if n_visitations == 0:
                action_unvisited = True

        # package action
        action_ = {actor: actor_action}
        agent_action = actor_action if self.agent == actor else action.NO_TURN

        ### 3. Propagate state
        propagated_state = ipomdp.transition(
            state=particle.state, action_=action_, model=model
        )

        # calculate value of taking action
        agent_action_value = ipomdp.reward(
            state=particle.state, action_=action_, agent=self.agent, model=model
        )

        # create a new node if necessary
        if agent_action not in node.child_nodes:
            new_agent_action_history = node.agent_action_history + (agent_action,)
            new_node = Node(
                tree=self,
                agent_action_history=new_agent_action_history,
                parent_node=node,
            )
            node.child_nodes[agent_action] = new_node

        next_node = node.child_nodes[agent_action]

        # partially initialise a new particle
        new_joint_action_history = particle.joint_action_history + (action_,)
        new_particle = Particle(
            state=propagated_state,
            joint_action_history=new_joint_action_history,
            node=next_node,
            previous_particle=particle,
        )

        ### 4. Generate observations for the tree agent and all the other
        ### agents one level below and weight particles in the next nodes

        if not action_unvisited:
            for tree in self.forest.child_trees(parent_tree=self, include_parent=True):
                # find correct node
                try:
                    tree_agent_node = tree.get_node(
                        new_particle.agent_action_history(tree.agent)
                    )
                except Exception:
                    # there is no node, so we cannot create beliefs
                    # print("When creating beliefs, the child tree node was not found")
                    continue

                # generate observation
                tree_agent_obs = ipomdp.sample_observation(
                    state=propagated_state,
                    action=action_,
                    agent=tree.agent,
                    model=model,
                )

                # assign weights to particles
                tree_agent_node.weight_particles(tree_agent_obs)

        ### 5. Repeat

        future_value = self.simulate_from(
            particle=new_particle, do_rollout=action_unvisited, depth=depth + 1
        )

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
        orig_root_nodes = self.root_nodes

        # for root_node in self.root_nodes:
        #     assert(set(root_node.child_nodes.keys()) ==
        #            set(possible_actions(model=model, agent=self.agent)).union({action.NO_TURN}))

        ### 1. Find new root nodes
        root_nodes = list(
            child_node
            for node in self.root_nodes
            for child_node in node.child_nodes.values()
        )
        self.root_nodes = []

        # store calculated weights of particles in parent tree root node(s)
        # for each root node
        compatible_particle_weights = dict()

        ### 2. Determine weights of new root nodes
        for root_node in root_nodes:
            # if root node does not have particles, it has not been deemed
            # very important in the planning phase. Therefore we prune it
            # if len(root_node.particles) == 0:
            #     continue

            compatible_particle_weights[root_node] = dict()

            for parent_root_node in parent_tree.root_nodes:
                weight_sum = sum(p.weight for p in parent_root_node.belief)

                particle_weights = {
                    p: (p.weight / weight_sum) * parent_root_node.root_node_weight
                    for p in parent_root_node.belief
                    if (
                        p.agent_action_history(self.agent)
                        == root_node.agent_action_history
                    )
                }

                compatible_particle_weights[root_node].update(particle_weights)

            root_node_weight = sum(compatible_particle_weights[root_node].values())
            root_node.root_node_weight = root_node_weight

        # prune nodes with 0 weight
        root_nodes = [n for n in root_nodes if n.root_node_weight > 0]

        if len(root_nodes) == 0:
            print("No root nodes left after weighting")

        # temporarily normalise root node weights to get the numbers of
        # reinvigoration particles right. Note that these may not be the
        # final weights, as some root nodes may still be pruned
        root_node_weight_sum = sum(n.root_node_weight for n in root_nodes)
        for n in root_nodes:
            n.root_node_weight /= root_node_weight_sum

        ### 3. Create a belief for new root node
        for root_node in root_nodes:
            # particles matching this root node from parent root nodes
            node_compatible_particles = tuple(
                compatible_particle_weights[root_node].keys()
            )
            node_compatible_particle_weight_sum = sum(
                compatible_particle_weights[root_node].values()
            )
            node_compatible_particle_weights = tuple(
                w / node_compatible_particle_weight_sum
                for w in compatible_particle_weights[root_node].values()
            )

            # create new particles by reinvigorating
            reinvig_particles = root_node.generate_reinvigorated_particles()

            particles_to_weight = root_node.particles + reinvig_particles
            belief = {p: 0 for p in particles_to_weight}
            n_samples = model.n_belief_update_samples

            # choose compatible particles for each sample
            particles = model.random.choices(
                node_compatible_particles,
                weights=node_compatible_particle_weights,
                k=n_samples,
            )

            for particle in particles:
                # simulate an observation for the current tree agent
                obs = ipomdp.sample_observation(
                    state=particle.state,
                    action=particle.joint_action_history[-1],
                    agent=self.agent,
                    model=model,
                )

                # weight particles
                weight_particles(
                    particles=particles_to_weight,
                    observation=obs,
                    agent=self.agent,
                    model=model,
                )
                particle_weights = tuple(p.weight for p in particles_to_weight)

                # if all weights are zero, this means none of the particles
                # are possible representations of the world under this
                # observation. In that case we simply try again
                if sum(particle_weights) == 0:
                    n_samples -= 1
                    continue

                # sample particle
                particle = model.random.choices(
                    particles_to_weight, weights=particle_weights
                )[0]

                # add to the counts
                belief[particle] += 1

            # if there were no successful samples, this means that none of the
            # particles we have are representative of the model state.
            # Therefore we prune the node.
            if n_samples == 0:
                print(
                    "No successful samples, pruning root node",
                    root_node.agent_action_history,
                )
                continue
            else:
                # normalise and save beliefs
                for particle in particles_to_weight:
                    particle.weight = belief[particle] / n_samples

            # generate belief particles
            root_node.generate_root_belief(particles=particles_to_weight)

            # remove references to previous particles
            for particle in root_node.particles:
                particle.previous_particle = None

            weight_sum = sum(p.weight for p in root_node.belief)
            assert round(weight_sum, 4) == 1

            # add finished root node to tree
            self.root_nodes.append(root_node)

        if len(self.root_nodes) == 0:
            print("No root nodes left")

        # if there are too many root nodes, only choose the 10 most probable
        # ones
        # if len(self.root_nodes) > 10:
        #     top_nodes = sorted(self.root_nodes, key=lambda x: x.root_node_weight)[-10:]
        #     self.root_nodes = top_nodes

        # normalise weights of root nodes. Sometimes they do not sum to 1
        # if a new root node is pruned because no observation generated is
        # compatible with any particle in it
        root_node_weight_sum = sum(n.root_node_weight for n in self.root_nodes)
        for n in self.root_nodes:
            n.root_node_weight /= root_node_weight_sum

        # print(f"{self.signature}:",
        #         *(f"({n.agent_action_history}, {n.root_node_weight:.3f})"
        #         for n in self.root_nodes))
        print(
            f"{self.signature}:",
            len(self.root_nodes),
            sorted(
                list(
                    (
                        f"{n.root_node_weight:.2f}",
                        len(n.particles),
                        n.agent_action_history[-1],
                    )
                    for n in self.root_nodes
                ),
                key=lambda x: x[0],
            ),
        )

        # recurse
        for child_tree in self.forest.child_trees(
            parent_tree=self, include_parent=False
        ):
            child_tree.update_beliefs()

    def tree_policy(
        self, node: Node, explore: bool, softargmax: bool = False
    ) -> AgentAction:
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
        softargmax: whether to use a softargmax function to weight actions and
                    choose an action using these weights. This only has
                    an effect if explore is False.

        Return value:
        A (next_action, action_unvisited) tuple, where action_unvisited
        indicates whether the given action is unvisited under the current
        belief (thus indicating that a rollout policy should be used next).
        """
        model = node.tree.forest.owner.model

        # check that all particles have weights and actions
        assert sum(p.weight is None for p in node.particles) == 0
        assert sum(p.next_agent_action is None for p in node.particles) == 0

        # calculate necessary quantities
        actions = self.agent.possible_actions()
        N = sum(
            particle.weight > 0
            for particle in node.particles
            if particle.next_agent_action != action.NO_TURN
        )
        W_a = tuple(
            sum(p.weight for p in node.particles if p.next_agent_action == act)
            for act in actions
        )
        W = sum(W_a)

        if explore and 0 in W_a:
            # if there are unexpanded actions under this belief, sample one
            unexpanded_actions = tuple(
                action for action, weight in zip(actions, W_a) if weight == 0
            )
            return model.random.choice(unexpanded_actions), True

        elif not explore and N == 0:
            # if none of the actions are expanded, choose action according to
            # level 0 default policy
            return (ipomdp.level0_opponent_policy(agent=self.agent, model=model), False)

        elif not explore and 0 in W_a:
            # if some of the actions are not expanded and we are not exploring,
            # ignore these unexpanded actions
            W_a, actions = zip(
                *(
                    (weight, action)
                    for weight, action in zip(W_a, actions)
                    if weight > 0
                )
            )

        assert len(W_a) > 0
        W_tilde = tuple((w_a / W) * N for w_a in W_a)

        # calculate values of different actions
        Q = tuple(
            sum(
                p.weight * p.value
                for p in node.particles
                if p.next_agent_action == action
            )
            / weight
            for action, weight in zip(actions, W_a)
        )

        if explore:
            # add exploration bonuses
            Q = tuple(
                q + model.exploration_coef * np.sqrt(np.log(N) / w_tilde)
                for w_tilde, q in zip(W_tilde, Q)
            )

        if not explore and softargmax:
            # use softargmax to calculate weights of different actions
            action_weights = tuple(np.exp(w_tilde / np.sqrt(N)) for w_tilde in W_tilde)
            action_weight_sum = sum(action_weights)
            action_weights = tuple(w / action_weight_sum for w in action_weights)

            # choose action
            choice = model.random.choices(actions, weights=action_weights)[0]
            return choice, False

        # return the action that maximises Q
        choice = max(zip(Q, actions), key=lambda x: x[0])[1]
        return choice, False


class Node:
    """
    A single node in a tree. Corresponds to a specific agent action history.

    A node contains a set of particles. In addition, a node at the root of the
    tree contains a set of belief particles which represent the belief of the
    tree agent about the current state of the world.

    A root node of a tree can also have a root_node_weight attribute, which
    denotes its likelihood in comparison to other root nodes.
    """

    def __init__(
        self,
        tree: Tree,
        agent_action_history: AgentActionHistory,
        parent_node: Node,
        initial_belief: Belief = None,
    ) -> None:
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
        self.parent_node = parent_node
        self.child_nodes = dict()
        self.belief = None

        if initial_belief is not None:
            self.belief = [
                BeliefParticle(
                    state=state,
                    joint_action_history=(),
                    weight=1 / len(initial_belief),
                    node=self,
                )
                for state in initial_belief
            ]
            self.root_node_weight = 1

    def __repr__(self) -> str:
        if self.belief is None:
            return (
                f"Node({self.agent_action_history}, "
                + f"{len(self.particles)} particles)"
            )
        else:
            return (
                f"Node({self.agent_action_history}, "
                + f"{len(self.particles)} particles, "
                + f"{len(self.belief)} belief particles)"
            )

    def clear_child_particle_weights(self):
        """
        Clears weights of particles from all child nodes of node
        """
        for child_node in self.child_nodes.values():
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
        weight_particles(
            particles=self.particles,
            observation=observation,
            agent=self.tree.agent,
            model=self.tree.forest.owner.model,
        )

    def generate_root_belief(self, particles: Iterable[Particle]):
        """
        Convert given particles into belief particles and save them in the
        root node belief particle set.
        """
        self.belief = [
            BeliefParticle(
                state=p.state,
                joint_action_history=p.joint_action_history,
                weight=p.weight,
                node=self,
            )
            for p in particles
        ]

    def generate_reinvigorated_particles(self) -> List[Particle]:
        """
        Returns reinvigorated particles in proportion to the weight of the
        root node. These particles are not yet weighted.

        A single new particle is created by sampling a particle from the
        node's parent root node belief without weights (so low probability
        particles get more presentation), sampling an action from another agent
        if necessary and propagating that particle with the action.
        """
        model = self.tree.forest.owner.model
        agent = self.tree.agent

        n_samples = round(self.root_node_weight * model.n_reinvigoration_particles)

        # always create at least a given number of particles
        n_samples = max(n_samples, 5)

        if n_samples == 0:
            return []

        # sample initial particles without weights
        particles = model.random.choices(
            tuple(p for p in self.parent_node.belief if p.weight > 0), k=n_samples
        )

        # last action performed by tree agent
        last_agent_action = self.agent_action_history[-1]

        # if the tree agent did not have a turn, choose one other agent to
        # act and choose their action based on the level 0 default policy
        if last_agent_action == action.NO_TURN:
            # choose an actor
            possible_actors = [*model.agents]
            possible_actors.remove(agent)
            actors = model.random.choices(possible_actors, k=n_samples)

            # choose an action
            actor_actions = tuple(
                model.random.choice(actor.possible_actions()) for actor in actors
            )
            actions = (
                {actor: actor_action}
                for actor, actor_action in zip(actors, actor_actions)
            )

        else:
            actions = ({agent: last_agent_action} for _ in range(n_samples))

        new_particles = []

        for particle, action_ in zip(particles, actions):
            # create new particle
            new_state = ipomdp.transition(
                state=particle.state, action_=action_, model=model
            )
            new_joint_action_history = particle.joint_action_history + (action_,)
            new_particle = Particle(
                state=new_state,
                joint_action_history=new_joint_action_history,
                node=self,
                previous_particle=particle,
            )

            # add to list
            new_particles.append(new_particle)

        return new_particles


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

    def __init__(
        self,
        state: State,
        joint_action_history: JointActionHistory,
        node: Node,
        next_agent_action: AgentAction | None = None,
        value: float | None = None,
        weight: int | None = None,
        previous_particle: Particle | None = None,
    ) -> None:
        self.state = state
        self.joint_action_history = joint_action_history
        self.node = node
        self.next_agent_action = next_agent_action
        self.value = value
        self.weight = weight  # can be None
        self.previous_particle = previous_particle  # can be None

    def agent_action_history(
        self, agent: civilisation.Civilisation
    ) -> AgentActionHistory:
        """
        Returns the agent action history of the given agent, extracted from the
        joint action history stored in the particle.
        """
        return joint_to_agent_action_history(self.joint_action_history, agent=agent)

    def __repr__(self) -> str:
        model = self.node.tree.forest.owner.model
        levels = [
            round(growth.tech_level(state=agent_state, model=model), 2)
            for agent_state in self.state
        ]
        return (
            f"Particle(levels {levels}, "
            + f"{self.joint_action_history}, {self.next_agent_action}, "
            + f"weight {self.weight}, value {self.value})"
        )


class BeliefParticle:
    """
    A set of belief particles represents a belief about the model state and the
    actions that have taken place so far.

    Each particle consists of:
    - a model state
    - a joint action history
    - a weight
    """

    def __init__(
        self,
        state: State,
        joint_action_history: JointActionHistory,
        weight: float | None,
        node: Node,
    ) -> None:
        self.state = state
        self.joint_action_history = joint_action_history
        self.weight = weight
        self.node = node

    def agent_action_history(
        self, agent: civilisation.Civilisation
    ) -> AgentActionHistory:
        """
        Returns the agent action history of the given agent, extracted from the
        joint action history stored in the particle.
        """
        return joint_to_agent_action_history(self.joint_action_history, agent=agent)

    def __repr__(self) -> str:
        model = self.node.tree.forest.owner.model
        levels = [
            round(growth.tech_level(state=agent_state, model=model), 2)
            for agent_state in self.state
        ]
        return (
            f"BeliefParticle(levels {levels}, "
            + f"{self.joint_action_history}, "
            + f"weight {self.weight})"
        )
