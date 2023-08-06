# %%

# avoids having to give type annotations as strings
from __future__ import annotations

from typing import TYPE_CHECKING, Tuple, List, Union, Dict, Generator, Any
from numpy.typing import NDArray
from model import ipomdp, growth, action
import numpy as np
from numba import njit

if TYPE_CHECKING:
    # avoid circular imports with type hints
    from model import civilisation, universe

    AgentAction = Union[
        action.NO_TURN, action.NO_ACTION, action.HIDE, civilisation.Civilisation
    ]
    Action = Tuple[civilisation.Civilisation, AgentAction]
    JointActionHistory = Tuple[Action, ...]
    AgentActionHistory = Tuple[AgentAction, ...]
    TreeSignature = Tuple[civilisation.Civilisation, ...]
    State = NDArray
    Observation = Tuple
    Belief = NDArray
    Frame = Dict[str, Any]

WEIGHT_STORAGE_DTYPE = np.float32


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
        act[1] if agent == act[0] else action.NO_TURN for act in joint_action_history
    )


@njit
def calculate_action_qualities(
    belief: NDArray,  # shape (n_states,)
    n_expansions: NDArray,  # shape (n_states,)
    n_expansions_act: NDArray,  # shape (n_states, n_actions)
    act_value: NDArray,  # shape (n_states, n_actions)
    explore: bool,
    exploration_coef: float,
) -> NDArray:  # shape (n_actions,)
    """
    Estimates the values of different actions given the current beliefs.

    Keyword arguments:
    explore: whether to add an exploration term to the action values

    Return value:
    An array of action qualities
    """

    # weighted number of times each action has been expanded
    W_a = belief @ n_expansions_act.astype(np.float_)

    # if there is at least one unexpanded action, return an array with infinity
    # for the quality of the unexpanded action(s), so they get chosen
    if 0 in W_a:
        W_a[W_a == 0] = np.infty
        return W_a

    # total number of expansions for positive-weight particles
    N = (belief > 0).astype(np.float_) @ n_expansions.astype(np.float_)

    # values estimates of different actions
    Q = belief @ act_value / (belief @ (n_expansions_act > 0).astype(np.float_))

    if explore:
        # add exploration bonuses
        N_a = (W_a / W_a.sum()) * N
        Q += exploration_coef * np.sqrt(np.log(N) / N_a)

    return Q


@njit
def calculate_action_probabilities(
    belief: NDArray,  # shape (n_states,)
    n_expansions: NDArray,  # shape (n_states,)
    n_expansions_act: NDArray,  # shape (n_states, n_actions)
    act_value: NDArray,  # shape (n_states, n_actions)
    softargmax_coef: float,
) -> Tuple[NDArray, int]:  # shape (n_actions,)
    """
    Estimates the values of different actions given the current beliefs. Then, returns
    the probabilities for choosing each action according to softargmax.

    Return value:
    A tuple where the first element is an array of probabilities for the different
    actions. The second element is an event code:
    10 - successful
    12 - the belief has diverged (all weights 0 or empty array)
    13 - some action unexpanded
    """

    if belief.sum() == 0:
        # belief has diverged, return the corresponding event code
        return np.zeros(0), 12

    # weighted number of times each action has been expanded
    W_a = belief @ n_expansions_act.astype(np.float_)

    if 0 in W_a:
        # some actions are unexpanded
        return np.zeros(0), 13

    # total number of expansions for positive-weight particles
    N = (belief > 0).astype(np.float_) @ n_expansions.astype(np.float_)

    # values estimates of different actions
    Q = belief @ act_value / (belief @ (n_expansions_act > 0).astype(np.float_))

    # use softargmax to calculate weights of different actions
    action_weights = Q / (softargmax_coef * (1 / np.sqrt(N)))

    # subtract the maximum to keep softargmax from underflowing
    # (this does not change the action probabilities)
    action_weights -= action_weights.max()
    action_weights = np.exp(action_weights)

    # normalise
    action_weights /= action_weights.sum()

    assert not np.isnan(action_weights).any()

    return action_weights, 10


@njit
def systematic_resample(
    weights: NDArray,
    r: float,
    size: int,
) -> NDArray:
    """
    Performs a systematic resampling of the elements using the weights.

    Inspired by code in the filterPy library.

    NOTE: While this is not vectorised, in practice this is very fast, especially with
    Numba.

    Keyword arguments:
    weights: weights of elements in sample
    r: a random number on the interval [0, 1]
    size: desired resample size. If not supplied, resample will be same size as sample

    Returns:
    counts for each element in sample, same length as weights
    """
    counts = np.zeros(len(weights))

    # calculate cumulative weights
    cum_weights = np.cumsum(weights)
    cum_weights[-1] = 1  # make sure last weight is exactly 1

    # determine sample points (split interval [0,1] into “size” intervals and choose
    # points from these intervals with r as the offset from the interval boundary)
    points = (r + np.arange(size)) / size

    # find particles points land on
    point_i, element_i = 0, 0
    while point_i < size:
        if points[point_i] < cum_weights[element_i]:
            # add element to resample
            counts[element_i] += 1

            point_i += 1
        else:
            # move on to next element
            element_i += 1

    return counts


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

        # create trees for agents at lower levels
        if owner.level > 0:
            self.create_child_trees(
                parent_tree_level=owner.level,
                parent_tree_agent=owner,
                parent_tree_signature=(owner,),
            )

        # create agent's own tree at the highest level
        self.trees[(owner,)] = Tree(signature=(owner,), forest=self)

    def create_child_trees(
        self,
        parent_tree_level: int,
        parent_tree_agent: civilisation.Civilisation,
        parent_tree_signature: TreeSignature,
    ) -> None:
        """
        Create trees for the opponents of agent who is at the given level.
        Works recursively. Lowest level trees are created first.
        """
        model = parent_tree_agent.model

        # find agents to create trees for
        other_agents = (ag for ag in model.agents if ag != parent_tree_agent)

        # level of all child trees created
        tree_level = parent_tree_level - 1

        for other_agent in other_agents:
            # signature of the new tree
            tree_signature = parent_tree_signature + (other_agent,)

            # first create child trees for this tree if applicable
            if tree_level > 0:
                self.create_child_trees(
                    parent_tree_level=tree_level,
                    parent_tree_agent=other_agent,
                    parent_tree_signature=tree_signature,
                )

            # create the tree at this level
            self.trees[tree_signature] = Tree(signature=tree_signature, forest=self)

    def plan(self):
        """
        Uses the MCTS (Monte Carlo Tree Search) based algorithm to simulate
        planning by the owner of the forest.

        Trees are expanded from the bottom up, starting at the lowest level.
        """
        model = self.owner.model

        for level in range(0, self.owner.level + 1):
            # find trees at this level
            trees = self.get_trees_at_level(level)

            # expand trees
            for tree in trees:
                if model.debug == 2:
                    print(f"Planning in {tree.signature}")

                # count the number of successful simulations
                n_successful = 0

                for _ in range(model.n_tree_simulations):
                    n_successful += tree.expand()

                if model.debug >= 1:
                    print(
                        f"{n_successful} / {model.n_tree_simulations} successful simulations in {tree}"
                    )

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
        model = self.owner.model
        if model.debug >= 1:
            print(f"{self.owner} observed {owner_observation}")

        ### 1. update the beliefs in the top-level tree of the owner

        # find the old and new root nodes
        old_root_node = self.top_level_tree_root_node
        new_root_node = old_root_node.child_nodes[owner_action]

        # resample the old root node
        old_root_node.resample_particles()

        # weight particles in the top-level tree root node
        new_root_node.weight_particles(owner_observation)

        # check that the weights do not sum to 0
        if new_root_node.belief.sum() == 0:
            raise Exception(
                "The weights of the particles in the top-level tree root node are all 0"
            )

        ### 2. Update belief in the interactive states over the lower-level models
        new_root_node.update_lower_beliefs()

        # remove references to previous particles
        for particle in new_root_node.particles:
            particle.previous_particle = None

        # remove reference to previous node
        new_root_node.parent_node = None

        # set correct node as the new root
        self.top_level_tree_root_node = new_root_node

        if model.debug >= 1:
            print(
                f"{self.top_level_tree.signature}: ({new_root_node.agent_action_history})",
            )

            # report the proportions of particles with different models
            if self.owner.level > 0:
                other_agents = tuple(ag for ag in model.agents if ag != self.owner)
                prob_indifferent = {other_agent: 0 for other_agent in other_agents}
                for particle in new_root_node.particles:
                    for other_agent in other_agents:
                        if (
                            particle.other_agent_frames[other_agent.id]["attack_reward"]
                            == 0
                        ):
                            prob_indifferent[other_agent] += particle.weight
                print(
                    f"In {self.top_level_tree}, {prob_indifferent} of weight is given to indifferent models of the other agents"
                )

        ### 2. update the beliefs in the child trees recursively

        for child_tree in self.child_trees(self.top_level_tree, include_parent=False):
            child_tree.update_beliefs()

    def initialise_simulation(
        self, particle: Particle
    ) -> Dict[civilisation.Civilisation, Node]:
        """
        Initialise the weights in the immediate child trees of the tree particle \
        is in using the weights stored in particle (or its oldest ancestor in \
        case it itself is not in a root node).

        Keyword arguments:
        particle: The particle to use as a base for weighting the particles in the \
                  lower tree.

        Returns:
        A dictionary pointing to the nodes of other agents in the trees one level below.
        These are the nodes in which beliefs were initialised. The keys are the agents.
        """
        particle_tree = particle.node.tree
        model: universe.Universe = self.owner.model

        # if we are already on level 0, there are no child trees to weight
        if particle_tree.level == 0:
            return None

        # find the particle which holds weights for particles in the root nodes of
        # lower level trees. If start_particle is in a root node, then it is the
        # same as ancestor_particle
        ancestor_particle = particle.ancestor_particle()

        # find the nodes corresponding to ancestor particle in the trees of other
        # agents on the level below
        other_agent_nodes = {
            other_agent: self.get_matching_lower_node(
                particle=ancestor_particle, agent=other_agent
            )
            for other_agent in model.agents
            if other_agent != particle_tree.agent
        }

        for other_agent, other_agent_node in other_agent_nodes.items():
            # check that other_agent has weights stored for its tree
            assert ancestor_particle.lower_particle_dist[other_agent.id] is not None

            # extract and store weights
            mask, values = ancestor_particle.lower_particle_dist[other_agent.id]
            other_agent_belief = np.zeros(len(mask))
            other_agent_belief[mask] = values
            other_agent_node.belief = other_agent_belief

        return other_agent_nodes

    def get_trees_at_level(self, level: int) -> Generator[Tree, None, None]:
        """
        Returns a generator for all trees at a given level, in arbitrary order.
        """
        return (tree for tree in self.trees.values() if tree.level == level)

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

    def child_trees(
        self, parent_tree: Tree, include_parent: bool = False
    ) -> Generator[Tree, None, None]:
        """
        Generates all the child trees (trees at exactly one level lower) of the
        given tree.

        Keyword arguments:
        parent_tree: the parent tree of interest
        include_parent: whether to include the parent tree in the generator
        """
        parent_signature = parent_tree.signature
        agents = self.owner.model.agents

        if include_parent:
            yield parent_tree

        if parent_tree.level == 0:
            return

        yield from (
            self.trees[parent_signature + (agent,)]
            for agent in agents
            if agent != parent_tree.agent
        )

    def get_matching_lower_node(
        self, particle: Particle, agent: civilisation.Civilisation
    ) -> Node:
        """
        Given a particle, use the joint action history and frame stored in it
        to find the matching node in the tree of the given agent at the level below.
        """
        tree = particle.node.tree
        assert tree.level > 0

        # find agent action history based on joint action history stored in particle
        agent_action_history = particle.agent_action_history(agent)

        # find child tree
        child_tree_signature = tree.signature + (agent,)
        child_tree = self.trees[child_tree_signature]

        # find node in child tree
        node = child_tree.get_node(
            agent_action_history=agent_action_history,
            frame=particle.ancestor_particle().other_agent_frames[agent.id],
        )

        return node

    def optimal_action(self):
        """
        Return the optimal action of the owner of the forest.
        """
        return self.top_level_tree.tree_policy(
            node=self.top_level_tree_root_node, explore=False, softargmax=False
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
    tree). The tree is actually a collection of multiple trees. The root nodes of
    each of these correspond to a possible agent history and frame (as identified
    by the attack reward) for the tree agent.
    """

    def __init__(self, signature: TreeSignature, forest: BeliefForest) -> None:
        """
        Initialise the tree.

        Keyword argument:
        signature: the sequence of agents representing the ownership of this tree
        forest: the belief forest that this tree is a part of
        """
        self.agent = signature[-1]
        self.forest = forest
        self.signature = signature
        self.level = forest.get_tree_level(self)

        # create a list of possible frames
        model = forest.owner.model
        possible_frames = [{"attack_reward": model.rewards["attack"]}]

        if self.level < forest.owner.level:
            if model.prob_indifferent > 0 and model.rewards["attack"] != 0:
                possible_frames.append({"attack_reward": 0})

        # create root node corresponding to empty agent action history for each possible
        # frame (= attack reward)
        self.root_nodes = []
        for frame in possible_frames:
            root_node = Node(
                tree=self,
                agent_action_history=(),
                frame=frame,
                parent_node=None,
            )
            root_node.create_initial_particles()
            self.root_nodes.append(root_node)

    def get_node(
        self, agent_action_history: AgentActionHistory, frame: Frame
    ) -> Node | None:
        """
        Find node in the tree matching the given agent action history and frame.

        Returns None if node is not found.
        """

        # find the correct root node
        for root_node in self.root_nodes:
            root_length = len(root_node.agent_action_history)

            if (
                agent_action_history[:root_length] == root_node.agent_action_history
                and root_node.frame == frame
            ):
                # found a match
                current_node = root_node
                break
        else:
            # no root node can lead to the desired node
            return None

        # traverse the tree until the desired node is found
        for agent_action in agent_action_history[root_length:]:
            if agent_action not in current_node.child_nodes:
                # could not find the node
                return None

            current_node = current_node.child_nodes[agent_action]

        assert current_node.agent_action_history == agent_action_history
        assert current_node.frame == frame

        return current_node

    def expand(self) -> bool:
        """
        Expand the current tree by
        1. Randomly choosing particles top-down to determine which root node to start
           expanding from. This is also how we find weights for the particles in the
           chosen root node.
        2. Sampling a belief particle from the chosen root node.
        3. Using a version of MCTS to traverse the tree, at each step randomly
           selecting who gets to act and choosing others' actions by simulating
           their beliefs at the level below.
        4. When reaching a node that has some untried agent actions, choosing
           one of these and creating a new corresponding tree node.
        5. Determining the value of taking that action by performing a random
           rollout until the discounting horizon is reached
        6. Propagating this value up the tree to all the new particles created
           (dicounting appropriately)
        (7. Removing the created temporary weights from the particles)

        Returns:
        Whether simulation was successful or skipped because of diverged root beliefs.
        """

        model = self.forest.owner.model

        # 1. Choose the root node to start sampling from

        node = self.forest.top_level_tree_root_node
        particle_weights = node.belief

        for next_agent in self.signature[1:]:
            # choose random particle given weights
            particle: Particle = model.random.choices(
                node.particles, weights=particle_weights
            )[0]

            # find matching node
            node = self.forest.get_matching_lower_node(
                particle=particle, agent=next_agent
            )

            # get weights for node based on the particle
            mask, values = particle.lower_particle_dist[next_agent.id]
            particle_weights = np.zeros(len(mask))
            particle_weights[mask] = values

            # if belief is diverged, we cannot simulate
            if particle_weights.sum() == 0:
                return False

        assert node in self.root_nodes

        # save weights to particles
        node.belief = particle_weights

        # 2. Sample a particle to start simulating from
        start_particle = model.random.choices(node.particles, weights=particle_weights)[
            0
        ]

        # weight particles in the immediate child trees
        other_agent_nodes = self.forest.initialise_simulation(particle=start_particle)

        # 3. - 6. simulate starting from particle
        self.simulate_from(particle=start_particle, other_agent_nodes=other_agent_nodes)

        return True

    def simulate_from(
        self,
        particle: Particle,
        other_agent_nodes: Dict[civilisation.Civilisation, Node] = None,
        do_rollout: bool = False,
        depth: int = 0,
    ) -> float:
        """
        Simulate decision-making from the current particle at node by
        1. choosing who gets to act
        2. choosing the actor's action
        3. propagating the particle with this action using the transition \
           function of the I-POMDP
        4. generating an observation for the tree agent and creating a new belief in \
           the next node
        5. generating an observation for each other agent and updating the beliefs in \
           the trees on the level below
        6. repeat

        Keyword arguments:
        particle: particle to start simulating from
        other_agent_nodes: nodes corresponding to particle in the trees one level below
        do_rollout: whether to stop the simulation process and do a rollout to \
                    determine the value in the leaf node
        depth: how deep into the recursion we are. 0 corresponds to the starting point \
               for the simulation.

        Returns:
        The value of taking the chosen action from particle and then continuing \
        optimally.
        """
        model: universe.Universe = self.forest.owner.model
        node = particle.node

        # don't simulate farther than the discount horizon
        discount_horizon_reached = (
            model.discount_factor**depth < model.discount_epsilon
        )
        if discount_horizon_reached:
            if model.debug == 2:
                print(
                    f"Simulation reached the discount horizon at {node.agent_action_history}"
                )
            do_rollout = True

        # if we used an unexpanded action last time,
        # perform a rollout and end recursion
        if do_rollout:
            if not discount_horizon_reached and model.debug == 2:
                print(f"Simulation reached a leaf node at {node.agent_action_history}")

            # make a copy because rollout changes state in place
            start_state = particle.state.copy()
            value = ipomdp.rollout(
                state=start_state, agent=self.agent, frame=node.frame, model=model
            )

            # add the new particle (it will have no expansions)
            if not discount_horizon_reached:
                node.add_particle(particle)

            # end recursion
            return value

        ### 1. choose actor
        actor: civilisation.Civilisation = model.random.choice(model.agents)

        ### 2. choose an action
        action_unvisited = False

        if actor == self.agent:
            # use tree policy to choose action
            actor_action, action_unvisited = self.tree_policy(node=node, explore=True)

        elif self.level == 0:
            # use the default policy to choose the action of others
            actor_action = ipomdp.level0_opponent_policy(agent=actor, model=model)

        else:
            ### use the tree below to choose the action

            # find matching node
            lower_node = other_agent_nodes[actor]

            # if node is not found, choose with default policy
            if lower_node is None:
                model.add_log_event(event_type=11, event_data=(self.signature,))
                actor_action = ipomdp.level0_opponent_policy(agent=actor, model=model)

            else:
                # determine action from the other agent's tree
                actor_action, _ = lower_node.tree.tree_policy(
                    node=lower_node, explore=False, softargmax=True, simulated_tree=self
                )

        # package action
        action_ = (actor, actor_action)
        agent_action = actor_action if self.agent == actor else action.NO_TURN

        # calculate value of taking action
        agent_action_value = ipomdp.reward(
            state=particle.state,
            action_=action_,
            agent=self.agent,
            model=model,
            attack_reward=node.frame["attack_reward"],
        )

        ### 3. Propagate state
        propagated_state = ipomdp.transition(
            state=particle.state, action_=action_, model=model
        )

        # create a new node if necessary
        if agent_action not in node.child_nodes:
            new_agent_action_history = node.agent_action_history + (agent_action,)
            new_node = Node(
                tree=self,
                agent_action_history=new_agent_action_history,
                parent_node=node,
                frame=node.frame,
            )
            node.child_nodes[agent_action] = new_node

        next_node = node.child_nodes[agent_action]

        # initialise a new particle
        new_joint_action_history = particle.joint_action_history + (action_,)
        next_particle = Particle(
            state=propagated_state,
            joint_action_history=new_joint_action_history,
            node=next_node,
            previous_particle=particle,
        )

        # if particle has already been propagated with this action, add some noise to
        # the state of next_particle
        if particle.has_been_propagated_with(action_):
            assert particle.node in self.root_nodes
            next_particle.add_noise()

        ### 4. Update the belief in the current tree
        # if the action is unvisited, creating weights is not necessary as we perform
        # a rollout from it on the next recursion
        if not action_unvisited:
            # first resample the belief in the current node
            node.resample_particles()

            # generate an observation
            agent_obs = ipomdp.sample_observation(
                state=propagated_state,
                action=action_,
                agent=self.agent,
                model=model,
            )

            # weight particles in the next node
            next_node.weight_particles(agent_obs)

        ### 5. Generate observations for other agents and update beliefs about
        ### interactive states at the level below
        next_other_agent_nodes = None

        if not action_unvisited and self.level > 0:
            # find the nodes in the trees one level below for the next time step
            next_other_agent_nodes = {
                other_agent: None
                if other_agent_node is None
                else other_agent_node.child_nodes.get(
                    actor_action if actor == other_agent else action.NO_TURN
                )
                for other_agent, other_agent_node in other_agent_nodes.items()
            }

            other_agents = (ag for ag in model.agents if ag != self.agent)

            for other_agent in other_agents:
                # get lower node of the other agent which we will want to resample
                lower_node = other_agent_nodes[other_agent]

                if lower_node is None:
                    # couldn't find lower node
                    # model.add_log_event(
                    #     event_type=31, event_data=(self.signature, other_agent)
                    # )
                    continue

                # find node to weight
                next_lower_node = next_other_agent_nodes[other_agent]

                if next_lower_node is None:
                    # there is no node, so we cannot create beliefs
                    # model.add_log_event(
                    #     event_type=32, event_data=(self.signature, other_agent)
                    # )
                    continue

                # resample lower node
                lower_node.resample_particles()

                # generate observation
                other_agent_obs = ipomdp.sample_observation(
                    state=propagated_state,
                    action=action_,
                    agent=other_agent,
                    model=model,
                )

                # assign weights to particles
                next_lower_node.weight_particles(other_agent_obs)

                # log successful creation of lower node belief
                # model.add_log_event(
                #     event_type=30, event_data=(self.signature, other_agent)
                # )

        ### 5. Repeat

        future_value = self.simulate_from(
            particle=next_particle,
            other_agent_nodes=next_other_agent_nodes,
            do_rollout=action_unvisited,
            depth=depth + 1,
        )

        # add particle to node
        # (except if this is the particle where we started simulating)
        if depth > 0:
            node.add_particle(particle)

        # save value and next agent action to particle
        value = agent_action_value + model.discount_factor * future_value
        particle.add_expansion(action=action_, value=value, next_particle=next_particle)

        # clean up beliefs to save memory
        if node != self.forest.top_level_tree_root_node:
            node.belief = None
        else:
            # for the top level tree root node, we only clear the resampling
            # this is necessary from a functional standpoint, not just memory
            node.resample_counts = None

        # clean up beliefs from the lower nodes
        if self.level > 0:
            for lower_node in other_agent_nodes.values():
                if lower_node is None:
                    continue

                lower_node.belief = None

        return value

    def update_beliefs(self):
        """
        Update beliefs in each root node of this tree. This assumes that the parent
        tree has already been updated.
        """
        model = self.forest.owner.model
        parent_tree = self.forest.get_parent_tree(self)

        ### 1. Find new root nodes
        root_nodes = tuple(
            child_node
            for node in self.root_nodes
            for child_node in node.child_nodes.values()
        )
        self.root_nodes: List[Node] = []

        ### 2. Prune new root nodes that are no longer needed (i.e. will never be
        ###    expanded because the agent action histories they represent are no
        ###    longer present in particles on the level above)
        for root_node in root_nodes:
            # check all parent tree root nodes for matching particles
            parent_particles = (
                p
                for parent_root_node in parent_tree.root_nodes
                for p in parent_root_node.particles
            )
            for p in parent_particles:
                if (
                    p.agent_action_history(self.agent) == root_node.agent_action_history
                    and p.other_agent_frames[self.agent.id] == root_node.frame
                ):
                    # found a matching particle, so this root node gets to stay.
                    break

            else:
                # no matching particles, so root node is pruned
                continue

            ### 3. Update beliefs about interactive states at the level below
            root_node.update_lower_beliefs()

            # remove references to previous particles
            for particle in root_node.particles:
                particle.previous_particle = None

            # remove reference to parent node
            root_node.parent_node = None

            # add to set of root nodes
            self.root_nodes.append(root_node)

        if len(self.root_nodes) == 0:
            raise Exception("No root nodes left")

        if model.debug >= 1:
            print(
                f"{self.signature}:",
                len(self.root_nodes),
                sorted(
                    list(
                        (len(n.particles), n.agent_action_history[-1], n.frame)
                        for n in self.root_nodes
                    ),
                    key=lambda x: x[0],
                    reverse=True,
                ),
            )

        # recurse
        for child_tree in self.forest.child_trees(
            parent_tree=self, include_parent=False
        ):
            child_tree.update_beliefs()

    def tree_policy(
        self,
        node: Node,
        explore: bool,
        softargmax: bool = False,
        simulated_tree: Tree = None,
    ) -> Tuple[AgentAction, bool]:
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
        simulated_tree: the tree that is asking self for an action. Only used for
                        logging query success rate.

        Return value:
        A (agent_action, action_unvisited) tuple where action_unvisited
        indicates whether the given action is unvisited under the current
        belief (thus indicating that a rollout policy should be used next).
        """
        model: universe.Universe = node.tree.forest.owner.model

        assert node.tree == self
        assert not (explore and softargmax)

        actions = node.tree.agent.possible_actions()

        if softargmax:
            act_probs, status_code = calculate_action_probabilities(
                belief=node.belief,
                n_expansions=node.n_expansions,
                n_expansions_act=node.n_expansions_act,
                act_value=node.act_value,
                softargmax_coef=model.softargmax_coef,
            )

            # log the successfulness of the event
            model.add_log_event(
                event_type=status_code, event_data=(simulated_tree.signature,)
            )

            if status_code in (12, 13):
                choice = ipomdp.level0_opponent_policy(
                    agent=node.tree.agent, model=model
                )
            else:
                choice = model.random.choices(actions, weights=act_probs)[0]

            return choice, False

        # calculate action qualities
        Q = calculate_action_qualities(
            belief=node.belief,
            n_expansions=node.n_expansions,
            n_expansions_act=node.n_expansions_act,
            act_value=node.act_value,
            explore=explore,
            exploration_coef=model.exploration_coef,
        )

        # find actions with the highest quality and choose one randomly
        # unexpanded actions have a quality np.infty
        max_q = Q.max()
        max_action_i = (Q == max_q).nonzero()[0]
        if len(max_action_i) > 0:
            max_action_i = model.random.choice(max_action_i)

        unexpanded_action = np.isinf(max_q)
        return actions[max_action_i], unexpanded_action

    def __repr__(self) -> str:
        return f"Tree({self.signature}, {len(self.root_nodes)} root nodes)"


class Node:
    """
    A single node in a tree. Corresponds to a specific agent action history and frame.
    A node contains a set of particles.
    """

    def __init__(
        self,
        tree: Tree,
        agent_action_history: AgentActionHistory,
        frame: Frame,
        parent_node: Node,
    ) -> None:
        """
        Initialise the node.

        Keyword arguments:
        tree: the tree object this node belongs to
        agent_action_history: the agent action history this node corresponds to
        frame: the frame to use in this node
        parent_node: the parent node of this node
        """
        # identifiers of node
        self.tree = tree
        self.agent_action_history = agent_action_history
        self.frame = frame

        # relation of node to other nodes in the tree
        self.parent_node = parent_node
        self.child_nodes: Dict[AgentAction, Node] = dict()

        # particles is a list of all particles stored in the node
        self.particles: List[Particle] = []
        # belief contains weights for particles
        self._belief = np.array([])
        # contains the counts of particles if the particles have been resampled
        self.resample_counts = None
        # stores the observation used to generate the belief stored in ‘belief’.
        # Included for debugging purposes.
        self.belief_observation = None

        self.n_particles = 0
        self.array_increase_size = 50  # how much to increase arrays when they fill up
        self.array_size = 5  # initial size of arrays
        self.n_actions = len(tree.agent.possible_actions())
        model: universe.Universe = tree.forest.owner.model

        # these arrays store the information of the n particles:
        # - states
        self._states = np.zeros(
            shape=(self.array_size, model.n_agents, model.agent_state_size)
        )
        # - number of times a particle has been expanded
        self._n_expansions = np.zeros(shape=(self.array_size,), dtype=np.uint16)
        # - number of times a particle has been expanded with each action
        self._n_expansions_act = np.zeros(
            shape=(self.array_size, self.n_actions), dtype=np.uint16
        )
        # - value estimates of each action
        self._act_value = np.zeros(shape=(self.array_size, self.n_actions))
        # - ids of actors in the previous action
        self._prev_action_actor_ids = np.full(
            shape=(self.array_size,), fill_value=-1, dtype=np.int8
        )
        # - ids of targets of attacks in the previous action
        self._prev_action_target_ids = np.full(
            shape=(self.array_size,), fill_value=-1, dtype=np.int8
        )
        # - ids of previous particles
        self._prev_particle_ids = np.full(
            shape=(self.array_size,), fill_value=-1, dtype=np.int32
        )

    @property
    def belief(self) -> NDArray:
        if self.is_resampled:
            # when belief is resampled, the weights of the chosen particles are uniform
            # n_particles = self.resample_counts.sum()
            n_particles = self.n_particles
            return (1 / n_particles) * (self.resample_counts > 0)

        assert len(self._belief) == self.n_particles
        return self._belief

    @belief.setter
    def belief(self, new_belief: NDArray) -> None:
        assert not (new_belief is not None and new_belief.shape[0] != self.n_particles)
        self._belief = new_belief
        self.resample_counts = None
        self.belief_observation = None

    @property
    def states(self) -> NDArray:
        return self._states[: self.n_particles]

    @property
    def n_expansions(self) -> NDArray:
        return self._n_expansions[: self.n_particles]

    @property
    def n_expansions_act(self) -> NDArray:
        return self._n_expansions_act[: self.n_particles]

    @property
    def act_value(self) -> NDArray:
        return self._act_value[: self.n_particles]

    @property
    def prev_action_actor_ids(self) -> NDArray:
        return self._prev_action_actor_ids[: self.n_particles]

    @property
    def prev_action_target_ids(self) -> NDArray:
        return self._prev_action_target_ids[: self.n_particles]

    @property
    def prev_particle_ids(self) -> NDArray:
        return self._prev_particle_ids[: self.n_particles]

    @property
    def is_resampled(self) -> bool:
        return self.resample_counts is not None

    def add_particle(self, particle: Particle) -> None:
        """
        Add particle to node
        """
        assert particle.node == self

        # give particle its index in the arrays
        particle.id = self.n_particles

        # add particle
        self.particles.append(particle)
        self.n_particles += 1

        # increase array sizes if necessary
        if self.n_particles > self.array_size:
            # states
            new_states = np.zeros(
                shape=(self.array_increase_size, *self._states.shape[1:])
            )
            self._states = np.concatenate((self._states, new_states), axis=0)

            # n_expansions
            new_n_expansions = np.zeros(
                shape=(self.array_increase_size,), dtype=np.uint16
            )
            self._n_expansions = np.concatenate(
                (self._n_expansions, new_n_expansions), axis=0
            )

            # n_expansions_act
            new_n_expansions_act = np.zeros(
                shape=(self.array_increase_size, self.n_actions),
                dtype=np.uint16,
            )
            self._n_expansions_act = np.concatenate(
                (self._n_expansions_act, new_n_expansions_act), axis=0
            )

            # act_value
            new_act_value = np.zeros(shape=(self.array_increase_size, self.n_actions))
            self._act_value = np.concatenate((self._act_value, new_act_value), axis=0)

            # prev_action_actor_ids
            new_prev_action_actor_ids = np.full(
                shape=(self.array_increase_size,), fill_value=-1, dtype=np.int8
            )
            self._prev_action_actor_ids = np.concatenate(
                (self._prev_action_actor_ids, new_prev_action_actor_ids), axis=0
            )

            # prev_action_target_ids
            new_prev_action_target_ids = np.full(
                shape=(self.array_increase_size,), fill_value=-1, dtype=np.int8
            )
            self._prev_action_target_ids = np.concatenate(
                (self._prev_action_target_ids, new_prev_action_target_ids), axis=0
            )

            # prev_particle_ids
            new_prev_particle_ids = np.full(
                shape=(self.array_increase_size,), fill_value=-1, dtype=np.int32
            )
            self._prev_particle_ids = np.concatenate(
                (self._prev_particle_ids, new_prev_particle_ids), axis=0
            )

            # increase size of array size variable to match the new array lengths
            self.array_size += self.array_increase_size

        ### store state
        self._states[particle.id] = particle._state
        particle._state = None

        ### previous particle id
        prev_p = particle.previous_particle
        if prev_p is not None and prev_p.id is not None:
            self._prev_particle_ids[particle.id] = prev_p.id

        ### store previous action ids
        if len(particle.joint_action_history) == 0:
            return

        # store information about the previous action
        actor, actor_action = particle.joint_action_history[-1]
        self._prev_action_actor_ids[particle.id] = actor.id

        # if action is an attack, store the id of the target
        if not isinstance(actor_action, int):
            self._prev_action_target_ids[particle.id] = actor_action.id

    def create_initial_particles(self) -> None:
        """
        Create particles corresponding to initial beliefs.
        """
        model = self.tree.forest.owner.model
        n_particles = model.n_root_belief_samples

        # determine if we are in a top or bottom level tree
        in_top_level_tree = self.tree.level == self.tree.forest.owner.level
        in_bottom_level_tree = self.tree.level == 0

        # sample initial states
        # If the tree is the top-level tree, the agent's state is used to
        # constrain the initial belief as the agent is certain about its own state.
        if model.initial_belief == "uniform":
            initial_particle_states = ipomdp.uniform_initial_belief(
                n_samples=n_particles,
                model=model,
                agent=self.tree.agent if in_top_level_tree else None,
            )
        else:
            raise Exception(f"Unrecognised initial belief '{model.initial_belief}'")

        # create particles
        particles = [
            Particle(state=state, joint_action_history=(), node=self)
            for state in initial_particle_states
        ]

        # add particles to the node
        for p in particles:
            self.add_particle(p)

        # assign frames for other agents
        if not in_bottom_level_tree:
            # create a list of possible frames
            possible_frames = [{"attack_reward": model.rewards["attack"]}]
            if model.prob_indifferent > 0 and model.rewards["attack"] != 0:
                possible_frames.append({"attack_reward": 0})

            # we need a frame for every (particle, other agent) combination
            n_frames = len(particles) * (len(model.agents) - 1)

            # randomly sample frames if multiple are possible. This is done independent
            # of the frame of the current node
            if len(possible_frames) == 2:
                frames = model.random.choices(
                    possible_frames,
                    weights=[1 - model.prob_indifferent, model.prob_indifferent],
                    k=n_frames,
                )
            elif len(possible_frames) == 1:
                frames = (possible_frames[0] for _ in range(n_frames))

            # assign the generated frames
            for (p, other_agent), frame in zip(
                (
                    (p, ag)
                    for p in particles
                    for ag in model.agents
                    if ag != self.tree.agent
                ),
                frames,
            ):
                if p.other_agent_frames is None:
                    p.other_agent_frames = [None] * len(model.agents)

                p.other_agent_frames[other_agent.id] = frame

        # if the node is the root node in some other tree than level 0, assign weights
        # to particles in the root nodes on the level below
        if not in_bottom_level_tree:
            for particle in particles:
                for other_agent in (ag for ag in model.agents if ag != self.tree.agent):
                    # create uniform weights
                    particle_weights = np.array(n_particles * (1 / n_particles,))
                    weights_mask = particle_weights > 0
                    weights_values = particle_weights[weights_mask].astype(
                        WEIGHT_STORAGE_DTYPE
                    )

                    # store created belief
                    particle.lower_particle_dist[other_agent.id] = (
                        weights_mask,
                        weights_values,
                    )

        # if the node is in the top level tree, its particles need weights
        if in_top_level_tree:
            self.belief = np.array(n_particles * (1 / n_particles,))

    def __repr__(self) -> str:
        return (
            f"Node({self.agent_action_history}, {self.frame}, "
            + f"{len(self.particles)} particles)"
        )

    def weight_particles(self, observation: Observation) -> None:
        """
        Weights each particle in the node according to how likely they are under the
        given observation.

        Keyword arguments:
        observation: the observation to use for weighting the particles
        """
        if self.n_particles == 0:
            self.belief = np.array([])
            self.belief_observation = observation
            return

        model = self.tree.forest.owner.model

        # find prior weights
        assert -1 not in self.prev_particle_ids
        prior_weights = self.parent_node.belief[self.prev_particle_ids]

        # if the prior is already diverged, no need to calculate further
        if prior_weights.sum() == 0:
            self.belief = prior_weights
            self.belief_observation = observation
            return

        # calculate weights
        weights = ipomdp.prob_observation(
            observation=observation,
            states=self.states,
            prev_action_actor_ids=self.prev_action_actor_ids,
            prev_action_target_ids=self.prev_action_target_ids,
            observer=self.tree.agent,
            model=model,
        )

        # multiply likelihood by prior to get final weights
        weights *= prior_weights

        # normalise
        weight_sum = weights.sum()
        if weight_sum > 0:
            weights /= weight_sum

        # save weights
        self.belief = weights

        # save observation used to create the weights
        self.belief_observation = observation

    def resample_particles(self):
        """
        Resamples the current belief. The resampled belief is accessed as normal
        through the belief property.
        """
        assert not self.is_resampled

        model = self.tree.forest.owner.model

        if self.n_particles == 0 or self.belief.sum() == 0:
            return

        self.resample_counts = systematic_resample(
            weights=self._belief,
            r=model.random.random(),
            size=self.n_particles,
        )

    def update_lower_beliefs(self):
        """
        Creates beliefs over lower-level interactive states in each particle in this
        node by updating prior beliefs.

        Assumes that particles still have a reference to previous particles.
        """
        if self.tree.level == 0:
            return

        forest = self.tree.forest
        model: universe.Universe = forest.owner.model

        other_agents = tuple(ag for ag in model.agents if ag != self.tree.agent)

        for particle in self.particles:
            # store the frames for other agents, these are inherited from the ancestor
            particle.other_agent_frames = particle.previous_particle.other_agent_frames

            # weight the particles in the lower trees using the ancestor of particle
            forest.initialise_simulation(particle=particle)

            for other_agent in other_agents:
                # find particles in the lower tree to assign weights to
                lower_node = forest.get_matching_lower_node(
                    particle=particle, agent=other_agent
                )

                if lower_node is None:
                    # Couldn't find the corresponding node.
                    # This likely means the number of simulations done per tree was too
                    # low.
                    # We will create a new empty node (which will never be simulated)

                    # find parent of new node (it will always exist)
                    lower_node_parent = forest.get_matching_lower_node(
                        particle=particle.previous_particle, agent=other_agent
                    )

                    lower_node_agent_action_history = particle.agent_action_history(
                        other_agent
                    )
                    prev_agent_action = lower_node_agent_action_history[-1]

                    assert prev_agent_action not in lower_node_parent.child_nodes

                    # create new node
                    lower_node = Node(
                        tree=lower_node_parent.tree,
                        agent_action_history=lower_node_agent_action_history,
                        frame=particle.other_agent_frames[other_agent.id],
                        parent_node=lower_node_parent,
                    )
                    lower_node_parent.child_nodes[prev_agent_action] = lower_node

                    # save an empty belief to particle
                    particle.lower_particle_dist[other_agent.id] = np.array(
                        [], dtype=np.bool_
                    ), np.array([], dtype=WEIGHT_STORAGE_DTYPE)

                    # # log event
                    # model.add_log_event(
                    #     event_type=22,
                    #     event_data=(
                    #         self.tree.signature,
                    #         self.agent_action_history,
                    #         lower_node.tree.signature,
                    #         lower_node.agent_action_history,
                    #     ),
                    # )
                    continue

                if lower_node.n_particles == 0:
                    # the target node has no particles so there is nothing to weight
                    particle.lower_particle_dist[other_agent.id] = np.array(
                        [], dtype=np.bool_
                    ), np.array([], dtype=WEIGHT_STORAGE_DTYPE)

                    # # log event
                    # model.add_log_event(
                    #     event_type=23,
                    #     event_data=(
                    #         self.tree.signature,
                    #         self.agent_action_history,
                    #         lower_node.tree.signature,
                    #         lower_node.agent_action_history,
                    #     ),
                    # )
                    continue

                # simulate an observation for the other agent given this particle
                other_agent_obs = ipomdp.sample_observation(
                    state=particle.state,
                    action=particle.joint_action_history[-1],
                    agent=other_agent,
                    model=model,
                )

                # resample particles in the parent node of lower_node
                lower_node.parent_node.resample_particles()

                # find updated weights
                lower_node.weight_particles(observation=other_agent_obs)

                # check if the belief in the lower node has diverged
                # if lower_node.belief.sum() == 0:
                #     # report that belief is diverged
                #     model.add_log_event(
                #         event_type=21,
                #         event_data=(
                #             self.tree.signature,
                #             self.agent_action_history,
                #             lower_node.tree.signature,
                #             lower_node.agent_action_history,
                #         ),
                #     )

                # save weights (in sparse format to save memory)
                belief_mask = lower_node.belief > 0
                belief_values = lower_node.belief[belief_mask].astype(
                    WEIGHT_STORAGE_DTYPE
                )
                particle.lower_particle_dist[other_agent.id] = (
                    belief_mask,
                    belief_values,
                )


class Particle:
    """
    A particle consists of
    - a model state
    - a joint action history
    - a distribution for particles for each agent in the trees one level below (empty
      for particles in a level 0 tree)
    - a frame for each other agent (empty for particles in a level 0 tree)
    - number of times the particle has been expanded with each of the possible actions
    - for each possible action, the value of choosing the next agent action and then
      continuing optimally afterwards
    - a weight to represent belief (changes and can be empty)
    - a reference to the previous particle (is empty for particles in a root node)
    """

    # pre-define names of attributes to reduce memory usage
    __slots__ = (
        "node",
        "previous_particle",
        "id",
        "_state",
        "joint_action_history",
        "lower_particle_dist",
        "other_agent_frames",
        "propagated_actions",
    )

    def __init__(
        self,
        state: State,
        joint_action_history: JointActionHistory,
        node: Node,
        previous_particle: Particle | None = None,
    ) -> None:
        # identifiers of the particle
        self.node = node
        self.previous_particle = previous_particle  # can be None if in root node
        self.id = None  # this is set by the Node when the particle is added to it

        # properties of the particle
        self._state = state
        self.joint_action_history = joint_action_history
        self.other_agent_frames: List[Frame | None] | None = None
        self.lower_particle_dist: List[Tuple[NDArray, NDArray] | None] = [None] * len(
            node.tree.forest.owner.model.agents
        )

        # this keeps track of others' actions that have been used to propagate
        # this particle
        self.propagated_actions: List[Action] = []

    def update_previous_particle_id(self, new_id) -> None:
        self.node._prev_particle_ids[self.id] = new_id

    @property
    def state(self) -> State:
        return self._state if not self.added_to_node else self.node.states[self.id]

    @property
    def weight(self) -> float:
        return self.node.belief[self.id]

    @property
    def added_to_node(self) -> bool:
        return self.id is not None

    @property
    def n_expansions(self) -> State:
        return 0 if not self.added_to_node else self.node.n_expansions[self.id]

    def n_expansions_act(self, agent_action: AgentAction):
        if not self.added_to_node:
            return 0

        agent = self.node.tree.agent
        action_index = agent.possible_actions().index(agent_action)
        return self.node.n_expansions_act[self.id, action_index]

    def act_value(self, agent_action: AgentAction):
        if not self.added_to_node:
            return 0

        agent = self.node.tree.agent
        action_index = agent.possible_actions().index(agent_action)
        return self.node.act_value[self.id, action_index]

    def agent_action_history(
        self, agent: civilisation.Civilisation
    ) -> AgentActionHistory:
        """
        Returns the agent action history of the given agent, extracted from the
        joint action history stored in the particle.
        """
        return joint_to_agent_action_history(self.joint_action_history, agent=agent)

    def has_been_propagated_with(self, action: Action):
        """
        Checks whether this particle has been propagated with the given action before.
        """
        # if action is someone else's, check propagated_actions
        agent = self.node.tree.agent
        actor = action[0]
        if agent != actor:
            return action in self.propagated_actions

        # otherwise check number of times self has been expanded with agent action
        agent_action = action[1]
        return self.n_expansions_act(agent_action) > 0

    def add_expansion(
        self, action: Action, value: float, next_particle: Particle
    ) -> None:
        """
        Add information about an expansion performed starting from the particle.

        Particle should be added to the node before this can be called.

        Keyword arguments:
        action: action used to propagate the particle
        value: value received for taking this action and continuing optimally afterwards
        next_particle: the particle that resulted from taking action from self
        """
        agent = self.node.tree.agent

        assert self.added_to_node

        # tell next particle about our id
        if next_particle.added_to_node:
            next_particle.update_previous_particle_id(self.id)

        # if someone else acted, we only store the action
        actor = action[0]
        if agent != actor:
            if not self.has_been_propagated_with(action):
                self.propagated_actions.append(action)
            return

        # find index of action
        agent_action = action[1]
        action_index = agent.possible_actions().index(agent_action)

        prev_n_expansions = self.n_expansions_act(agent_action)
        prev_value = self.act_value(agent_action)

        # calculate new average value based on previous average and the new value
        new_value = prev_value + (value - prev_value) / (prev_n_expansions + 1)

        # update information
        self.node.n_expansions[self.id] += 1
        self.node.n_expansions_act[self.id, action_index] += 1
        self.node.act_value[self.id, action_index] = new_value

    def add_noise(self):
        model = self.node.tree.forest.owner.model
        state = self.state

        # add noise to growth parameters
        if model.agent_growth == growth.sigmoid_growth:
            speed_range = model.agent_growth_params["speed_range"]
            takeoff_time_range = model.agent_growth_params["takeoff_time_range"]

            speed_noise_scale = model.agent_growth_params["speed_noise_scale"]
            takeoff_time_noise_scale = model.agent_growth_params[
                "takeoff_time_noise_scale"
            ]

            speed_noise_dist = model.agent_growth_params["speed_noise_dist"]

            if speed_range[0] < speed_range[1] and speed_noise_scale > 0:
                if speed_noise_dist == "normal":
                    state[:, 2] += model.rng.normal(
                        loc=0, scale=speed_noise_scale, size=model.n_agents
                    )
                else:
                    raise NotImplementedError(
                        f"{speed_noise_dist} is not a valid speed noise distribution"
                    )

                # make sure values stay within allowed range
                state[:, 2] = state[:, 2].clip(*speed_range)

            if (
                takeoff_time_range[0] < takeoff_time_range[1]
                and takeoff_time_noise_scale > 0
            ):
                state[:, 3] += model.rng.integers(
                    low=-takeoff_time_noise_scale,
                    high=takeoff_time_noise_scale,
                    endpoint=True,
                    size=model.n_agents,
                )

                # make sure values stay within allowed range
                state[:, 3] = state[:, 3].clip(*takeoff_time_range)
        else:
            raise NotImplementedError()

    def ancestor_particle(self):
        """
        Returns the oldest ancestor of the particle, i.e. the particle in one of the
        root nodes where the simulation ending in self started from.
        """
        particle = self

        while particle.previous_particle is not None:
            particle = particle.previous_particle

        return particle

    def __repr__(self) -> str:
        model = self.node.tree.forest.owner.model
        levels = tuple(growth.tech_level(state=self.state, model=model).round(2))
        return (
            f"Particle(levels {levels}, "
            + f"{self.joint_action_history}, {self.n_expansions} expansions, "
            + f"weight {self.weight})"
        )
