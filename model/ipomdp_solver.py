# %%

# avoids having to give type annotations as strings
from __future__ import annotations

from typing import TYPE_CHECKING, Tuple, List, Union, Dict, Generator
from numpy.typing import NDArray
from model import ipomdp, growth, action
import numpy as np
from numba import njit

if TYPE_CHECKING:
    # avoid circular imports with type hints
    from model import civilisation, universe
    import random

    AgentAction = Union[
        action.NO_TURN, action.NO_ACTION, action.HIDE, civilisation.Civilisation
    ]
    Action = Dict[civilisation.Civilisation, AgentAction]
    JointActionHistory = Tuple[Action, ...]
    AgentActionHistory = Tuple[AgentAction, ...]
    TreeSignature = Tuple[civilisation.Civilisation, ...]
    State = NDArray
    Observation = Tuple
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


@njit
def calculate_action_qualities(
    belief: NDArray,  # shape (n_states,)
    n_expansions: NDArray,  # shape (n_states,)
    n_expansions_act: NDArray,  # shape (n_states, n_actions)
    act_value: NDArray,  # shape (n_states, n_actions)
    explore: bool,
    softargmax: bool,
    exploration_coef: float = None,
) -> NDArray:  # shape (n_actions,)
    """
    Calculates the next action to take in the tree.

    Keyword arguments:
    explore: whether to add an exploration term to the action values
    softargmax: if True, action probabilities are returned instead of qualities

    Return value:
    Index of the chosen action.
    """

    # calculate necessary quantities
    W_a = belief @ n_expansions_act
    # if there is at least one unexpanded action, return an array with infinity
    # for the quality of the unexpanded action(s), so they get chosen
    if 0 in W_a:
        W_a[W_a == 0] = np.infty
        return W_a

    # N = (belief > 0).dot(n_expansions)
    N = (belief > 0).astype(np.float_) @ n_expansions
    W = W_a.sum()
    N_a = (W_a / W) * N

    if softargmax:
        # use softargmax to calculate weights of different actions
        action_weights = np.exp(N_a / np.sqrt(N))
        action_weights /= action_weights.sum()

        return action_weights

    # calculate values of different actions
    Q = belief @ act_value / (belief @ (n_expansions_act > 0).astype(np.float_))

    if explore:
        # add exploration bonuses
        # ignore possible divide by 0 errors since they are desired; unexpanded actions
        # will get an infinitely big weight
        Q += exploration_coef * np.sqrt(np.log(N) / N_a)

    return Q


@njit
def systematic_resample(
    weights: NDArray,
    r: float,
    size: int,
) -> NDArray:
    """
    Performs a systematic resampling of the elements in the list using the weights.

    Inspired by code in the filterPy library.

    Keyword arguments:
    sample: sample to resample
    weights: weights of elements in sample
    r: a number on the interval [0, 1]
    size: desired resample size. If not supplied, resample will be same size as sample

    Returns:
    counts for each element in sample
    """
    counts = np.zeros(len(weights))

    # calculate cumulative weights
    cum_weights = np.cumsum(weights)
    cum_weights[-1] = 1  # make sure last weight is 1

    # determine sample points (split interval [0,1] into “size” intervals and choose
    # points from these intervals with r as the offset from the interval boundary)
    points = (r + np.arange(size)) / size

    # calculate number of times each element in sample is sampled
    # each row corresponds to one cumulative weight, so we sum over columns to get
    # cumulative counts
    # cum_counts = points <= cum_weights[np.newaxis].transpose()
    # cum_counts = cum_counts.sum(axis=1)

    # # calculate counts from cumulative counts
    # cum_counts[1:] -= cum_counts[:-1]
    # counts = cum_counts

    # # create resample
    # resample = [
    #     element
    #     for element, count in zip(sample, counts, strict=True)
    #     for _ in range(count)
    # ]

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


def bin_resample(
    particles: List[Particle],
    model: universe.Universe,
    size: int = None,
) -> List[Particle]:
    """
    Resample particles so that membership in different bins is respected.

    If the total weights of particles in a bin is W, then the number of particles
    sampled from that bin is either floor(size*W) or floor(size*W) + 1.

    Bins are sampled with systematic resampling, and particles inside the bins are also
    sampled with systematic resampling.

    If size is not provided, the size will be the same as original.
    """
    if size is None:
        size = len(particles)

    resample = []
    particle_bins = bin_particles(particles, model)

    # count sum of weights in each bin
    bin_weights = dict()

    for particle, particle_bin in zip(particles, particle_bins):
        if particle_bin not in bin_weights:
            bin_weights[particle_bin] = 0

        bin_weights[particle_bin] += particle.weight

    bins = tuple(bin_weights.keys())

    # sample bins with weights
    bin_counts = systematic_resample(
        sample=bins,
        weights=tuple(bin_weights.values()),
        size=size,
        rng=model.random,
    )[1]

    # sample the sampled number from within each bin
    for bin, bin_count in zip(bins, bin_counts):
        # find particles belonging to bin
        bin_ps = tuple(p for p, p_bin in zip(particles, particle_bins) if p_bin == bin)

        if bin_count == 0 or len(bin_ps) == 0:
            continue

        # normalise weights of particles in the bin
        bin_particle_weights = tuple(p.weight for p in bin_ps)
        bin_particle_weight_sum = sum(bin_particle_weights)
        bin_particle_weights = tuple(
            w / bin_particle_weight_sum for w in bin_particle_weights
        )

        # sample designated number of particles from this bin
        bin_sample = systematic_resample(
            sample=bin_ps,
            weights=bin_particle_weights,
            size=bin_count,
            rng=model.random,
        )[0]

        resample.extend(bin_sample)

    return resample


def bin_particles(
    particles: List[Particle], model: universe.Universe
) -> Tuple[Tuple[int, ...]]:
    """
    Determines a bin membership for each particle in the list.

    A bin is a tuple (x_1, ..., x_n), where n is the number of agents and x_i is the
    number of other agents that agent i can observe / attack (including itself).
    In total there are n^n bins.
    """

    bins = []
    states = np.stack(tuple(p.state for p in particles), axis=0)
    tech_levels = growth.tech_level(state=states, model=model)

    for particle_tech_levels in tech_levels:
        particle_bin = []

        for agent in model.agents:
            # count number of other agents agent can observe / attack in particle
            count = (
                model._distances_tech_level[agent.id, :]
                < particle_tech_levels[agent.id]
            ).sum()
            particle_bin.append(count)

        bins.append(tuple(particle_bin))

    return tuple(bins)


def generate_random_particles(
    n_particles: int,
    node: Node,
    model: universe.Universe,
    observation: Observation = None,
):
    """
    Generates particles with a random state. If an observation is supplied, all
    generated particles will have a positive likelihood under the observation.
    """

    agent = node.tree.agent
    agent_action_history = node.agent_action_history

    # sample joint action histories (based on agent action history)
    other_agents = tuple(ag for ag in model.agents if ag != agent)
    joint_action_histories = tuple(
        tuple(
            {agent: agent_action}
            if agent_action != action.NO_TURN
            else {
                (actor := model.random.choice(other_agents)): model.random.choice(
                    actor.possible_actions()
                )
            }
            for agent_action in agent_action_history
        )
        for _ in range(n_particles)
    )

    # sample initial states
    states = ipomdp.sample_init(n_samples=n_particles, model=model)

    # propagate all initial states with the generated joint action histories
    for state, joint_action_history in zip(states, joint_action_histories):
        for action_ in joint_action_history:
            ipomdp.transition(state=state, action_=action_, model=model, in_place=True)

    # create particles
    particles = [
        Particle(
            state=state,
            joint_action_history=joint_action_history,
            node=node,
        )
        for state, joint_action_history in zip(states, joint_action_histories)
    ]

    return particles


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

                # n_simulations = model.n_tree_simulations * (
                #     self.owner.level - level + 1
                # )

                for _ in range(model.n_tree_simulations):
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
        model = self.owner.model
        ### 1. update the beliefs in the top-level tree of the owner

        # find the old and new root nodes
        old_root_node = self.top_level_tree_root_node
        new_root_node = old_root_node.child_nodes[owner_action]

        # resample the old root node
        old_root_node.resample_particles()

        # weight particles in the top-level tree root node
        new_root_node.weight_particles(owner_observation)

        # check that the weights do not sum to 0
        if len(self.top_level_tree_root_node.belief) == 0:
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

        ### 2. update the beliefs in the child trees recursively

        for child_tree in self.child_trees(self.top_level_tree, include_parent=False):
            child_tree.update_beliefs()

    def initialise_simulation(self, particle: Particle) -> None:
        """
        Initialise the weights in the immediate child trees of the tree particle \
        is in using the weights stored in particle (or its oldest ancestor in \
        case it itself is not in a root node).

        Keyword arguments:
        particle: The particle to use as a base for weighting the particles in the \
                  lower tree.
        """
        particle_tree = particle.node.tree
        model = self.owner.model

        # if we are already on level 0, there are no child trees to weight
        if particle_tree.level == 0:
            return

        # find the particle which holds weights for particles in the root nodes of
        # lower level trees. If start_particle is in a root node, then it is the
        # same as ancestor_particle
        ancestor_particle = particle.ancestor_particle()

        other_agents = (ag for ag in model.agents if ag != particle_tree.agent)
        for other_agent in other_agents:
            # find matching node
            lower_node = self.get_matching_lower_node(
                particle=ancestor_particle, agent=other_agent
            )

            # extract weights for particles
            lower_particle_weights = ancestor_particle.lower_particle_dist[other_agent]

            # store weights
            lower_node.belief = lower_particle_weights

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
        Given a particle, use the joint action history stored in it to find the
        matching node in the tree of the given agent at the level below.
        """
        tree = particle.node.tree
        assert tree.level > 0

        # find agent action history based on joint action history stored in particle
        agent_action_history = particle.agent_action_history(agent)

        # find child tree
        child_tree_signature = tree.signature + (agent,)
        child_tree = self.trees[child_tree_signature]

        # find node in child tree
        node = child_tree.get_node(agent_action_history)

        return node

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
    tree). The tree is actually a collection of multiple trees. The root nodes of
    each of these correspond to a possible agent history for the tree agent.
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

        # create root node corresponding to empty agent action history
        root_node = Node(
            tree=self,
            agent_action_history=(),
            parent_node=None,
        )
        root_node.create_initial_particles()

        self.root_nodes = [root_node]

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
            raise LookupError("Node not found")

        # traverse the tree until the desired node is found
        for agent_action in agent_action_history[root_length:]:
            if agent_action not in current_node.child_nodes:
                raise LookupError("Node not found")

            current_node = current_node.child_nodes[agent_action]

        assert current_node.agent_action_history == agent_action_history
        return current_node

    def expand(self):
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
        """

        model = self.forest.owner.model

        # 1. Choose the root node to start sampling from

        node = self.forest.top_level_tree_root_node
        particle_weights = node.belief

        for next_agent in self.signature[1:]:
            # choose random particle given weights
            particle = model.random.choices(node.particles, weights=particle_weights)[0]

            # find matching node
            node = self.forest.get_matching_lower_node(
                particle=particle, agent=next_agent
            )

            # get weights for node based on the particle
            particle_weights = particle.lower_particle_dist[next_agent]

        assert node in self.root_nodes

        # save weights to particles
        node.belief = particle_weights

        # 2. Sample a particle to start simulating from
        start_particle = model.random.choices(node.particles, weights=particle_weights)[
            0
        ]

        # weight particles in the immediate child trees
        self.forest.initialise_simulation(particle=start_particle)

        # 3. - 6. simulate starting from particle
        self.simulate_from(particle=start_particle)

    def simulate_from(
        self,
        particle: Particle,
        do_rollout: bool = False,
        depth: int = 0,
        joint_observation_history: Dict[
            civilisation.Civilisation, List[Observation]
        ] = None,
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
        do_rollout: whether to stop the simulation process and do a rollout to \
                    determine the value in the leaf node
        depth: how deep into the recursion we are. 0 corresponds to the starting point \
               for the simulation.
        joint_observation_history: the observations made so far by all agents since \
                                   starting at depth 0. Can be used to restore beliefs \
                                   after backtracking.

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
            value = ipomdp.rollout(state=start_state, agent=self.agent, model=model)

            # add the new particle (it will have no expansions)
            if not discount_horizon_reached:
                node.add_particle(particle)

            # end recursion
            return value

        # create an empty joint observation history
        if joint_observation_history is None:
            joint_observation_history = {ag: [] for ag in model.agents}

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
            # use the tree below to choose the action
            try:
                lower_node = self.forest.get_matching_lower_node(
                    particle=particle, agent=actor
                )

                # check that there are particles
                assert lower_node.belief.sum() > 0

                # check that all actions have been expanded
                W_a = lower_node.belief @ lower_node.n_expansions_act
                assert 0 not in W_a

                # # check that there have been a sufficient number of expansions
                # N = sum(p.n_expansions for p in lower_node.particles if p.weight > 0)
                # assert N > 20

            except LookupError:
                if model.debug == 2:
                    print("Could not find a matching node in child tree")
                actor_action = ipomdp.level0_opponent_policy(agent=actor, model=model)
                model.add_log_event(event_type=11, event_data=self.signature)

            except Exception:
                # TODO
                if lower_node.belief.sum() == 0:
                    if model.debug == 2:
                        print("Belief in lower tree has diverged")
                    model.add_log_event(event_type=12, event_data=self.signature)
                else:
                    if model.debug == 2:
                        print("All actions in lower node have not been expanded")
                    model.add_log_event(event_type=13, event_data=self.signature)

                actor_action = ipomdp.level0_opponent_policy(agent=actor, model=model)
            else:
                # determine action
                actor_action, _ = lower_node.tree.tree_policy(
                    node=lower_node, explore=False, softargmax=True
                )

                model.add_log_event(event_type=10, event_data=self.signature)

        # package action
        action_ = {actor: actor_action}
        agent_action = actor_action if self.agent == actor else action.NO_TURN

        # calculate value of taking action
        agent_action_value = ipomdp.reward(
            state=particle.state, action_=action_, agent=self.agent, model=model
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
        if not action_unvisited and self.level > 0:
            other_agents = (ag for ag in model.agents if ag != self.agent)

            for other_agent in other_agents:
                # resample lower node

                try:
                    lower_node = self.forest.get_matching_lower_node(
                        particle=particle, agent=other_agent
                    )
                except Exception:
                    # couldn't find lower node
                    if model.debug == 2:
                        print(
                            "Could not resample node in child tree (node doesn't exist)"
                        )
                    continue
                else:
                    lower_node.resample_particles()

                # generate observation
                other_agent_obs = ipomdp.sample_observation(
                    state=propagated_state,
                    action=action_,
                    agent=other_agent,
                    model=model,
                )

                # store observation
                joint_observation_history[other_agent].append(other_agent_obs)

                try:
                    next_lower_node = self.forest.get_matching_lower_node(
                        particle=next_particle, agent=other_agent
                    )

                    assert len(next_lower_node.particles) > 0
                except Exception:
                    # there is no node, so we cannot create beliefs
                    if model.debug == 2:
                        print(
                            "Could not create belief in child tree (node doesn't exist)"
                        )
                    continue

                # assign weights to particles
                next_lower_node.weight_particles(other_agent_obs)

        ### 5. Repeat

        future_value = self.simulate_from(
            particle=next_particle,
            do_rollout=action_unvisited,
            depth=depth + 1,
            joint_observation_history=joint_observation_history,
        )

        # add particle to node
        # (except if this is the particle where we started simulating)
        if depth > 0:
            node.add_particle(particle)

        # save value and next agent action to particle
        value = agent_action_value + model.discount_factor * future_value
        particle.add_expansion(action=action_, value=value, next_particle=next_particle)

        return value

    def update_beliefs(self):
        """
        Update the beliefs of this tree using the parent tree.
        """
        model = self.forest.owner.model
        parent_tree = self.forest.get_parent_tree(self)

        ### 1. Find new root nodes
        root_nodes = list(
            child_node
            for node in self.root_nodes
            for child_node in node.child_nodes.values()
        )
        self.root_nodes: List[Node] = []

        ### 2. Prune new root nodes that are no longer needed (will never be expanded
        ###    because the agent action histories they represent are no longer present
        ###    in particles on the level above)
        for root_node in root_nodes:
            # check all parent tree root nodes for matching particles
            parent_particles = (
                p
                for parent_root_node in parent_tree.root_nodes
                for p in parent_root_node.particles
            )
            for p in parent_particles:
                if p.agent_action_history(self.agent) == root_node.agent_action_history:
                    # found a matching particle, so this root node gets to stay.
                    break

            else:
                # no matching particles, so root node is pruned
                continue

            # if root node does not have particles, it has not been deemed
            # very important in the planning phase. It can also not be used for
            # planning anymore. Therefore we prune it.
            if len(root_node.particles) == 0:
                if self.forest.owner.model.debug >= 1:
                    print("Pruning root node with no particles (can cause issues!)")
                continue

            # update beliefs about interactive states at the level below
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
                        (
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
        A (next_action, action_unvisited) tuple where action_unvisited
        indicates whether the given action is unvisited under the current
        belief (thus indicating that a rollout policy should be used next).
        """
        model: universe.Universe = node.tree.forest.owner.model

        assert node.tree == self
        assert not (explore and softargmax)

        actions = node.tree.agent.possible_actions()
        n_actions = len(actions)

        Q = calculate_action_qualities(
            belief=node.belief,
            n_expansions=node.n_expansions,
            n_expansions_act=node.n_expansions_act,
            act_value=node.act_value,
            explore=explore,
            softargmax=softargmax,
            exploration_coef=model.exploration_coef,
        )

        if softargmax:
            # choose action
            choice = model.random.choices(actions, weights=Q)[0]
            return choice, False

        # find actions with the highest quality and choose one randomly
        # unexpanded actions have a quality np.infty
        max_q = Q.max()
        max_action_i = np.arange(n_actions)[Q == max_q]
        if len(max_action_i) > 0:
            max_action_i = model.random.choice(max_action_i)

        unexpanded_action = np.isinf(max_q)
        return actions[max_action_i], unexpanded_action

    def __repr__(self) -> str:
        return f"Tree({self.signature}, {len(self.root_nodes)} root nodes)"


class Node:
    """
    A single node in a tree. Corresponds to a specific agent action history.
    A node contains a set of particles.
    """

    def __init__(
        self,
        tree: Tree,
        agent_action_history: AgentActionHistory,
        parent_node: Node,
    ) -> None:
        """
        Initialise the node.

        Keyword arguments:
        tree: the tree object this node belongs to
        agent_action_history: the agent action history this node corresponds to
        parent_node: the parent node of this node
        """
        # identifiers of node
        self.tree = tree
        self.agent_action_history = agent_action_history

        # relation of node to other nodes in the tree
        self.parent_node = parent_node
        self.child_nodes: Dict[AgentAction, Node] = dict()

        # particles is a list of all particles stored in the node
        self.particles: List[Particle] = []
        # belief contains weights for particles
        self.belief = np.array([])
        # contains the counts of particles if the particles have been resampled
        self.resample_counts = None
        # stores the observation used to generate the belief stored in ‘belief’
        self.belief_observation = None

        self.n_particles = 0
        self.array_increase_size = 100
        self.array_size = self.array_increase_size
        self.n_actions = len(tree.agent.possible_actions())
        model: universe.Universe = tree.forest.owner.model

        # these arrays store the information of the n particles:
        # - states
        self._states = np.zeros(
            shape=(self.array_size, model.n_agents, model.agent_state_size)
        )
        # - number of times a particle has been expanded
        self._n_expansions = np.zeros(shape=(self.array_size,))
        # - number of times a particle has been expanded with each action
        self._n_expansions_act = np.zeros(shape=(self.array_size, self.n_actions))
        # - value estimates of each action
        self._act_value = np.zeros(shape=(self.array_size, self.n_actions))
        # - ids of actors in the previous action
        self._prev_action_actor_ids = np.full(
            shape=(self.array_size,), fill_value=-1, dtype=int
        )
        # - ids of targets of attacks in the previous action
        self._prev_action_target_ids = np.full(
            shape=(self.array_size,), fill_value=-1, dtype=int
        )
        # - ids of previous particles
        self._prev_particle_ids = np.full(
            shape=(self.array_size,), fill_value=-1, dtype=int
        )

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
            new_n_expansions = np.zeros(shape=(self.array_increase_size,))
            self._n_expansions = np.concatenate(
                (self._n_expansions, new_n_expansions), axis=0
            )

            # n_expansions_act
            new_n_expansions_act = np.zeros(
                shape=(self.array_increase_size, self.n_actions)
            )
            self._n_expansions_act = np.concatenate(
                (self._n_expansions_act, new_n_expansions_act), axis=0
            )

            # act_value
            new_act_value = np.zeros(shape=(self.array_increase_size, self.n_actions))
            self._act_value = np.concatenate((self._act_value, new_act_value), axis=0)

            # prev_action_actor_ids
            new_prev_action_actor_ids = np.full(
                shape=(self.array_increase_size,), fill_value=-1, dtype=int
            )
            self._prev_action_actor_ids = np.concatenate(
                (self._prev_action_actor_ids, new_prev_action_actor_ids), axis=0
            )

            # prev_action_target_ids
            new_prev_action_target_ids = np.full(
                shape=(self.array_increase_size,), fill_value=-1, dtype=int
            )
            self._prev_action_target_ids = np.concatenate(
                (self._prev_action_target_ids, new_prev_action_target_ids), axis=0
            )

            # prev_particle_ids
            new_prev_particle_ids = np.full(
                shape=(self.array_increase_size,), fill_value=-1, dtype=int
            )
            self._prev_particle_ids = np.concatenate(
                (self._prev_particle_ids, new_prev_particle_ids), axis=0
            )

            # increase size of array size variable to match the new array lengths
            self.array_size += self.array_increase_size

        # store state
        self._states[particle.id] = particle._state
        particle._state = None

        # previous particle id
        prev_p = particle.previous_particle
        if prev_p is not None and prev_p.id is not None:
            self._prev_particle_ids[particle.id] = prev_p.id

        # store previous action ids
        if len(particle.joint_action_history) == 0:
            return

        previous_action = particle.joint_action_history[-1]
        actor, actor_action = next(iter(previous_action.items()))
        self._prev_action_actor_ids[particle.id] = actor.id
        if not isinstance(actor_action, int):
            self._prev_action_target_ids[particle.id] = actor_action.id

    def create_initial_particles(self) -> None:
        """
        Create particles corresponding to initial beliefs.
        """
        model = self.tree.forest.owner.model

        # determine if we are in a top or bottom level tree
        in_top_level_tree = self.tree.level == self.tree.forest.owner.level
        in_bottom_level_tree = self.tree.level == 0

        # sample initial states
        # If the tree is the top-level tree, the agent's state is used to
        # constrain the initial belief as the agent is certain about its own state.
        initial_particle_states = ipomdp.sample_init(
            n_samples=model.n_root_belief_samples,
            model=model,
            agent=self.tree.agent if in_top_level_tree else None,  # TODO
        )

        # create particles
        particles = [
            Particle(state=state, joint_action_history=(), node=self)
            for state in initial_particle_states
        ]
        for p in particles:
            self.add_particle(p)

        # if the node is the root node in some other tree than level 0, assign weights
        # to particles in the root nodes on the level below
        if not in_bottom_level_tree:
            for child_tree in self.tree.forest.child_trees(parent_tree=self.tree):
                # find root node in the child tree
                assert len(child_tree.root_nodes) == 1
                child_tree_root_node = child_tree.root_nodes[0]

                # find number of particles
                n_particles = len(child_tree_root_node.particles)
                assert n_particles == model.n_root_belief_samples

                # assign uniform weights
                particle_weights = np.array(n_particles * (1 / n_particles,))
                for particle in self.particles:
                    particle.lower_particle_dist[child_tree.agent] = particle_weights

        # if the node is in the top level tree, its particles need weights
        if in_top_level_tree:
            self.belief = np.array(
                model.n_root_belief_samples * (1 / model.n_root_belief_samples,)
            )

    def __repr__(self) -> str:
        return (
            f"Node({self.agent_action_history}, " + f"{len(self.particles)} particles)"
        )

    # def clear_child_particle_weights(self):
    #     """
    #     Clears weights of particles from all child nodes of node
    #     """
    #     for child_node in self.child_nodes.values():
    #         # clear weights
    #         for particle in child_node.particles:
    #             particle.weight = None

    #         # recurse
    #         child_node.clear_child_particle_weights()

    def weight_particles(
        self,
        observation: Observation,
    ) -> Tuple[float, ...]:
        """
        Weights each particle in the node according to how likely they are under the
        given observation.

        Keyword arguments:
        observation: the observation to use for weighting the particles
        """
        if self.n_particles == 0:
            return

        model = self.tree.forest.owner.model

        # find prior weights
        prior_weights = self.parent_node.belief[self.prev_particle_ids]

        # assert -1 not in self.prev_particle_ids
        # prior_weights_og = np.array(
        #     tuple(p.previous_particle.weight for p in self.particles)
        # )
        # assert np.allclose(prior_weights_og, prior_weights)

        # calculate weights
        weights = (
            ipomdp.prob_observation(
                observation=observation,
                states=self.states,
                prev_action_actor_ids=self.prev_action_actor_ids,
                prev_action_target_ids=self.prev_action_target_ids,
                observer=self.tree.agent,
                model=model,
            )
            * prior_weights
        )

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
        Resamples the current belief and replaces it in the belief attribute.
        """

        model = self.tree.forest.owner.model
        # TODO: Maybe better to use true number of particles instead?
        n_particles = (self.belief > 0).sum()

        if n_particles == 0:
            return

        self.resample_counts = systematic_resample(
            weights=self.belief,
            r=model.random.random(),
            size=n_particles,
        )

        # reset weights of particles
        self.belief = np.where(self.resample_counts > 0, 1 / n_particles, 0)

    def update_lower_beliefs(self):
        """
        Creates beliefs over lower-level interactive states in each particle in this
        node by updating prior beliefs.

        Assumes that particles still have a reference to previous particles.
        """
        if self.tree.level == 0:
            return

        forest = self.tree.forest
        model = forest.owner.model

        other_agents = (ag for ag in model.agents if ag != self.tree.agent)

        for other_agent in other_agents:
            for particle in self.particles:
                # find particles in the lower tree to assign weights to
                try:
                    lower_node = forest.get_matching_lower_node(
                        particle=particle, agent=other_agent
                    )
                except Exception:
                    # couldn't find the corresponding node. This should never happen
                    raise Exception(
                        f"Couldn't find a lower level node based on particle {particle.joint_action_history} in tree {self.tree.signature}"
                    )

                # simulate an observation for the other agent given this particle
                other_agent_obs = ipomdp.sample_observation(
                    state=particle.state,
                    action=particle.joint_action_history[-1],
                    agent=other_agent,
                    model=model,
                )

                # weight the particles in the parent node of lower_node
                forest.initialise_simulation(particle=particle)

                # resample particles in the parent node of lower_node
                lower_node.parent_node.resample_particles()

                # update
                lower_node.weight_particles(
                    observation=other_agent_obs,
                )

                # save weights
                particle.lower_particle_dist[other_agent] = lower_node.belief


class Particle:
    """
    A particle consists of
    - a model state
    - a joint action history
    - a distribution for particles for each agent in the trees one level below (empty
      for particles in a level 0 tree)
    - number of times the particle has been expanded with each of the possible actions
    - for each possible action, the value of choosing the next agent action and then
      continuing optimally afterwards
    - a weight to represent belief (changes and can be empty)
    - a reference to the previous particle (is empty for particles in a root node)
    """

    def __init__(
        self,
        state: State,
        joint_action_history: JointActionHistory,
        node: Node,
        previous_particle: Particle | None = None,
    ) -> None:
        # identifiers of the particle
        self.node = node
        self.previous_particle = previous_particle  # can be None
        self.id = None  # this is set by the Node when the particle is added to it

        # properties of the particle
        self._state = state
        self.joint_action_history = joint_action_history
        self.lower_particle_dist = dict()
        # self.weight = weight  # can be None

        # initialise dictionaries tracking number of times expanded and average values
        # agent = node.tree.agent
        # possible_actions = agent.possible_actions()
        # self._n_expansions = 0
        # self._n_expansions_act = {act: 0 for act in possible_actions}
        # self._act_value = {act: 0 for act in possible_actions}

        # this keeps track of others' actions that have been used to propagate
        # this particle
        self.propagated_actions: List[Action] = []

    # @property
    # def id(self) -> int:
    #     return self._id

    # @id.setter
    # def id(self, new_id) -> None:
    #     self._id = new_id

    #     # notify next particles of new id
    #     for particle in self.next_particles:
    #         particle.update_previous_particle_id(new_id)

    def update_previous_particle_id(self, new_id) -> None:
        self.node._prev_particle_ids[self.id] = new_id

    @property
    def state(self) -> State:
        return self._state if not self.added_to_node else self.node.states[self.id]

    @property
    def weight(self) -> float:
        # assert self.added_to_node
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
        if agent not in action:
            return action in self.propagated_actions

        # otherwise check number of times self has been expanded with agent action
        agent_action = action[agent]
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
        if agent not in action:
            if not self.has_been_propagated_with(action):
                self.propagated_actions.append(action)
            return

        # find index of action
        agent_action = action[agent]
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

        # add noise to growth parameters (TODO: add noise as a model parameter)
        if model.agent_growth == growth.sigmoid_growth:
            speed_range = model.agent_growth_params["speed_range"]
            takeoff_time_range = model.agent_growth_params["takeoff_time_range"]

            speed_noise_scale = 0.03
            takeoff_time_noise_scale = 3

            if speed_range[0] < speed_range[1]:
                state[:, 2] += model.rng.normal(
                    loc=0, scale=speed_noise_scale, size=model.n_agents
                )
                state[:, 2] = state[:, 2].clip(*speed_range)

            if takeoff_time_range[0] < takeoff_time_range[1]:
                state[:, 3] += model.rng.integers(
                    low=-takeoff_time_noise_scale,
                    high=takeoff_time_noise_scale,
                    endpoint=True,
                    size=model.n_agents,
                )
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

        assert particle.node in self.node.tree.root_nodes

        return particle

    def __repr__(self) -> str:
        model = self.node.tree.forest.owner.model
        levels = tuple(growth.tech_level(state=self.state, model=model).round(2))
        return (
            f"Particle(levels {levels}, "
            + f"{self.joint_action_history}, {self.n_expansions} expansions, "
            + f"weight {self.weight})"
        )
