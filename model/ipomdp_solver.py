# %%

# avoids having to give type annotations as strings
from __future__ import annotations

from typing import TYPE_CHECKING, Tuple, List, Union, Dict, Generator
from numpy.typing import NDArray
from model import ipomdp, growth, action
import numpy as np
import math

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


def systematic_resample(
    sample: List,
    weights: Tuple,
    rng: random.Random,
    size: int = None,
) -> Tuple[List, List]:
    """
    Performs a systematic resampling of the elements in the list using the weights.

    Inspired by code in the filterPy library.

    Keyword arguments:
    sample: sample to resample
    weights: weights of elements in sample
    rng: random number generator
    size: desired resample size. If not supplied, resample will be same size as sample

    Returns:
    the new sample and counts for each element in sample
    """
    counts = [0] * len(sample)
    resample = []

    if size is None:
        size = len(sample)

    # calculate cumulative weights
    cum_weights = np.cumsum(weights)
    cum_weights[-1] = 1  # make sure last weight is 1

    # sample random number
    r = rng.random()

    # determine sample points (split interval [0,1] into “size” intervals and choose
    # points from these intervals with r as the offset from the interval boundary)
    points = (r + np.arange(size)) / size

    # find particles points land on
    point_i, element_i = 0, 0
    while point_i < size:
        if points[point_i] < cum_weights[element_i]:
            # add element to sample
            resample.append(sample[element_i])
            counts[element_i] += 1

            point_i += 1
        else:
            # move on to next element
            element_i += 1

    return resample, counts


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


def weight_particles(
    particles: Tuple[Particle, ...],
    observation: Observation,
    agent: civilisation.Civilisation,
    model: universe.Universe,
    prior_weights: Tuple[float] = None,
) -> Tuple[float, ...]:
    """
    Return weights to each particle according to how likely they are under the
    given observation.

    Keyword arguments:
    particles: particles to weight
    observation: the observation to use for weighting the particles
    agent: the observer
    model: a Universe
    prior_weights: can be supplied to give particles prior weights instead of accessing
                   them through particle.previous_particle.weight
    """
    if len(particles) == 0:
        return ()

    # combine states
    states = np.stack(tuple(p.state for p in particles), axis=0)
    actions = tuple(p.joint_action_history[-1] for p in particles)

    # find prior weights
    if prior_weights is None:
        prior_weights = np.array(tuple(p.previous_particle.weight for p in particles))

    # calculate weights
    weights = ipomdp.prob_observation(
        observation=observation,
        states=states,
        actions=actions,
        observer=agent,
        model=model,
    )
    weights *= prior_weights

    # normalise
    weight_sum = weights.sum()
    if weight_sum > 0:
        weights /= weight_sum

    return weights


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

        ### 1. update the beliefs in the top-level tree of the owner

        # set correct node as the new root
        old_root_node = self.top_level_tree_root_node
        new_root_node = old_root_node.child_nodes[owner_action]
        self.top_level_tree_root_node = new_root_node

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
            for particle, weight in zip(
                lower_node.particles, lower_particle_weights, strict=True
            ):
                particle.weight = weight

            # store weighted particles as a belief
            lower_node.belief = tuple(p for p in lower_node.particles if p.weight > 0)

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
        particle_weights = tuple(p.weight for p in node.particles)

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
        for particle, weight in zip(node.particles, particle_weights):
            particle.weight = weight

        # form a belief
        node.belief = tuple(p for p in node.particles if p.weight > 0)

        # 2. Sample a particle to start simulating from
        start_particle = model.random.choices(node.particles, weights=particle_weights)[
            0
        ]

        # weight particles in the immediate child trees
        self.forest.initialise_simulation(particle=start_particle)

        # 3. - 6. simulate starting from particle
        self.simulate_from(particle=start_particle)

        # 7. Clear particle weights from self and trees one level below
        # (ignoring root node weights in both cases)
        # for tree in self.forest.child_trees(self, include_parent=True):
        #     for root_node in tree.root_nodes:
        #         root_node.clear_child_particle_weights()

    def simulate_through(
        self,
        agent_action_history: AgentActionHistory,
        agent_observation_history,
        n_samples: int,
    ):
        """
        Performs a given amount of simulations starting from the given node (as
        specified by its agent action history) which should already hold the desired
        belief.

        If the specified node does not exist or it does not contain any particles,
        this will recursively backtrack along the tree until a node with a belief is
        found. Starting from this node, each node along the path is expanded a given
        number of times (note that expansions don't necessarily follow this path).

        Keyword arguments:
        agent_action_history: specifies the node in the current tree which we want to
                              expand
        agent_observation_history: history of observations for agent starting from the
                                   root of the tree and extending up to agent action
                                   history. Can be used to generate beliefs as we
                                   go up the path.

        Returns:
        The node corresponding to agent_action_history after all the simulations have
        been performed.
        """
        model = self.forest.owner.model

        tab = (len(self.signature) - 1) * "\t"
        print(tab, "Simulating through", agent_action_history, "in", self.signature)

        try:
            # attempt to find node
            node = self.get_node(agent_action_history)

            if node.parent_node is not None:
                # check that this node has a belief (which it does if there are
                # particles in it)
                assert len(node.particles) > 0

                # check that all actions have been expanded under the current belief
                W_a = tuple(
                    sum(p.weight * p.n_expansions_act[act] for p in node.particles)
                    for act in self.agent.possible_actions()
                )
                assert 0 not in W_a

                # check that there have been a sufficient number of expansions
                # TODO: make condition of sufficient number of expansions more rigorous
                N = sum(p.n_expansions for p in node.particles if p.weight > 0)
                assert N > 20

        except Exception:
            # first expand through the previous node
            prev_agent_action_history = agent_action_history[:-1]
            prev_agent_observation_history = agent_observation_history[:-1]
            prev_node = self.simulate_through(
                agent_action_history=prev_agent_action_history,
                agent_observation_history=prev_agent_observation_history,
                n_samples=n_samples,
            )

            # get current node
            node = prev_node.child_nodes[agent_action_history[-1]]

            # create belief with the stored observation
            node.weight_particles(agent_observation_history[-1])

        print(
            tab,
            node.agent_action_history,
            "Belief with",
            sum(p.weight > 0 for p in node.particles),
            "/",
            len(node.particles),
            ", max",
            round(max(tuple(p.weight for p in node.particles)), 3),
        )

        # check that we have a valid belief
        if sum(p.weight for p in node.particles) == 0:
            raise Exception("Particle filter has diverged.")

        # we have now found a node with a belief that we want to expand.
        # Start a given number of simulations from the node.
        # TODO: add number of simulations as a model parameter
        #       or base on condition of “sufficiently expanded”

        particle_sample = bin_resample(node.particles, size=n_samples, model=model)
        # particle_sample = model.random.choices(
        #     node.particles,
        #     weights=tuple(p.weight for p in node.particles),
        #     k=n_samples,
        # )

        for start_particle in particle_sample:
            # weight particles in the corresponding root nodes of child trees
            self.initialise_simulation(start_particle)

            # simulate observations for other agents up to the depth of start_particle
            other_agents = tuple(ag for ag in model.agents if ag != self.agent)
            joint_observation_history = {ag: [] for ag in other_agents}
            sampling_particle = start_particle

            while sampling_particle.previous_particle is not None:
                for other_agent in other_agents:
                    # sample observation for other agent from sampling particle
                    other_agent_observation = ipomdp.sample_observation(
                        state=sampling_particle.state,
                        action=sampling_particle.joint_action_history[-1],
                        agent=other_agent,
                        model=model,
                    )

                    joint_observation_history[other_agent].append(
                        other_agent_observation
                    )

                sampling_particle = sampling_particle.previous_particle

            # reverse observation sequences (now they are oldest observations last)
            for other_agent in other_agents:
                joint_observation_history[other_agent].reverse()

            # start simulation
            self.simulate_from(
                start_particle, joint_observation_history=joint_observation_history
            )

        # after finishing, return the node
        return node

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
        model = self.forest.owner.model
        node = particle.node

        # don't simulate farther than the discount horizon
        if model.discount_factor**depth < model.discount_epsilon:
            print(
                f"Simulation reached the discount horizon at {node.agent_action_history}"
            )
            return 0

        # if we used an unexpanded action last time,
        # perform a rollout and end recursion
        if do_rollout:
            print(f"Simulation reached a leaf node at {node.agent_action_history}")

            # make a copy because rollout changes state in place
            start_state = particle.state.copy()
            value = ipomdp.rollout(state=start_state, agent=self.agent, model=model)

            # add the new particle (it will have no expansions)
            node.particles.append(particle)

            # end recursion
            return value

        # create an empty joint observation history
        if joint_observation_history is None:
            joint_observation_history = {ag: [] for ag in model.agents}

        ### 1. choose actor
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
            try:
                lower_node = self.forest.get_matching_lower_node(
                    particle=particle, agent=actor
                )

                # check that there are particles
                assert len(lower_node.belief) > 0

                # check that all actions have been expanded
                W_a = tuple(
                    sum(p.weight * p.n_expansions_act[act] for p in lower_node.belief)
                    for act in actor.possible_actions()
                )
                assert 0 not in W_a

                # # check that there have been a sufficient number of expansions
                # N = sum(p.n_expansions for p in lower_node.particles if p.weight > 0)
                # assert N > 20

            except LookupError:
                print("Could not find a matching node in child tree")
                actor_action = ipomdp.level0_opponent_policy(agent=actor, model=model)
            except Exception:
                # TODO
                if len(lower_node.belief) == 0:
                    print("Belief in lower tree has diverged")
                else:
                    print("All actions in lower node have not been expanded")
                actor_action = ipomdp.level0_opponent_policy(agent=actor, model=model)
            else:
                # determine action
                actor_action, _ = lower_node.tree.tree_policy(
                    node=lower_node, explore=False, softargmax=True
                )

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
        if (actor == self.agent and particle.n_expansions_act[agent_action] > 0) or (
            actor != self.agent and action_ in particle.propagated_actions
        ):
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
                    print("Could not resample node in child tree (node doesn't exist)")
                    pass
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
                    print("Could not create belief in child tree (node doesn't exist)")
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

        # save value and next agent action to particle
        value = agent_action_value + model.discount_factor * future_value
        particle.add_expansion(action=action_, value=value)

        # add particle to node
        # (except if this is the particle where we started simulating)
        if depth > 0:
            node.particles.append(particle)

        return value

    def update_beliefs(self):
        """
        Update the beliefs of this tree using the parent tree.
        """
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
        model = node.tree.forest.owner.model

        assert node.tree == self

        # check that there is a belief
        assert sum(p.weight is None for p in node.belief) == 0

        # calculate necessary quantities
        actions = self.agent.possible_actions()
        N = sum(particle.n_expansions for particle in node.belief)
        W_a = tuple(
            sum(p.weight * p.n_expansions_act[act] for p in node.belief)
            for act in actions
        )
        W = sum(W_a)

        if explore and min(W_a) < 1e-10:
            # if there are unexpanded actions under this belief, sample one
            unexpanded_actions = tuple(
                action for action, weight in zip(actions, W_a) if weight < 1e-10
            )
            return model.random.choice(unexpanded_actions), True

        elif not explore and N == 0:
            # if none of the actions are expanded, choose action according to
            # level 0 default policy
            raise Exception("TODO: Lower node diverged")
            return (ipomdp.level0_opponent_policy(agent=self.agent, model=model), False)

        elif not explore and 0 in W_a:
            # if some of the actions are not expanded and we are not exploring,
            # ignore these unexpanded actions
            raise Exception("TODO: Lower node has unexplored actions")
            W_a, actions = zip(
                *(
                    (weight, action)
                    for weight, action in zip(W_a, actions)
                    if weight > 0
                )
            )

        N_a = tuple((w_a / W) * N for w_a in W_a)

        # calculate values of different actions
        Q = tuple(
            sum(
                p.weight * p.act_value[action]
                for p in node.belief
                if p.n_expansions_act[action] > 0
            )
            / sum(p.weight for p in node.belief if p.n_expansions_act[action] > 0)
            for action in actions
        )

        if explore:
            # add exploration bonuses
            Q = tuple(
                q + model.exploration_coef * math.sqrt(math.log(N) / n_a)
                for n_a, q in zip(N_a, Q, strict=True)
            )

        if not explore and softargmax:
            # use softargmax to calculate weights of different actions
            action_weights = tuple(math.exp(n_a / math.sqrt(N)) for n_a in N_a)
            action_weight_sum = sum(action_weights)
            action_weights = tuple(w / action_weight_sum for w in action_weights)

            # choose action
            choice = model.random.choices(actions, weights=action_weights)[0]
            return choice, False

        # return the action that maximises Q
        choice = max(zip(Q, actions, strict=True), key=lambda x: x[0])[1]
        return choice, False

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
        self.particles = []
        # belief is a tuple of weighted particles, weights summing to 1
        self.belief = tuple()
        # stores the observation used to generate the belief stored in ‘belief’
        self.belief_observation = None

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
        self.particles = [
            Particle(state=state, joint_action_history=(), node=self)
            for state in initial_particle_states
        ]

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
                particle_weights = n_particles * (1 / n_particles,)
                for particle in self.particles:
                    particle.lower_particle_dist[child_tree.agent] = particle_weights

        # if the node is in the top level tree, its particles need weights
        if in_top_level_tree:
            for particle in self.particles:
                particle.weight = 1 / len(initial_particle_states)

            # store created belief
            self.belief = tuple(p for p in self.particles)

    def __repr__(self) -> str:
        return (
            f"Node({self.agent_action_history}, " + f"{len(self.particles)} particles)"
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
        weights = weight_particles(
            particles=self.particles,
            observation=observation,
            agent=self.tree.agent,
            model=self.tree.forest.owner.model,
        )

        # save weights
        for particle, weight in zip(self.particles, weights, strict=True):
            particle.weight = weight

        # save particles with positive weights into belief
        self.belief = tuple(p for p in self.particles if p.weight > 0)

        # save observation used to create the weights
        self.belief_observation = observation

    def resample_particles(self):
        """
        Resamples the current belief and replaces it in the belief attribute.
        """
        model = self.tree.forest.owner.model
        n_particles = len(self.belief)

        if n_particles == 0:
            return

        particle_weights = tuple(p.weight for p in self.belief)

        self.belief, _ = systematic_resample(
            sample=self.belief,
            weights=particle_weights,
            rng=model.random,
        )

        # reset weights of particles
        for particle in self.particles:
            particle.weight = 0

        for particle in self.belief:
            particle.weight = 1 / n_particles

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
                posterior_weights = weight_particles(
                    particles=lower_node.particles,
                    observation=other_agent_obs,
                    agent=other_agent,
                    model=model,
                )

                # save
                particle.lower_particle_dist[other_agent] = posterior_weights


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
        weight: float | None = None,
        previous_particle: Particle | None = None,
    ) -> None:
        # identifiers of the particle
        self.node = node
        self.previous_particle = previous_particle  # can be None

        # properties of the particle
        self.state = state
        self.joint_action_history = joint_action_history
        self.lower_particle_dist = dict()
        self.weight = weight  # can be None

        # initialise dictionaries tracking number of times expanded and average values
        agent = node.tree.agent
        possible_actions = agent.possible_actions()
        self.n_expansions = 0
        self.n_expansions_act = {act: 0 for act in possible_actions}
        self.act_value = {act: 0 for act in possible_actions}

        # this keeps track of others' actions that have been used to propagate
        # this particle
        self.propagated_actions: List[Action] = []

    def agent_action_history(
        self, agent: civilisation.Civilisation
    ) -> AgentActionHistory:
        """
        Returns the agent action history of the given agent, extracted from the
        joint action history stored in the particle.
        """
        return joint_to_agent_action_history(self.joint_action_history, agent=agent)

    def add_expansion(self, action: Action, value: float) -> None:
        """
        Add information about an expansion performed starting from the particle.

        Keyword arguments:
        action: action used to propagate the particle
        value: value received for taking this action and continuing optimally afterwards
        """
        # if someone else acted, we only store the action
        agent = self.node.tree.agent

        if agent not in action:
            if action not in self.propagated_actions:
                self.propagated_actions.append(action)

            return

        agent_action = action[agent]
        assert agent_action in self.n_expansions_act

        prev_n_expansions = self.n_expansions_act[agent_action]
        prev_value = self.act_value[agent_action]

        # calculate new average value based on previous average and the new value
        new_value = prev_value + (value - prev_value) / (prev_n_expansions + 1)

        # update information
        self.n_expansions += 1
        self.n_expansions_act[agent_action] += 1
        self.act_value[agent_action] = new_value

    def add_noise(self):
        model = self.node.tree.forest.owner.model

        # add noise to growth parameters (TODO: add noise as a model parameter)
        if model.agent_growth == growth.sigmoid_growth:
            speed_range = model.agent_growth_params["speed_range"]
            takeoff_time_range = model.agent_growth_params["takeoff_time_range"]

            speed_noise_scale = 0.03
            takeoff_time_noise_scale = 3

            if speed_range[0] < speed_range[1]:
                self.state[:, 2] += model.rng.normal(
                    loc=0, scale=speed_noise_scale, size=model.n_agents
                )
                self.state[:, 2] = self.state[:, 2].clip(*speed_range)

            if takeoff_time_range[0] < takeoff_time_range[1]:
                self.state[:, 3] += model.rng.integers(
                    low=-takeoff_time_noise_scale,
                    high=takeoff_time_noise_scale,
                    endpoint=True,
                    size=model.n_agents,
                )
                self.state[:, 3] = self.state[:, 3].clip(*takeoff_time_range)
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
