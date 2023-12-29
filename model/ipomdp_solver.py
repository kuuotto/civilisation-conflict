# %%

# avoids having to give type annotations as strings
from __future__ import annotations

from random import Random
from typing import (
    Iterable,
    Mapping,
    Sequence,
    cast,
)

import numpy as np
from numba import njit
from numpy.typing import NDArray

from envs.ipomdp import (
    IPOMDP,
    Action,
    ActivationSchedule,
    Agent,
    JointAction,
    Model,
    Observation,
    State,
)

JointActionHistory = tuple[JointAction, ...]
AgentActionHistory = tuple[Action, ...]
ForestSignature = tuple[Agent, ...]
Belief = NDArray[np.float_]


# parameters which can be adjusted to change momory consumption
WEIGHT_STORAGE_DTYPE = np.float32
SparseBelief = tuple[NDArray[np.bool_], NDArray[WEIGHT_STORAGE_DTYPE]]
NODE_ARRAY_INIT_SIZE = 5
NODE_ARRAY_INCREASE_SIZE = 50


def encode_belief(belief: Belief) -> SparseBelief:
    """
    Calculates a representation of the belief in sparse format.
    """
    weight_mask = belief > 0
    weight_values = belief[weight_mask].astype(WEIGHT_STORAGE_DTYPE)
    return weight_mask, weight_values


def decode_belief(
    encoded_belief: SparseBelief,
) -> Belief:
    """
    Returns the original belief from a sparse representation.
    """
    weight_mask, weight_values = encoded_belief
    belief = np.zeros(len(weight_mask))
    belief[weight_mask] = weight_values
    return belief


def joint_to_agent_action_history(
    joint_action_history: JointActionHistory, agent: Agent
) -> AgentActionHistory:
    """
    Extracts the agent action history of the specified agent from a joint
    action history.

    Keyword arguments:
    joint_history: joint action history
    agent: agent whose action history to determine
    """
    return tuple(joint_action[agent] for joint_action in joint_action_history)


@njit
def calculate_action_qualities(
    belief: Belief,  # shape (n_states,)
    n_expansions: NDArray,  # shape (n_states,)
    n_expansions_act: NDArray,  # shape (n_states, n_actions)
    act_value: NDArray,  # shape (n_states, n_actions)
    explore: bool,
    exploration_coef: float | None = None,
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
        assert exploration_coef is not None
        # add exploration bonuses
        N_a = (W_a / W_a.sum()) * N
        Q += exploration_coef * np.sqrt(np.log(N) / N_a)

    return Q


@njit
def calculate_action_probabilities(
    belief: Belief,  # shape (n_states,)
    n_expansions: NDArray,  # shape (n_states,)
    n_expansions_act: NDArray,  # shape (n_states, n_actions)
    act_value: NDArray,  # shape (n_states, n_actions)
    softargmax_coef: float,
) -> tuple[NDArray, int]:  # shape (n_actions,)
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


class BeliefForestIPOMDPSolver:
    """
    Solves an I-POMDP using a nested hierarchy of forests, each of which
    represents either the owner's own decision process or its model of another agent's
    decision process.
    """

    def __init__(
        self,
        ipomdp: IPOMDP,
        random: Random,
        n_tree_simulations: int,
        n_initial_particles: int,
        discount_epsilon: float,
        debug: int,
    ) -> None:
        """
        Keyword arguments:
        ipomdp: the underlying I-POMDP to be solved
        random: a random number generator object
        """
        self.random = random
        self.n_tree_simulations = n_tree_simulations
        self.n_initial_particles = n_initial_particles
        self.discount_epsilon = discount_epsilon
        self.debug = debug

        # create the owner's own forest at the highest level. Lower-level forests
        # are created recursively.
        self.owner_forest_group = ForestGroup(
            parent_forest=None, models=(ipomdp,), solver=self, agent=ipomdp.owner
        )

    def plan(self):
        """
        Uses the MCTS (Monte Carlo Tree Search) based algorithm to simulate
        planning by the owner of the forest.

        Forests are expanded from the bottom up, starting at the lowest level.
        """
        self.owner_forest_group.plan()

    def update_beliefs(
        self, owner_action: Action, owner_observation: Observation
    ) -> None:
        """
        Updates the beliefs in all forests after the owner takes the given
        action.

        Keyword arguments:
        owner_action: action taken by the owner of the forest
        owner_observation: observation received by the owner
        """
        if self.debug >= 1:
            print(f"{self.owner_forest.agent} observed {owner_observation}")

        ### 1. update the beliefs in the top-level forest of the owner

        # find the old and new root nodes
        old_root_node = self.owner_forest_root_node
        new_root_node = old_root_node.child_nodes[owner_action]

        # resample the old root node
        old_root_node.resample_particles()

        # weight particles in the top-level tree root node
        new_root_node.weight_particles(owner_observation)

        # check that the weights do not sum to 0
        if new_root_node.belief.sum() == 0:
            raise Exception("The weights in the top-level forest root node are all 0")

        ### 2. Update belief in the interactive states over the lower-level models
        new_root_node.update_other_agents_beliefs()

        # remove references to previous particles
        for particle in new_root_node.particles:
            particle.previous_particle = None

        # remove reference to previous node
        new_root_node.parent_node = None

        # set correct node as the new root
        self.owner_forest_root_node = new_root_node

        # if self.debug >= 1:
        #     print(
        #         f"{self.owner_forest}: ({new_root_node.agent_action_history})",
        #     )

        #     # report the proportions of particles with different models
        #     if self.owner_reasoning_level > 0:
        #         other_agents = tuple(
        #             ag for ag in self.ipomdp.agents if ag != self.owner
        #         )
        #         prob_indifferent = {other_agent: 0 for other_agent in other_agents}
        #         for particle in new_root_node.particles:
        #             for other_agent in other_agents:
        #                 if (
        #                     particle.other_agent_frames[other_agent.id]["attack_reward"]
        #                     == 0
        #                 ):
        #                     prob_indifferent[other_agent] += particle.weight
        #         print(
        #             f"In {self.owner_forest}, {prob_indifferent} of weight is given to indifferent models of the other agents"
        #         )

        ### 3. update the beliefs in the child forests recursively
        for other_agent in self.owner_forest.ipomdp.other_agents:
            # group corresponding to the owner's models of other_agent
            other_agent_forest_group = self.owner_forest.other_agent_forest_group[
                other_agent
            ]

            # update beliefs in each forest in the group. Recursively updates lower
            # beliefs
            for other_agent_forest in other_agent_forest_group.forests:
                other_agent_forest.step_forward()

    def optimal_action(self):
        """
        Return the optimal action of the owner of the I-POMDP.
        """
        return self.owner_forest_root_node.tree_policy(explore=False, softargmax=False)[
            0
        ]

    @property
    def owner_forest(self) -> Forest:
        assert len(self.owner_forest_group.forests) == 1
        return self.owner_forest_group.forests[0]

    @property
    def owner_forest_root_node(self) -> Node:
        return self.owner_forest.root_nodes[0]

    @owner_forest_root_node.setter
    def owner_forest_root_node(self, new_node: Node) -> None:
        self.owner_forest.root_nodes[0] = new_node


class ForestGroup:
    """
    A forest group corresponds to a group of alternative I-POMDPs for an agent. Each
    forest group is simulated a fixed number of times, the different forests simulated
    in proportion to their likelihood.
    """

    def __init__(
        self,
        parent_forest: Forest | None,
        models: Iterable[Model],
        solver: BeliefForestIPOMDPSolver,
        agent: Agent,
    ) -> None:
        self.parent_forest = parent_forest
        self.solver = solver
        self.agent = agent
        self.forests: list[Forest] = []

        # create a forest for each I-POMDP
        for model in models:
            # if the model is not an I-POMDP, we do not create a forest for it
            if not isinstance(model, IPOMDP):
                continue

            # check that the I-POMDP is agent's
            assert model.owner == agent

            # create the forest. This will recursively create lower-level forests.
            forest = Forest(ipomdp=model, forest_group=self, solver=solver)

            # add it to the collection of child forests
            self.forests.append(forest)

        # calculate signature, which shows the ownership hierarchy of this forest group
        self.signature: tuple[Agent, ...] = (
            (agent,)
            if parent_forest is None
            else parent_forest.forest_group.signature + (agent,)
        )

    def plan(self):
        """
        Uses the MCTS (Monte Carlo Tree Search) based algorithm to simulate
        planning by the owner of the forest group.

        Forest groups are expanded from the bottom up, starting at the lowest level.
        """
        # first plan in the forest groups of the other agents
        for forest in self.forests:
            for other_agent in forest.ipomdp.other_agents:
                forest.other_agent_forest_group[other_agent].plan()

        # then plan in this forest group
        if self.solver.debug >= 2:
            print(f"Planning in {self}")

        # count the number of successful simulations
        n_successful = 0

        # simulate
        for _ in range(self.solver.n_tree_simulations):
            n_successful += self.simulate()

        if self.solver.debug >= 1:
            print(
                f"{n_successful} / {self.solver.n_tree_simulations}",
                "successful simulations in {self}",
            )

    def simulate(self) -> bool:
        """
        Expand the current forest group by
        1. Randomly choosing particles top-down to determine which forest and root node
           to start expanding from. This is also how we find weights for the particles
           in the chosen root node.
        2. Sampling a particle from the chosen root node.
        3. Using a version of MCTS to traverse the tree, at each step choosing others'
           actions by simulating their beliefs and corresponding optimal actions at
           the level below.
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
        # 1. Choose the root node to start sampling from
        current_forest = self.solver.owner_forest
        current_node = self.solver.owner_forest_root_node
        particle_weights = current_node.belief

        for next_agent in self.signature[1:]:
            # choose random particle given weights
            particle = self.solver.random.choices(
                current_node.particles,
                # cast weights to a sequence so that type checker is happy
                weights=cast(Sequence[float], particle_weights),
            )[0]

            # find matching node
            current_forest, current_node = particle.get_agent_child_forest_node(
                agent=next_agent
            )
            assert current_node is not None

            # get weights for node based on the particle
            belief = particle.other_agents_belief[next_agent]
            assert belief is not None
            mask, values = belief
            particle_weights = np.zeros(len(mask))
            particle_weights[mask] = values

            # if belief is diverged, we cannot simulate
            if particle_weights.sum() == 0:
                return False

        assert current_forest in self.forests
        assert current_node in current_forest.root_nodes

        # save weights to particles
        current_node.belief = particle_weights

        # 2. Sample a particle to start simulating from
        start_particle = self.solver.random.choices(
            current_node.particles,
            weights=cast(Sequence[float], particle_weights),
        )[0]

        # weight particles in the immediate child trees
        other_agent_nodes = start_particle.initialise_child_forest_beliefs()

        # 3. - 6. simulate starting from particle
        current_forest.simulate_from_particle(
            particle=start_particle, other_agent_nodes=other_agent_nodes
        )

        return True

    def is_empty(self) -> bool:
        return len(self.forests) == 0

    def get_forest(self, ipomdp: IPOMDP) -> Forest | None:
        """
        Given an ipomdp, find the forest in this forest group that represents the
        ipomdp. Return None if no such forest exists.
        """
        for forest in self.forests:
            if forest.ipomdp is ipomdp:
                return forest

        return None


class Forest:
    """
    A Forest corresponds to a single agent. The forest is the child of another forest
    that corresponds to an agent at the level above (unless it is the owner's top-level
    forest). The forest is a collection of multiple trees. The root nodes of
    each of these correspond to a possible agent history and frame for the tree agent.
    """

    def __init__(
        self,
        ipomdp: IPOMDP,
        forest_group: ForestGroup,
        solver: BeliefForestIPOMDPSolver,
    ) -> None:
        """
        Initialise the forest.

        Keyword argument:
        ipomdp: the I-POMDP this forest represents.
        forest_group: the group this forest belongs to
        solver: the solver using this forest
        """
        self.ipomdp = ipomdp
        self.forest_group = forest_group
        self.agent = ipomdp.owner
        self.solver = solver

        # create a root node corresponding to empty agent action history
        root_node = Node(
            forest=self,
            agent_action_history=(),
            parent_node=None,
            random=solver.random,
        )
        self.root_nodes = [root_node]

        # create child forests
        self.other_agent_forest_group = {
            agent: ForestGroup(
                parent_forest=self,
                models=ipomdp.agent_models(agent)[1],
                solver=solver,
                agent=agent,
            )
            for agent in ipomdp.other_agents
        }

        # create initial particles in the root node
        root_node.create_initial_particles(
            n_particles=solver.n_initial_particles,
            other_agents_forest_group=self.other_agent_forest_group,
        )

        # if this is the top level forest, initialise it with a uniform belief
        if self is solver.owner_forest:
            root_node.belief = np.array(
                solver.n_initial_particles * (1 / solver.n_initial_particles,)
            )

    # def _create_child_forest_groups(self) -> None:
    #     """
    #     Create forests for the opponents of the owner of the current forest.
    #     """
    #     # create child forests for every opponent
    #     for other_agent in self.ipomdp.other_agents:
    #         # find the different possible models used for other_agent
    #         models = self.ipomdp.agent_models(other_agent)

    #         # create a forest for each intentional model in the list of models
    #         self.other_agent_forest_group[other_agent] = ForestGroup(models=models)

    def get_node(self, agent_action_history: AgentActionHistory) -> Node | None:
        """
        Find node in the forest matching the given agent action history.

        Returns None if node is not found.
        """

        # find the correct root node
        for root_node in self.root_nodes:
            root_length = len(root_node.agent_action_history)

            if agent_action_history[:root_length] == root_node.agent_action_history:
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

        return current_node

    def simulate_from_particle(
        self,
        particle: Particle,
        other_agent_nodes: Mapping[Agent, Node | None],
        do_rollout: bool = False,
        depth: int = 0,
    ) -> float:
        """
        Simulate decision-making from the current particle at node by
        1. choosing an action from each agent
        2. propagating the particle with this action using the transition \
           function of the I-POMDP
        3. generating an observation for the forest agent and creating a new belief in \
           the next node
        4. generating an observation for each other agent and updating the beliefs in \
           the forest on the level below
        5. repeat

        Keyword arguments:
        particle: particle to start simulating from
        other_agent_nodes: nodes corresponding to particle in the trees one level below.
                           Only contains entries for agents that are modelled as
                           rational (with an I-POMDP).
        do_rollout: whether to stop the simulation process and do a rollout to \
                    determine the value in the leaf node
        depth: how deep into the recursion we are. 0 corresponds to the starting point \
               for the simulation.

        Returns:
        The value of taking the chosen action from particle and then continuing
        optimally.
        """
        solver = self.solver
        ipomdp = self.ipomdp
        node = particle.node
        agent = ipomdp.owner

        # don't simulate farther than the discount horizon
        discount_horizon_reached = (
            ipomdp.discount_factor**depth < solver.discount_epsilon
        )
        if discount_horizon_reached:
            if solver.debug == 2:
                print(
                    f"Simulation reached the discount horizon at {node.agent_action_history}"
                )
            do_rollout = True

        # if we used an unexpanded action last time,
        # perform a rollout and end recursion
        if do_rollout:
            if not discount_horizon_reached and solver.debug == 2:
                print(f"Simulation reached a leaf node at {node.agent_action_history}")

            # obtain an estimate of the utility of the state
            value = ipomdp.utility_estimate(state=particle.state)

            # add the new particle (it will have no expansions)
            if not discount_horizon_reached:
                node.add_particle(particle)

            # end recursion
            return value

        ### 1. choose an action for each agent according to the activation schedule
        action_unvisited = False

        # determine actor(s) according to the activation schedule
        if self.ipomdp.activation_schedule == ActivationSchedule.JointActivation:
            actors = ipomdp.agents
        elif self.ipomdp.activation_schedule == ActivationSchedule.RandomActivation:
            actors = (solver.random.choice(ipomdp.agents),)
        else:
            raise NotImplementedError

        joint_action = {}

        for actor in actors:
            if actor == agent:
                # use tree policy to choose action
                actor_action, action_unvisited = node.tree_policy(explore=True)

            else:
                ### someone else is acting

                # check if the actor is modelled with a rational model and obtain the
                # default (non-rational) policy
                actor_default_policy, _ = ipomdp.agent_models(actor)
                actor_is_rational = actor in other_agent_nodes

                if actor_is_rational:
                    ### use the forest below to choose the action

                    # get matching node
                    other_agent_node = other_agent_nodes[actor]

                    # if node is not found, choose with default policy
                    if other_agent_node is None:
                        # TODO: Log
                        # solver.add_log_event(
                        #     event_type=11, event_data=(self.signature,)
                        # )
                        actor_action = actor_default_policy.act()

                    else:
                        # determine action from the other agent's tree
                        actor_action, _ = other_agent_node.tree_policy(
                            explore=False,
                            softargmax=True,
                            simulated_forest=self,
                        )

                else:
                    # if we don't model the other agent as rational (we don't have a
                    # corresponding lower level forest modelling them) then their action
                    # is chosen according to the default policy
                    actor_action = actor_default_policy.act()

            # add to joint action
            joint_action[actor] = actor_action

        # retrieve the forest agent's action. This is None if the agent did not act.
        agent_action = (
            joint_action[agent] if agent in joint_action else ipomdp.no_turn_action
        )

        ### 2. Propagate state
        propagated_state, agent_reward = ipomdp.transition(
            state=particle.state, joint_action=joint_action
        )

        # create a new node if necessary
        if agent_action not in node.child_nodes:
            new_agent_action_history = node.agent_action_history + (agent_action,)
            new_node = Node(
                forest=self,
                agent_action_history=new_agent_action_history,
                parent_node=node,
                random=solver.random,
            )
            node.child_nodes[agent_action] = new_node

        next_node = node.child_nodes[agent_action]

        # initialise a new particle
        new_joint_action_history = particle.joint_action_history + (joint_action,)
        next_particle = Particle(
            state=propagated_state,
            joint_action_history=new_joint_action_history,
            node=next_node,
            previous_particle=particle,
        )

        # if particle has already been propagated with this action, add some noise to
        # the state of next_particle
        if particle.has_been_propagated_with(joint_action):
            # this should only happen in a root node
            assert particle.node in self.root_nodes

            next_particle.add_noise(solver.random)

        ### 3. Update the belief in the current forest
        # if the action is unvisited, creating weights is not necessary as we perform
        # a rollout from it on the next recursion
        if not action_unvisited:
            # first resample the belief in the current node
            node.resample_particles()

            # generate an observation
            agent_obs = ipomdp.sample_observation(
                state=propagated_state,
                prev_joint_action=joint_action,
            )

            # weight particles in the next node
            next_node.weight_particles(agent_obs)

        ### 4. Generate observations for other agents and update beliefs about
        ### interactive states at the level below
        next_other_agent_nodes = {}

        if not action_unvisited:
            # find the nodes in the trees one level below for the next time step
            next_other_agent_nodes = {
                other_agent: None
                if other_agent_node is None
                else other_agent_node.child_nodes.get(joint_action[other_agent])
                for other_agent, other_agent_node in other_agent_nodes.items()
            }

            for other_agent in next_other_agent_nodes:
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
                other_agent_obs = lower_node.ipomdp.sample_observation(
                    state=propagated_state,
                    prev_joint_action=joint_action,
                )

                # assign weights to particles
                next_lower_node.weight_particles(other_agent_obs)

                # log successful creation of lower node belief
                # model.add_log_event(
                #     event_type=30, event_data=(self.signature, other_agent)
                # )

        ### 5. Repeat

        future_value = self.simulate_from_particle(
            particle=next_particle,
            other_agent_nodes=next_other_agent_nodes,
            do_rollout=action_unvisited,
            depth=depth + 1,
        )

        ### Simulation is done
        # add particle to node
        # (except if this is the particle where we started simulating)
        if depth > 0:
            node.add_particle(particle)

        # save value and next agent action to particle
        value = agent_reward + ipomdp.discount_factor * future_value
        particle.add_expansion(
            joint_action=joint_action, value=value, next_particle=next_particle
        )

        # clean up beliefs to save memory
        if node != solver.owner_forest_root_node:
            node.belief = None
        else:
            # for the top level tree root node, we only clear the resampling
            # this is necessary from a functional standpoint, not just memory
            node.resample_counts = None

        # clean up beliefs from the lower nodes
        for lower_node in other_agent_nodes.values():
            if lower_node is None:
                continue

            lower_node.belief = None

        return value

    def step_forward(self):
        """
        Steps the forest forward in time by:
        1. Setting the immediate children of the current root nodes as the new root nodes
        2. Pruning new root nodes that are not referred to in the parent forest
        3. Updating other agents' beliefs in each particle

        Finally, child forests are stepped forward recursively.

        This assumes that the parent forest of self has already been stepped forward.
        """
        assert self.forest_group.parent_forest is not None

        ### 1. Find new root nodes
        new_root_nodes = tuple(
            child_node
            for node in self.root_nodes
            for child_node in node.child_nodes.values()
        )
        self.root_nodes: list[Node] = []

        ### 2. Prune new root nodes that are no longer needed (i.e. will never be
        ###    expanded because the agent action histories they represent are no
        ###    longer present in particles on the level above)
        for root_node in new_root_nodes:
            # check all parent forest root nodes for matching particles
            parent_particles = (
                p
                for parent_root_node in self.forest_group.parent_forest.root_nodes
                for p in parent_root_node.particles
            )
            for p in parent_particles:
                if (
                    p.agent_action_history(self.agent) == root_node.agent_action_history
                    and p.other_agents_forest[self.agent] is self
                ):
                    # found a matching particle, so this root node gets to stay.
                    break

            else:
                # no matching particles, so root node is pruned
                continue

            self.root_nodes.append(root_node)

        if len(self.root_nodes) == 0:
            raise Exception("No root nodes left")

        ### 3. Update other agents' beliefs
        for root_node in self.root_nodes:
            root_node.update_other_agents_beliefs()

            # Now we can remove references to previous particles
            for particle in root_node.particles:
                particle.previous_particle = None

            # remove reference to parent node
            root_node.parent_node = None

        # if model.debug >= 1:
        #     print(
        #         f"{self.signature}:",
        #         len(self.root_nodes),
        #         sorted(
        #             list(
        #                 (len(n.particles), n.agent_action_history[-1], n.frame)
        #                 for n in self.root_nodes
        #             ),
        #             key=lambda x: x[0],
        #             reverse=True,
        #         ),
        #     )

        # recurse
        for other_agent in self.ipomdp.other_agents:
            for other_agent_forest in self.other_agent_forest_group[
                other_agent
            ].forests:
                other_agent_forest.step_forward()

    def __repr__(self) -> str:
        return (
            f"Forest({self.forest_group.signature}, {len(self.root_nodes)} root nodes)"
        )


class Node:
    """
    A single node in a forest. Corresponds to a specific agent action history.
    A node contains a set of particles.
    """

    def __init__(
        self,
        forest: Forest,  # TODO: Remove and just supply ipomdp?
        agent_action_history: AgentActionHistory,
        parent_node: Node | None,
        random: Random,
    ) -> None:
        """
        Initialise the node.

        Keyword arguments:
        forest: the forest this node belongs to
        agent_action_history: the agent action history this node corresponds to
        parent_node: the parent node of this node
        random: a random number generator instance
        """
        self.random = random

        # identifiers of node
        # self.forest = forest
        self.agent_action_history = agent_action_history
        self.ipomdp = forest.ipomdp

        # relation of node to other nodes in the forest
        self.parent_node = parent_node
        self.child_nodes: dict[Action, Node] = dict()

        # particles is a list of all particles stored in the node
        self.particles: list[Particle] = []
        # belief contains weights for particles
        self._belief: Belief | None = None
        # contains the sampled counts of particles if the particles have been resampled
        self.resample_counts = None
        # stores the observation used to generate the belief stored in ‘belief’.
        # Included for debugging purposes.
        self.belief_observation = None
        # number of actions available in the I-POMDP for the owner
        self.n_actions = len(self.ipomdp.possible_actions(self.ipomdp.owner))

        # these arrays store the information of the n particles:
        # - states
        self._states = np.zeros(
            shape=(NODE_ARRAY_INIT_SIZE, *self.ipomdp.state_shape),
            dtype=self.ipomdp.state_dtype,
        )
        # - previous joint actions
        self._prev_joint_actions = np.zeros(
            shape=(
                NODE_ARRAY_INIT_SIZE,
                self.ipomdp.n_agents,
                *self.ipomdp.action_shape,
            ),
            dtype=self.ipomdp.action_dtype,
        )
        # - number of times a particle has been expanded
        # TODO: Does it make sense to store this when it could be calculated from
        # n_expansions_act?
        self._n_expansions = np.zeros(shape=(NODE_ARRAY_INIT_SIZE,), dtype=np.uint16)
        # - number of times a particle has been expanded with each action
        self._n_expansions_act = np.zeros(
            shape=(NODE_ARRAY_INIT_SIZE, self.n_actions), dtype=np.uint16
        )
        # - value estimates of each action
        self._act_value = np.zeros(shape=(NODE_ARRAY_INIT_SIZE, self.n_actions))
        # - ids of previous particles (the preceding particle in the previous node
        #   for each particle in this node)
        self._prev_particle_ids = np.full(
            shape=(NODE_ARRAY_INIT_SIZE,), fill_value=-1, dtype=np.int32
        )

    @property
    def belief(self) -> Belief:
        if self.is_resampled:
            assert self.resample_counts is not None
            # when belief is resampled, the weights of the chosen particles are uniform
            # TODO: If a particle is chosen multiple times in the resample, then the
            # number of particles chosen is smaller than self.n_particles and the belief
            # returned here is not normalised. Does this matter?
            n_particles = self.n_particles
            return (1 / n_particles) * (self.resample_counts > 0)

        assert self._belief is not None
        assert len(self._belief) == self.n_particles
        return self._belief

    @belief.setter
    def belief(self, new_belief: Belief | None) -> None:
        assert (new_belief is None) or (new_belief.shape[0] == self.n_particles)
        self._belief = new_belief
        self.resample_counts = None
        self.belief_observation = None

    @property
    def n_particles(self) -> int:
        return len(self.particles)

    @property
    def array_size(self) -> int:
        return self.states.shape[0]

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
    def prev_joint_actions(self) -> NDArray:
        return self._prev_joint_actions[: self.n_particles]

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
        # n_agents = self.forest.ipomdp.n_agents

        # give particle its index in the arrays
        particle.id = self.n_particles

        # add particle
        self.particles.append(particle)

        # increase array sizes if necessary
        if self.n_particles > self.array_size:
            # states
            new_states = np.zeros(
                shape=(NODE_ARRAY_INCREASE_SIZE, *self._states.shape[1:]),
                dtype=self._states.dtype,
            )
            self._states = np.concatenate((self._states, new_states), axis=0)

            # n_expansions
            new_n_expansions = np.zeros(
                shape=(NODE_ARRAY_INCREASE_SIZE,), dtype=self.n_expansions.dtype
            )
            self._n_expansions = np.concatenate(
                (self._n_expansions, new_n_expansions), axis=0
            )

            # n_expansions_act
            new_n_expansions_act = np.zeros(
                shape=(NODE_ARRAY_INCREASE_SIZE, self.n_actions),
                dtype=self._n_expansions_act.dtype,
            )
            self._n_expansions_act = np.concatenate(
                (self._n_expansions_act, new_n_expansions_act), axis=0
            )

            # act_value
            new_act_value = np.zeros(shape=(NODE_ARRAY_INCREASE_SIZE, self.n_actions))
            self._act_value = np.concatenate((self._act_value, new_act_value), axis=0)

            # prev_joint_actions
            new_prev_joint_actions = np.zeros(
                shape=(NODE_ARRAY_INCREASE_SIZE, *self._prev_joint_actions.shape[1:]),
                dtype=self._prev_joint_actions.dtype,
            )
            self._prev_joint_actions = np.concatenate(
                (self._prev_joint_actions, new_prev_joint_actions),
                axis=0,
            )

            # prev_particle_ids
            new_prev_particle_ids = np.full(
                shape=(NODE_ARRAY_INCREASE_SIZE,),
                fill_value=-1,
                dtype=self._prev_particle_ids.dtype,
            )
            self._prev_particle_ids = np.concatenate(
                (self._prev_particle_ids, new_prev_particle_ids), axis=0
            )

        ### store state
        self._states[particle.id] = particle._state
        particle._state = None

        ### previous particle id
        prev_p = particle.previous_particle
        if prev_p is not None and prev_p.id is not None:
            self._prev_particle_ids[particle.id] = prev_p.id

        ### store previous actions
        assert len(particle.joint_action_history) > 0
        previous_joint_action = particle.joint_action_history[-1]
        for actor_id, actor in enumerate(self.ipomdp.agents):
            self._prev_joint_actions[particle.id, actor_id] = previous_joint_action[
                actor
            ]

    def create_initial_particles(
        self,
        n_particles: int,
        other_agents_forest_group: dict[Agent, ForestGroup],
    ) -> None:
        """
        Create particles corresponding to initial beliefs.

        Keyword arguments:
        n_particles: the number of particles to initialise the node with
        other_agents_forest_group: a dictionary mapping an agent to a group of forests
                                   used to model that agent
        """
        # sample initial states
        initial_states, other_agent_ipomdps = self.ipomdp.initial_belief(
            n_states=n_particles
        )

        # create particles
        particles = [
            Particle(state=state, joint_action_history=(), node=self)
            for state in initial_states
        ]

        # assign models and beliefs to other agents
        for p, p_other_agent_ipomdps in zip(
            particles, other_agent_ipomdps, strict=True
        ):
            for other_agent, other_agent_ipomdp in p_other_agent_ipomdps.items():
                # no need to do anything if other agent is not modelled rationally
                if other_agent_ipomdp is None:
                    continue

                # store the forest corresponding to the I-POMDP used to model the
                # other agent
                p.other_agents_forest[other_agent] = other_agents_forest_group[
                    other_agent
                ].get_forest(other_agent_ipomdp)

                ### initialise uniform belief
                # create weights
                particle_weights = np.array(n_particles * (1 / n_particles,))

                # store weights in sparse format
                p.other_agents_belief[other_agent] = encode_belief(particle_weights)

        # add particles to the node
        for p in particles:
            self.add_particle(p)

    def __repr__(self) -> str:
        return f"Node({self.agent_action_history}, {len(self.particles)} particles)"

    def weight_particles(self, observation: Observation) -> None:
        """
        Weights each particle in the node according to how likely they are under the
        given observation.

        Keyword arguments:
        observation: the observation to use for weighting the particles
        """
        # if there are no particles to weight, weights are an empty array
        if self.n_particles == 0:
            self.belief = np.array([])
            self.belief_observation = observation
            return

        ### find prior weights
        #   check that every particle knows its preceding particle
        assert -1 not in self.prev_particle_ids

        #   we should have a parent node, since this method is not applied to root nodes
        assert self.parent_node is not None

        #   now we can get an array of prior weights, each corresponding to a particle
        #   in this node
        prior_weights = self.parent_node.belief[self.prev_particle_ids]

        # if the prior is already diverged, no need to calculate further
        if prior_weights.sum() == 0:
            self.belief = prior_weights
            self.belief_observation = observation
            return

        # calculate weights
        weights = self.ipomdp.prob_observation(
            observation=observation,
            states=self.states,
            prev_joint_actions=self.prev_joint_actions,
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

        if self.n_particles == 0 or self.belief.sum() == 0:
            return

        self.resample_counts = systematic_resample(
            weights=self.belief,
            r=self.random.random(),
            size=self.n_particles,
        )

    def update_other_agents_beliefs(self):
        """
        Creates beliefs over lower-level interactive states in each particle in this
        node by updating prior beliefs.

        Assumes that
        - particles in self still have a reference to previous particles
        - while self corresponds to time t+1, the root nodes in the
          child forests still correspond to time t
        """
        for particle in self.particles:
            assert particle.previous_particle is not None

            # store the frames for other agents, these are inherited from the ancestor
            particle.other_agents_forest = (
                particle.previous_particle.other_agents_forest
            )

            # weight the particles in the lower forests using the ancestor of particle.
            # This returns a dictionary of the nodes where the weights have been
            # initialised. Thus, if an agent is not modelled rationally, it will not be
            # included in the dictionary.
            other_agent_nodes = particle.initialise_child_forest_beliefs()

            # TODO: I think this is wrong -- other_agent_node corresponds to the time
            # t root nodes, not the time t+1 nodes for which we want to generate
            # beliefs.

            for other_agent, other_agent_node in other_agent_nodes.items():
                if other_agent_node is None:
                    # # Couldn't find the corresponding node.
                    # # This likely means the number of simulations done per tree was too
                    # # low.
                    # # We will create a new empty node (which will never be simulated)

                    # # find parent of new node (it will always exist)
                    # lower_node_parent = forest.get_matching_lower_node(
                    #     particle=particle.previous_particle, agent=other_agent
                    # )

                    # lower_node_agent_action_history = particle.agent_action_history(
                    #     other_agent
                    # )
                    # prev_agent_action = lower_node_agent_action_history[-1]

                    # assert prev_agent_action not in lower_node_parent.child_nodes

                    # # create new node
                    # lower_node = Node(
                    #     forest=lower_node_parent.tree,
                    #     agent_action_history=lower_node_agent_action_history,
                    #     frame=particle.other_agent_frames[other_agent.id],
                    #     parent_node=lower_node_parent,
                    # )
                    # lower_node_parent.child_nodes[prev_agent_action] = lower_node

                    # # save an empty belief to particle
                    # particle.lower_particle_dist[other_agent.id] = (
                    #     np.array([], dtype=np.bool_),
                    #     np.array([], dtype=WEIGHT_STORAGE_DTYPE),
                    # )

                    particle.other_agents_belief[other_agent] = None

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

                if other_agent_node.n_particles == 0:
                    # the target node has no particles so there is nothing to weight
                    particle.other_agents_belief[other_agent] = encode_belief(
                        np.array([])
                    )

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
                other_agent_obs = other_agent_node.ipomdp.sample_observation(
                    state=particle.state,
                    prev_joint_action=particle.joint_action_history[-1],
                )

                # resample particles in the parent node of other_agent_node
                assert other_agent_node.parent_node is not None
                other_agent_node.parent_node.resample_particles()

                # find updated weights
                other_agent_node.weight_particles(observation=other_agent_obs)

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

                # save weights
                particle.other_agents_belief[other_agent] = encode_belief(
                    other_agent_node.belief
                )

    def tree_policy(
        self,
        explore: bool,
        softargmax: bool = False,
        exploration_coef: float | None = None,
        softargmax_coef: float | None = None,
        simulated_forest: Forest | None = None,
    ) -> tuple[Action, bool]:
        """
        Choose an action according to the MCTS (Monte Carlo Tree Search)
        tree policy.

        Keyword arguments:
        explore: whether to include the exploration term when choosing the
                 action to take, and whether to choose an unvisited action if
                 there is one. This is set to false when this tree is used to
                 choose an action when expanding the parent tree (at which
                 point this tree is already fully expanded).
        softargmax: whether to use a softargmax function to weight actions and
                    choose an action using these weights. This only has
                    an effect if explore is False.
        simulated_forest: the forest that is asking self for an action. Only used for
                          logging query success rate.

        Return value:
        A (agent_action, action_unvisited) tuple where action_unvisited
        indicates whether the given action is unvisited under the current
        belief (thus indicating that a rollout policy should be used next).
        """
        assert not (explore and softargmax)

        actions = self.ipomdp.possible_actions(agent=self.ipomdp.owner)

        if softargmax:
            assert softargmax_coef is not None

            act_probs, status_code = calculate_action_probabilities(
                belief=self.belief,
                n_expansions=self.n_expansions,
                n_expansions_act=self.n_expansions_act,
                act_value=self.act_value,
                softargmax_coef=softargmax_coef,
            )

            # log the successfulness of the event
            # model.add_log_event(
            #     event_type=status_code, event_data=(simulated_forest.signature,)
            # )

            if status_code in (12, 13):
                # determining the action was unsuccessful, so we use the default model
                choice = self.ipomdp.agent_models(self.ipomdp.owner)[0].act()
            else:
                choice = self.random.choices(
                    actions, weights=cast(Sequence[float], act_probs)
                )[0]

            return choice, False

        assert exploration_coef is not None

        # calculate action qualities
        Q = calculate_action_qualities(
            belief=self.belief,
            n_expansions=self.n_expansions,
            n_expansions_act=self.n_expansions_act,
            act_value=self.act_value,
            explore=explore,
            exploration_coef=exploration_coef,
        )

        # find actions with the highest quality and choose one randomly
        # unexpanded actions have a quality np.infty
        max_q = Q.max()
        max_action_i = (Q == max_q).nonzero()[0]
        if len(max_action_i) > 0:
            max_action_i = self.random.choice(max_action_i)

        unexpanded_action = np.isinf(max_q)
        return actions[max_action_i], unexpanded_action


class Particle:
    """
    A particle consists of
    - a state
    - a joint action history
    - a forest (corresponding to an I-POMDP) for each other agent (except for the ones
      that are not modelled rationally)
    - a distribution for particles for each other agent in the forests one level below
      (except for the ones that are not modelled rationally)
    - number of times the particle has been expanded with each of the possible actions
    - for each possible action, the value of choosing the next agent action and then
      continuing optimally afterwards
    - a reference to the previous particle (empty for particles in a root node)
    """

    # pre-define names of attributes to reduce memory usage
    __slots__ = (
        "node",
        "previous_particle",
        "id",
        "_state",
        "joint_action_history",
        "other_agents_belief",
        "other_agents_forest",
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

        # the id of the particle is set by the Node when the particle is added to it
        self.id: int | None = None

        # properties of the particle
        self._state: State | None = state
        self.joint_action_history = joint_action_history
        self.other_agents_forest: dict[Agent, Forest | None] = {}
        self.other_agents_belief: dict[Agent, SparseBelief | None] = {}

        # this keeps track of joint actions that have been used to propagate this particle
        self.propagated_actions: set[JointAction] = set()

    def update_previous_particle_id(self, new_id: int) -> None:
        self.node._prev_particle_ids[self.id] = new_id

    @property
    def state(self) -> State:
        return self._state if self._state is not None else self.node.states[self.id]

    @state.setter
    def state(self, new_state: State) -> None:
        self._state = new_state

    @property
    def weight(self) -> float:
        assert self.added_to_node
        return cast(float, self.node.belief[self.id])

    @property
    def added_to_node(self) -> bool:
        return self.id is not None

    @property
    def n_expansions(self) -> int:
        return (
            0 if not self.added_to_node else cast(int, self.node.n_expansions[self.id])
        )

    def n_expansions_act(self, agent_action: Action):
        if not self.added_to_node:
            return 0

        action_index = self.node.ipomdp.possible_actions(self.node.ipomdp.owner).index(
            agent_action
        )
        return self.node.n_expansions_act[self.id, action_index]

    def act_value(self, agent_action: Action):
        if not self.added_to_node:
            return 0

        action_index = self.node.ipomdp.possible_actions(self.node.ipomdp.owner).index(
            agent_action
        )
        return self.node.act_value[self.id, action_index]

    def agent_action_history(self, agent: Agent) -> AgentActionHistory:
        """
        Returns the agent action history of the given agent, extracted from the
        joint action history stored in the particle.
        """
        return joint_to_agent_action_history(self.joint_action_history, agent=agent)

    def has_been_propagated_with(self, joint_action: JointAction):
        """
        Checks whether this particle has been propagated with the given action before.
        """
        return joint_action in self.propagated_actions

    def add_expansion(
        self, joint_action: JointAction, value: float, next_particle: Particle
    ) -> None:
        """
        Add information about an expansion performed starting from the particle.

        Particle should be added to the node before this can be called.

        Keyword arguments:
        joint_action: joint action used to propagate the particle
        value: value received for taking this action and continuing optimally afterwards
        next_particle: the particle that resulted from taking joint_action from self
        """
        assert self.added_to_node  # TODO: kind of redundant
        assert self.id is not None  # required here to make type checker happy

        ipomdp = self.node.ipomdp

        # tell next particle about our id
        if next_particle.added_to_node:
            next_particle.update_previous_particle_id(self.id)

        # store the action (used to check if noise needs to be added)
        self.propagated_actions.add(joint_action)

        # if agent didn't act, there's no further information to store
        if joint_action[ipomdp.owner] == ipomdp.no_turn_action:
            return

        # find index of action
        agent_action = joint_action[ipomdp.owner]
        action_index = ipomdp.possible_actions(ipomdp.owner).index(agent_action)

        # current information for agent_action
        prev_n_expansions = self.n_expansions_act(agent_action)
        prev_value = self.act_value(agent_action)

        # calculate new average value based on previous average and the new value
        new_value = prev_value + (value - prev_value) / (prev_n_expansions + 1)

        # update information
        self.node.n_expansions[self.id] += 1
        self.node.n_expansions_act[self.id, action_index] += 1
        self.node.act_value[self.id, action_index] = new_value

    def ancestor_particle(self):
        """
        Returns the oldest ancestor of the particle, i.e. the particle in one of the
        root nodes where the simulation ending in self started from.
        """
        particle = self

        while particle.previous_particle is not None:
            particle = particle.previous_particle

        return particle

    def get_agent_child_forest_node(self, agent: Agent) -> tuple[Forest, Node | None]:
        """
        Use the joint action history and forest stored in the particle to find the
        matching node in the forest of the given agent at the level below.

        Node may be None if it is not found.
        """
        # find agent action history based on joint action history stored in particle
        agent_action_history = self.agent_action_history(agent)

        # find child forest
        # TODO: might need to add .ancestor_particle() here
        agent_forest = self.other_agents_forest[agent]
        assert agent_forest is not None

        # find node in child forest
        node = agent_forest.get_node(agent_action_history=agent_action_history)

        return agent_forest, node

    def initialise_child_forest_beliefs(self) -> dict[Agent, Node]:
        """
        Initialise the weights in the immediate child forests of the forest the \
        particle is in using the weights stored in it (or its oldest ancestor in \
        case it itself is not in a root node).

        Returns:
        A dictionary pointing to the nodes of other agents in the forests one level 
        below. These are the nodes in which beliefs were initialised. The keys are 
        the agents.
        """
        # find the particle which holds weights for particles in the root nodes of
        # lower level forests. If start_particle is in a root node, then it is the
        # same as ancestor_particle
        ancestor_particle = self.ancestor_particle()

        # find the nodes corresponding to ancestor particle in the forests of other
        # agents on the level below.
        # we cast to convince type checker that returned nodes are never None.
        other_agent_nodes = {
            other_agent: cast(
                Node,
                ancestor_particle.get_agent_child_forest_node(agent=other_agent)[1],
            )
            for other_agent, other_agent_forest in self.other_agents_forest.items()
            if other_agent_forest is not None
        }

        ### initialise beliefs in the nodes
        for other_agent, other_agent_node in other_agent_nodes.items():
            # check that other_agent has weights stored for its forest
            other_agent_belief = ancestor_particle.other_agents_belief[other_agent]
            assert other_agent_belief is not None

            # assign belief to the node
            other_agent_node.belief = decode_belief(other_agent_belief)

        return other_agent_nodes

    def add_noise(self, random: Random):
        """
        Add noise to the state of the current particle.
        """
        self.state = self.node.ipomdp.add_noise_to_state(
            state=self.state,
            random=random,
        )

    def __repr__(self) -> str:
        return (
            "Particle("
            + f"{self.joint_action_history}, {self.n_expansions} expansions, "
            + f"weight {self.weight})"
        )
