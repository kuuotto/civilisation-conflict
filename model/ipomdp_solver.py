# %%

# avoids having to give type annotations as strings
from __future__ import annotations

from random import Random

from typing import (
    TYPE_CHECKING,
    Generator,
    Generic,
    Optional,
    Mapping,
    Sequence,
)
from numpy.typing import NDArray

import numpy as np
from numba import njit

from envs.ipomdp import (
    IPOMDP,
    Agent,
    State,
    Action,
    JointAction,
    Observation,
    Reward,
    ActivationSchedule,
    Model,
    FixedPolicyModel,
)

JointActionHistory = tuple[JointAction, ...]
AgentActionHistory = tuple[Action, ...]
ForestSignature = tuple[Agent, ...]
Belief = NDArray[np.float_]


# parameters which can be adjusted to change momory consumption
WEIGHT_STORAGE_DTYPE = np.float32
NODE_ARRAY_INIT_SIZE = 5
NODE_ARRAY_INCREASE_SIZE = 50


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
        ipomdp: IPOMDP[Agent, State, Action, Observation],
        random: Random,
        n_tree_simulations: int,
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
        self.discount_epsilon = discount_epsilon
        self.debug = debug

        # store highest and lowest reasoning level
        self.owner_reasoning_level = ipomdp.owner_reasoning_level

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
            raise Exception(
                "The weights of the particles in the top-level forest root node are all 0"
            )

        ### 2. Update belief in the interactive states over the lower-level models
        new_root_node.update_lower_beliefs()

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
                other_agent_forest.update_beliefs()

    def optimal_action(self):
        """
        Return the optimal action of the owner of the I-POMDP.
        """
        return self.owner_forest.tree_policy(
            node=self.owner_forest_root_node, explore=False, softargmax=False
        )[0]

    @property
    def owner_forest(self) -> Forest:
        assert len(self.owner_forest_group.forests) == 1
        return self.owner_forest_group.forests[0]

    @property
    def owner_forest_root_node(self) -> Node:
        return self.owner_forest.root_nodes[0]

    @owner_forest_root_node.setter
    def owner_forest_root_node(self, new_node) -> None:
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
        models: Sequence[Model],
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

            assert model.owner == agent

            # create the forest. This will recursively create lower-level forests.
            forest = Forest(ipomdp=model, forest_group=self, solver=solver)

            # add it to the collection of child forests
            self.forests.append(forest)

        # calculate signature, which shows the ownership hierarchy of this forest
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
                f"{n_successful} / {self.solver.n_tree_simulations} successful simulations in {self}"
            )

    def simulate(self) -> bool:
        """
        Expand the current forest group by
        1. Randomly choosing particles top-down to determine which forest and root node
           to start expanding from. This is also how we find weights for the particles
           in the chosen root node.
        2. Sampling a belief particle from the chosen root node.
        3. Using a version of MCTS to traverse the tree, at each choosing others'
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
        node = self.solver.owner_forest_root_node
        particle_weights = node.belief

        for next_agent in self.signature[1:]:
            # choose random particle given weights
            particle = self.solver.random.choices(
                node.particles, weights=particle_weights
            )[0]

            # find matching node
            node = node.forest.get_matching_child_forest_node(
                particle=particle, agent=next_agent
            )

            # get weights for node based on the particle
            mask, values = particle.other_agents_belief[next_agent]
            particle_weights: Belief = np.zeros(len(mask))
            particle_weights[mask] = values

            # if belief is diverged, we cannot simulate
            if particle_weights.sum() == 0:
                return False

        forest = node.forest

        assert forest in self.forests
        assert node in forest.root_nodes

        # save weights to particles
        node.belief = particle_weights

        # 2. Sample a particle to start simulating from
        start_particle = self.solver.random.choices(
            node.particles, weights=particle_weights
        )[0]

        # weight particles in the immediate child trees
        other_agent_nodes = forest.initialise_simulation(particle=start_particle)

        # 3. - 6. simulate starting from particle
        forest.simulate_from_particle(
            particle=start_particle, other_agent_nodes=other_agent_nodes
        )

        return True

    def is_empty(self) -> bool:
        return len(self.forests) == 0


class Forest(Generic[Agent, State]):
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
        parent_forest: the parent of this forest
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
        )
        self.root_nodes = [root_node]

        # create child forests
        self.other_agent_forest_group = {
            agent: ForestGroup(
                parent_forest=self,
                models=ipomdp.agent_models(agent),
                solver=solver,
                agent=agent,
            )
            for agent in ipomdp.other_agents
        }

        # create initial particles in the root node
        root_node.create_initial_particles()

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
        Find node in the tree matching the given agent action history.

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

    def initialise_simulation(self, particle: Particle) -> dict[Agent, Node]:
        """
        Initialise the weights in the immediate child forests of the forest the \
        particle is in using the weights stored in it (or its oldest ancestor in \
        case it itself is not in a root node).

        Keyword arguments:
        particle: The particle to use as a base for weighting the particles in the \
                  lower tree.

        Returns:
        A dictionary pointing to the nodes of other agents in the trees one level below.
        These are the nodes in which beliefs were initialised. The keys are the agents.
        """
        assert particle.node.forest is self

        # find the particle which holds weights for particles in the root nodes of
        # lower level trees. If start_particle is in a root node, then it is the
        # same as ancestor_particle
        ancestor_particle = particle.ancestor_particle()

        # find the nodes corresponding to ancestor particle in the forests of other
        # agents on the level below
        other_agent_nodes = {
            other_agent: self.get_matching_child_forest_node(
                particle=ancestor_particle, agent=other_agent
            )
            for other_agent in self.ipomdp.other_agents
            if not self.other_agent_forest_group[other_agent].is_empty()
        }

        for other_agent, other_agent_node in other_agent_nodes.items():
            # check that other_agent has weights stored for its tree
            assert ancestor_particle.other_agents_belief[other_agent] is not None

            # extract and store weights
            mask, values = ancestor_particle.other_agents_belief[other_agent]
            other_agent_belief = np.zeros(len(mask))
            other_agent_belief[mask] = values
            other_agent_node.belief = other_agent_belief

        return other_agent_nodes

    def simulate_from_particle(
        self,
        particle: Particle,
        other_agent_nodes: Mapping[Agent, Node],
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
        other_agent_nodes: nodes corresponding to particle in the trees one level below
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
        agent = node.forest.agent

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

            # make a copy because rollout changes state in place
            start_state = particle.state.copy()
            value = ipomdp.rollout(state=start_state)

            # add the new particle (it will have no expansions)
            if not discount_horizon_reached:
                node.add_particle(particle)

            # end recursion
            return value

        ### 1. choose an action from each agent according to the activation schedule
        action_unvisited = False
        action_ = []

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
                actor_action, action_unvisited = self.tree_policy(
                    node=node, explore=True
                )

            else:
                # someone else is acting.

                actor_is_rational = isinstance(ipomdp.agent_models(actor), IPOMDP)

                # find the node in the forest of the other agent
                lower_node = other_agent_nodes.get(actor)

                # if we don't model the other agent as rational (we don't have a
                # corresponding lower level forest modelling them) then their action is
                # chosen according to the level 0 model
                if lower_node is None:
                    # use the default policy to choose the action of others
                    default_policy = ipomdp.agent_models(actor)
                    assert len(default_policy) == 1
                    default_policy = default_policy[0]
                    assert isinstance(default_policy, FixedPolicyModel)
                    actor_action = default_policy.act()

                else:
                    ### use the tree below to choose the action

                    # get matching node
                    lower_node = other_agent_nodes[actor]

                    # if node is not found, choose with default policy
                    if lower_node is None:
                        solver.add_log_event(
                            event_type=11, event_data=(self.signature,)
                        )
                        actor_action = ipomdp.level0_opponent_policy(
                            agent=actor, model=model
                        )

                    else:
                        # determine action from the other agent's tree
                        actor_action, _ = lower_node.forest.tree_policy(
                            node=lower_node,
                            explore=False,
                            softargmax=True,
                            simulated_tree=self,
                        )

            # add to joint action
            joint_action[actor] = actor_action

        # package action
        action_ = tuple(action_)
        agent_action = action_[self.agent.id]

        ### 2. Propagate state
        propagated_state, rewards = ipomdp.transition(
            state=particle.state, action_=action_, model=model, frame=node.frame
        )
        agent_reward = rewards[self.agent.id]

        # create a new node if necessary
        if agent_action not in node.child_nodes:
            new_agent_action_history = node.agent_action_history + (agent_action,)
            new_node = Node(
                forest=self,
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

        ### 3. Update the belief in the current tree
        # if the action is unvisited, creating weights is not necessary as we perform
        # a rollout from it on the next recursion
        if not action_unvisited:
            # first resample the belief in the current node
            node.resample_particles()

            # generate an observation
            agent_obs = ipomdp.sample_observation(
                state=propagated_state,
                action=action_,
                observer=self.agent,
                model=model,
            )

            # weight particles in the next node
            next_node.weight_particles(agent_obs)

        ### 4. Generate observations for other agents and update beliefs about
        ### interactive states at the level below
        next_other_agent_nodes = None

        if not action_unvisited and self.level > 0:
            # find the nodes in the trees one level below for the next time step
            next_other_agent_nodes = {
                other_agent: None
                if other_agent_node is None
                else other_agent_node.child_nodes.get(action_[other_agent.id])
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
                    observer=other_agent,
                    model=model,
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
        value = agent_reward + model.discount_factor * future_value
        particle.add_expansion(
            action_=action_, value=value, next_particle=next_particle
        )

        # clean up beliefs to save memory
        if node != self.forest.owner_forest_root_node:
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
        Update beliefs in each root node of this forest. This assumes that the parent
        forest has already been updated.
        """

        ### 1. Find new root nodes
        root_nodes = tuple(
            child_node
            for node in self.root_nodes
            for child_node in node.child_nodes.values()
        )
        self.root_nodes: list[Node] = []

        ### 2. Prune new root nodes that are no longer needed (i.e. will never be
        ###    expanded because the agent action histories they represent are no
        ###    longer present in particles on the level above)
        for root_node in root_nodes:
            # check all parent tree root nodes for matching particles
            parent_particles = (
                p
                for parent_root_node in self.parent_forest.root_nodes
                for p in parent_root_node.particles
            )
            for p in parent_particles:
                if (
                    p.agent_action_history(self.agent) == root_node.agent_action_history
                    and p.other_agent_frames[self.agent.id] is self
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
        for child_tree in self.solver.child_forests(
            parent_forest=self, include_parent=False
        ):
            child_tree.update_beliefs()

    def get_matching_child_forest_node(self, particle: Particle, agent: Agent) -> Node:
        """
        Given a particle, use the joint action history and forest stored in it
        to find the matching node in the forest of the given agent at the level below.
        """
        assert particle.node.forest is self

        # find agent action history based on joint action history stored in particle
        agent_action_history = particle.agent_action_history(agent)

        # find child forest
        # TODO: might need to add .ancestor_particle() here
        agent_forest = particle.other_agents_forest[agent]

        # find node in child forest
        node = agent_forest.get_node(agent_action_history=agent_action_history)

        return node

    def tree_policy(
        self,
        node: Node,
        explore: bool,
        softargmax: bool = False,
        simulated_tree: Forest = None,
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
        model: universe.Universe = node.forest.solver.owner.model

        assert node.forest == self
        assert not (explore and softargmax)

        actions = node.forest.agent.possible_actions()

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
                    agent=node.forest.agent, model=model
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
        forest: Forest,
        agent_action_history: AgentActionHistory,
        parent_node: Node | None,
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
        self.forest = forest
        self.agent_action_history = agent_action_history

        # relation of node to other nodes in the tree
        self.parent_node = parent_node
        self.child_nodes: dict[Action, Node] = dict()

        # particles is a list of all particles stored in the node
        self.particles: list[Particle] = []
        # belief contains weights for particles
        self._belief = np.array([])
        # contains the counts of particles if the particles have been resampled
        self.resample_counts = None
        # stores the observation used to generate the belief stored in ‘belief’.
        # Included for debugging purposes.
        self.belief_observation = None

        self.n_particles = 0
        self.array_size = NODE_ARRAY_INIT_SIZE  # current size of arrays
        self.n_actions = len(forest.agent.possible_actions())
        model: universe.Universe = forest.solver.owner.model

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
        # - ids of targets of attacks in the previous action
        self._prev_action_attack_target_ids = np.full(
            shape=(self.array_size, model.n_agents), fill_value=-1, dtype=np.int8
        )
        # - ids of previous particles
        self._prev_particle_ids = np.full(
            shape=(self.array_size,), fill_value=-1, dtype=np.int32
        )

    @property
    def belief(self) -> Belief:
        if self.is_resampled:
            # when belief is resampled, the weights of the chosen particles are uniform
            # n_particles = self.resample_counts.sum()
            n_particles = self.n_particles
            return (1 / n_particles) * (self.resample_counts > 0)

        assert len(self._belief) == self.n_particles
        return self._belief

    @belief.setter
    def belief(self, new_belief: Belief) -> None:
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
    def prev_action_attack_target_ids(self) -> NDArray:
        return self._prev_action_attack_target_ids[: self.n_particles]

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
        n_agents = self.forest.solver.owner.model.n_agents

        # give particle its index in the arrays
        particle.id = self.n_particles

        # add particle
        self.particles.append(particle)
        self.n_particles += 1

        # increase array sizes if necessary
        if self.n_particles > self.array_size:
            # states
            new_states = np.zeros(
                shape=(NODE_ARRAY_INCREASE_SIZE, *self._states.shape[1:])
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

            # prev_action_attack_target_ids
            new_prev_action_attack_target_ids = np.full(
                shape=(NODE_ARRAY_INCREASE_SIZE, n_agents),
                fill_value=-1,
                dtype=self._prev_action_attack_target_ids.dtype,
            )
            self._prev_action_attack_target_ids = np.concatenate(
                (
                    self._prev_action_attack_target_ids,
                    new_prev_action_attack_target_ids,
                ),
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

            # increase size of array size variable to match the new array lengths
            self.array_size += NODE_ARRAY_INCREASE_SIZE

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
        for actor_id, actor_action in enumerate(particle.joint_action_history[-1]):
            # if action is an attack, store the id of the target
            if not isinstance(actor_action, int):
                self._prev_action_attack_target_ids[
                    particle.id, actor_id
                ] = actor_action.id

    def create_initial_particles(self) -> None:
        """
        Create particles corresponding to initial beliefs.
        """
        model = self.forest.solver.owner.model
        n_particles = model.n_root_belief_samples

        # determine if we are in a top or bottom level tree
        in_top_level_tree = self.forest.level == self.forest.solver.owner.level
        in_bottom_level_tree = self.forest.level == 0

        # sample initial states
        # If the tree is the top-level tree, the agent's state is used to
        # constrain the initial belief as the agent is certain about its own state.
        if model.initial_belief == "uniform":
            initial_particle_states = ipomdp.uniform_initial_belief(
                n_samples=n_particles,
                model=model,
                agent=self.forest.agent if in_top_level_tree else None,
            )
        elif model.initial_belief == "surpass_scenario":
            initial_particle_states = ipomdp.surpass_scenario_initial_belief(
                n_samples=n_particles,
                level=self.forest.level,
                model=model,
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
                    if ag != self.forest.agent
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
                for other_agent in (
                    ag for ag in model.agents if ag != self.forest.agent
                ):
                    # create uniform weights
                    particle_weights = np.array(n_particles * (1 / n_particles,))

                    # store created belief in sparse format
                    weights_mask = particle_weights > 0
                    weights_values = particle_weights[weights_mask].astype(
                        WEIGHT_STORAGE_DTYPE
                    )
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

        model = self.forest.solver.owner.model

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
            attack_target_ids=self.prev_action_attack_target_ids,
            observer=self.forest.agent,
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

        model = self.forest.solver.owner.model

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
        if self.forest.level == 0:
            return

        forest = self.forest.solver
        model: universe.Universe = forest.owner.model

        other_agents = tuple(ag for ag in model.agents if ag != self.forest.agent)

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
                        forest=lower_node_parent.tree,
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
                    observer=other_agent,
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
    - a distribution for particles for each agent in the forests one level below (empty
      for particles in the lowest level forest)
    - a forest (corresponding to an I-POMDP) for each other agent (empty for particles
      in the lowest level forest)
    - number of times the particle has been expanded with each of the possible actions
    - for each possible action, the value of choosing the next agent action and then
      continuing optimally afterwards
    - a weight to represent belief (changes and can be empty)
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
        self.id = None  # this is set by the Node when the particle is added to it

        # properties of the particle
        self._state = state
        self.joint_action_history = joint_action_history
        self.other_agents_forest: dict[Agent, Forest | None] = {}
        self.other_agents_belief: dict[Agent, tuple[NDArray, NDArray] | None] = {}

        # this keeps track of others' actions that have been used to propagate
        # this particle
        self.propagated_actions: set[Action] = set()

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

        agent = self.node.forest.agent
        action_index = agent.possible_actions().index(agent_action)
        return self.node.n_expansions_act[self.id, action_index]

    def act_value(self, agent_action: AgentAction):
        if not self.added_to_node:
            return 0

        agent = self.node.forest.agent
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
        return action in self.propagated_actions

    def add_expansion(
        self, action_: Action, value: float, next_particle: Particle
    ) -> None:
        """
        Add information about an expansion performed starting from the particle.

        Particle should be added to the node before this can be called.

        Keyword arguments:
        action_: action used to propagate the particle
        value: value received for taking this action and continuing optimally afterwards
        next_particle: the particle that resulted from taking action from self
        """
        agent = self.node.forest.agent

        assert self.added_to_node

        # tell next particle about our id
        if next_particle.added_to_node:
            next_particle.update_previous_particle_id(self.id)

        # store the action (used to check if noise needs to be added)
        self.propagated_actions.add(action_)

        # if agent didn't act, there's no further information to store
        if action_[agent.id] == action.NO_TURN:
            return

        # find index of action
        agent_action = action_[agent.id]
        action_index = agent.possible_actions().index(agent_action)

        # current information for agent_action
        prev_n_expansions = self.n_expansions_act(agent_action)
        prev_value = self.act_value(agent_action)

        # calculate new average value based on previous average and the new value
        new_value = prev_value + (value - prev_value) / (prev_n_expansions + 1)

        # update information
        self.node.n_expansions[self.id] += 1
        self.node.n_expansions_act[self.id, action_index] += 1
        self.node.act_value[self.id, action_index] = new_value

    def add_noise(self):
        model = self.node.forest.solver.owner.model
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
                    state[:, 3] += model.rng.normal(
                        loc=0, scale=speed_noise_scale, size=model.n_agents
                    )
                else:
                    raise NotImplementedError(
                        f"{speed_noise_dist} is not a valid speed noise distribution"
                    )

                # make sure values stay within allowed range
                state[:, 3] = state[:, 3].clip(*speed_range)

            if (
                takeoff_time_range[0] < takeoff_time_range[1]
                and takeoff_time_noise_scale > 0
            ):
                state[:, 4] += model.rng.integers(
                    low=-takeoff_time_noise_scale,
                    high=takeoff_time_noise_scale,
                    endpoint=True,
                    size=model.n_agents,
                )

                # make sure values stay within allowed range
                state[:, 4] = state[:, 4].clip(*takeoff_time_range)
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
        model = self.node.forest.solver.owner.model
        levels = tuple(growth.tech_level(state=self.state, model=model).round(2))
        return (
            f"Particle(levels {levels}, "
            + f"{self.joint_action_history}, {self.n_expansions} expansions, "
            + f"weight {self.weight})"
        )
