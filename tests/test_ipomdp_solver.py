import unittest

import numpy as np
from model import ipomdp_solver
from tests import helpers
import random
import math


class TestSolverForestStructure(unittest.TestCase):
    def test_forest_structure_1(self):
        # create model with two agents and level 2 reasoning
        model = helpers.create_small_universe(n_agents=2, reasoning_level=2)
        agent_0, agent_1 = model.agents

        ### check the trees of agent 0
        correct_trees = {(agent_0,), (agent_0, agent_1), (agent_0, agent_1, agent_0)}
        actual_trees = set(agent_0.forest.trees.keys())
        self.assertSetEqual(correct_trees, actual_trees)

        ### check the trees of agent 1
        correct_trees = {(agent_1,), (agent_1, agent_0), (agent_1, agent_0, agent_1)}
        actual_trees = set(agent_1.forest.trees.keys())
        self.assertSetEqual(correct_trees, actual_trees)

    def test_forest_structure_2(self):
        # create model with three agents and level 2 reasoning
        model = helpers.create_small_universe(n_agents=3, reasoning_level=2)
        agent_0, agent_1, agent_2 = model.agents

        ### check the trees of agent 0
        correct_trees = {
            (agent_0,),
            (agent_0, agent_1),
            (agent_0, agent_1, agent_0),
            (agent_0, agent_1, agent_2),
            (agent_0, agent_2),
            (agent_0, agent_2, agent_0),
            (agent_0, agent_2, agent_1),
        }
        actual_trees = set(agent_0.forest.trees.keys())
        self.assertSetEqual(correct_trees, actual_trees)

        ### check the trees of agent 1
        correct_trees = {
            (agent_1,),
            (agent_1, agent_0),
            (agent_1, agent_0, agent_1),
            (agent_1, agent_0, agent_2),
            (agent_1, agent_2),
            (agent_1, agent_2, agent_0),
            (agent_1, agent_2, agent_1),
        }
        actual_trees = set(agent_1.forest.trees.keys())
        self.assertSetEqual(correct_trees, actual_trees)

        ### check the trees of agent 2
        correct_trees = {
            (agent_2,),
            (agent_2, agent_0),
            (agent_2, agent_0, agent_1),
            (agent_2, agent_0, agent_2),
            (agent_2, agent_1),
            (agent_2, agent_1, agent_0),
            (agent_2, agent_1, agent_2),
        }
        actual_trees = set(agent_2.forest.trees.keys())
        self.assertSetEqual(correct_trees, actual_trees)


class TestSolverInitialState(unittest.TestCase):
    def test_root_node_initial_belief(self):
        # create model with two agents and level 2 reasoning
        model = helpers.create_small_universe(
            n_agents=2,
            reasoning_level=2,
            agent_growth="sigmoid",
            n_root_belief_samples=5,
        )
        a0, a1 = model.agents

        ### check trees for agent 0
        trees_to_check = ((a0,), (a0, a1), (a0, a1, a0))

        for tree_signature in trees_to_check:
            # check that the tree exists
            self.assertIn(tree_signature, a0.forest.trees)
            tree = a0.forest.trees[tree_signature]

            # check that the tree has a root node
            self.assertTrue(len(tree.root_nodes) == 1)
            node = tree.root_nodes[0]

            # check the number of particles
            self.assertEqual(len(node.particles), 5)

            # check the shape of samples
            for particle in node.particles:
                self.assertIsInstance(particle.state, np.ndarray)
                correct_shape = (2, 4)
                self.assertEqual(correct_shape, particle.state.shape)

        ### check trees for agent 1
        trees_to_check = ((a1,), (a1, a0), (a1, a0, a1))

        for tree_signature in trees_to_check:
            # check that the tree exists
            self.assertIn(tree_signature, a1.forest.trees)
            tree = a1.forest.trees[tree_signature]

            # check that the tree has a root node
            self.assertTrue(len(tree.root_nodes) == 1)
            node = tree.root_nodes[0]

            # check the number of particles
            self.assertEqual(len(node.particles), 5)

            # check the shape of samples
            for particle in node.particles:
                self.assertIsInstance(particle.state, np.ndarray)
                correct_shape = (2, 4)
                self.assertEqual(correct_shape, particle.state.shape)


class TestResampling(unittest.TestCase):
    def test_systematic_resampling_1(self):
        rng = random.Random(0)
        # sample = [0, 1, 2, 3, 4, 5]
        weights = np.array([0.0999, 0.0001, 0.2, 0.4, 0.2, 0.1])
        N = len(weights)

        # repeat resampling multiple times
        for _ in range(100):
            counts = ipomdp_solver.systematic_resample(
                weights=weights,
                r=rng.random(),
                size=N,
            )

            # check that each value was sampled between floor(Nw) and floor(NW) + 1 times
            for weight, count in zip(weights, counts):
                lower_bound = math.floor(N * weight)
                upper_bound = math.floor(N * weight) + 1

                self.assertGreaterEqual(count, lower_bound)
                self.assertLessEqual(count, upper_bound)

    def test_systematic_resampling_2(self):
        rng = random.Random(0)
        N = 10

        # repeat resampling multiple times
        for _ in range(100):
            # generate random weights
            weights = np.array([rng.random() for _ in range(N)])
            weights /= weights.sum()

            # resample
            counts = ipomdp_solver.systematic_resample(
                weights=weights,
                r=rng.random(),
                size=N,
            )

            # check that each value was sampled between floor(Nw) and floor(NW) + 1 times
            for weight, count in zip(weights, counts):
                lower_bound = math.floor(N * weight)
                upper_bound = math.floor(N * weight) + 1

                self.assertGreaterEqual(count, lower_bound)
                self.assertLessEqual(count, upper_bound)

    def test_belief_resampling(self):
        mdl = helpers.create_small_universe(
            n_agents=2,
            reasoning_level=0,
        )

        # find root node of agent 0's only tree
        agent = mdl.agents[0]
        node = agent.forest.trees[(agent,)].root_nodes[0]
        n_particles = len(node.particles)

        # assign random weights
        rng = mdl.rng
        weights = rng.random(size=n_particles)
        weights = weights / weights.sum()

        node.belief = weights

        # resample
        node.resample_particles()

        # make sure that weights of all particles are either 1/n_particles or 0
        for particle in node.particles:
            if node.resample_counts[particle.id] > 0:
                self.assertEqual(particle.weight, 1 / n_particles)
            else:
                self.assertEqual(particle.weight, 0)


class TestActionQualityCalculation(unittest.TestCase):
    def test_action_qualities_1(self):
        # tests calculating the action values with only a single particle
        belief = np.array([1.0])
        n_expansions = np.array([10]).astype(np.uint16)
        n_expansions_act = np.array([[2, 3, 5]]).astype(np.uint16)
        act_value = np.array([[-1, -0.5, -0.1]])

        # correct action qualities
        correct_action_qualities = np.array([-1, -0.5, -0.1])

        action_qualities = ipomdp_solver.calculate_action_qualities(
            belief=belief,
            n_expansions=n_expansions,
            n_expansions_act=n_expansions_act,
            act_value=act_value,
            explore=False,
            exploration_coef=0,
        )

        self.assertTrue((action_qualities == correct_action_qualities).all())

    def test_action_qualities_2(self):
        # calculates the action values for three particles simultaneously
        n_particles = 3
        n_actions = 3

        belief = np.array([0.4, 0.6, 0.0])
        n_expansions = np.array([10, 12, 5]).astype(np.uint16)
        n_expansions_act = np.array(
            [
                [2, 3, 5],
                [6, 6, 0],
                [0, 2, 3],
            ]
        ).astype(np.uint16)
        act_value = np.array(
            [
                [-1, -0.5, -0.1],
                [0, -0.5, 0],
                [0, -1.5, -0.8],
            ]
        )

        # correct action qualities
        Q = np.array(
            [
                sum(belief[p_i] * act_value[p_i, a_i] for p_i in range(n_particles))
                / sum(
                    belief[p_i]
                    for p_i in range(n_particles)
                    if n_expansions_act[p_i, a_i] > 0
                )
                for a_i in range(n_actions)
            ]
        )

        action_qualities = ipomdp_solver.calculate_action_qualities(
            belief=belief,
            n_expansions=n_expansions,
            n_expansions_act=n_expansions_act,
            act_value=act_value,
            explore=False,
            exploration_coef=0,
        )

        self.assertTrue((action_qualities == Q).all())

    def test_action_qualities_3(self):
        # testing unexplored actions
        # calculates the action values for three particles simultaneously
        n_particles = 4
        n_actions = 5

        belief = np.array([0.3, 0.3, 0.0, 0.4])
        n_expansions = np.array([1, 1, 1, 1]).astype(np.uint16)
        n_expansions_act = np.array(
            [
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
            ]
        ).astype(np.uint16)
        act_value = np.array(
            [
                [-2, 0, 0, 0, 0],
                [0, -0.25, 0, 0, 0],
                [0, 0, -0.1, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        )

        action_qualities = ipomdp_solver.calculate_action_qualities(
            belief=belief,
            n_expansions=n_expansions,
            n_expansions_act=n_expansions_act,
            act_value=act_value,
            explore=True,
            exploration_coef=0,
        )

        # unexplored action (index 4) should equal np.infty
        # action is unexplored because no particle has tried it
        self.assertTrue(np.isinf(action_qualities[4]))

        # other unexplored action (index 2) should equal np.infty
        # action is unexplored because only particle with index 2 (which has
        # weight 0) tried it
        self.assertTrue(np.isinf(action_qualities[2]))

        # other actions should not have an infinite quality
        self.assertTrue(not np.isinf(action_qualities[[0, 1, 3]]).any())

    def test_action_qualities_4(self):
        # testing softargmax
        # calculates the action values for three particles simultaneously
        n_particles = 3
        n_actions = 3
        softargmax_coef = 0.1

        belief = np.array([0.4, 0.6, 0.0])
        n_expansions = np.array([10, 12, 5]).astype(np.uint16)
        n_expansions_act = np.array(
            [
                [2, 3, 5],
                [6, 6, 0],
                [0, 2, 3],
            ]
        ).astype(np.uint16)
        act_value = np.array(
            [
                [-1, -0.5, -0.1],
                [0, -0.5, 0],
                [0, -1.5, -0.8],
            ]
        )

        # correct action qualities
        N = sum(n_expansions[p_i] for p_i in range(n_particles) if belief[p_i] > 0)
        Q = np.array(
            [
                sum(belief[p_i] * act_value[p_i, a_i] for p_i in range(n_particles))
                / sum(
                    belief[p_i]
                    for p_i in range(n_particles)
                    if n_expansions_act[p_i, a_i] > 0
                )
                for a_i in range(n_actions)
            ]
        )

        # calculate the correct (cumulative and unnormalised) action probabilities
        action_probabilities_correct = np.exp(Q / (softargmax_coef * (1 / np.sqrt(N))))
        action_probabilities_correct /= action_probabilities_correct.sum()

        (
            action_probabilities,
            status_code,
        ) = ipomdp_solver.calculate_action_probabilities(
            belief=belief,
            n_expansions=n_expansions,
            n_expansions_act=n_expansions_act,
            act_value=act_value,
            softargmax_coef=softargmax_coef,
        )

        # check that the calculated action probabilities are correct
        self.assertTrue(np.allclose(action_probabilities, action_probabilities_correct))

        # check that the returned status code is 10, which indicates success
        self.assertEqual(status_code, 10)

    def test_action_qualities_5(self):
        # testing exploration term
        # calculates the action values for three particles simultaneously
        n_particles = 4
        n_actions = 4
        exploration_coef = 0.5

        belief = np.array([0.3, 0.2, 0.1, 0.4])
        n_expansions = np.array([1, 1, 1, 1]).astype(np.uint16)
        n_expansions_act = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        ).astype(np.uint16)
        act_value = np.array(
            [
                [-2, 0, 0, 0],
                [0, -0.25, 0, 0],
                [0, 0, -0.1, 0],
                [0, 0, 0, 0],
            ]
        )

        # correct action qualities
        N = sum(n_expansions[p_i] for p_i in range(n_particles) if belief[p_i] > 0)
        W = sum(belief[p_i] * n_expansions[p_i] for p_i in range(n_particles))
        W_a = np.array(
            [
                sum(
                    belief[p_i] * n_expansions_act[p_i, a_i]
                    for p_i in range(n_particles)
                )
                for a_i in range(n_actions)
            ]
        )
        N_a = (W_a / W) * N
        Q = np.array(
            [
                sum(belief[p_i] * act_value[p_i, a_i] for p_i in range(n_particles))
                / sum(
                    belief[p_i]
                    for p_i in range(n_particles)
                    if n_expansions_act[p_i, a_i] > 0
                )
                for a_i in range(n_actions)
            ]
        )

        # add exploration term
        Q += exploration_coef * np.sqrt(np.log(N) / N_a)

        action_qualities = ipomdp_solver.calculate_action_qualities(
            belief=belief,
            n_expansions=n_expansions,
            n_expansions_act=n_expansions_act,
            act_value=act_value,
            explore=True,
            exploration_coef=exploration_coef,
        )

        self.assertTrue((action_qualities == Q).all())


class TestTreeExpansion(unittest.TestCase):
    def test_tree_properties_after_planning(self):
        # tests that the forests of agents look as expected after planning

        mdl = helpers.create_small_universe(
            n_agents=2,
            reasoning_level=1,
            n_tree_simulations=100,
        )
        ag0, ag1 = mdl.agents

        # plan
        ag0.forest.plan()
        ag1.forest.plan()

        # check all nodes in agent 0's trees
        for tree in ag0.forest.trees.values():
            for root_node in tree.root_nodes:
                self._check_node(node=root_node, single_planning_step=True)

        # check all nodes in agent 1's trees
        for tree in ag1.forest.trees.values():
            for root_node in tree.root_nodes:
                self._check_node(node=root_node, single_planning_step=True)

    def test_tree_properties_after_planning_2(self):
        # tests that the forests of agents look as expected after planning and
        # belief update

        mdl = helpers.create_small_universe(
            n_agents=2,
            reasoning_level=1,
            n_tree_simulations=100,
        )
        ag0, ag1 = mdl.agents

        # plan + belief update
        mdl.step()

        # check all nodes in agent 0's trees
        for tree in ag0.forest.trees.values():
            for root_node in tree.root_nodes:
                self._check_node(node=root_node, single_planning_step=True)

        # check all nodes in agent 1's trees
        for tree in ag1.forest.trees.values():
            for root_node in tree.root_nodes:
                self._check_node(node=root_node, single_planning_step=True)

    def _check_node(self, node: ipomdp_solver.Node, single_planning_step: bool):
        """
        Checks that the given node looks like it should (see comments for more
        information on what this means).
        """
        mdl = node.tree.forest.owner.model
        agent = node.tree.agent

        n_particles = len(node.particles)
        is_root_node = node.parent_node is None

        # if node is a root node, check that its particles have stored weights
        if node.tree.level > 0 and is_root_node:
            other_agents = tuple(a for a in mdl.agents if a != agent)

            for particle in node.particles:
                for other_agent in other_agents:
                    self.assertTrue(
                        particle.lower_particle_dist[other_agent.id] is not None
                    )

                    mask, values = particle.lower_particle_dist[other_agent.id]
                    weights = np.zeros(len(mask))
                    weights[mask] = values
                    self.assertIsInstance(weights, np.ndarray)

                    # find matching node
                    lower_node = node.tree.forest.get_matching_lower_node(
                        particle=particle, agent=other_agent
                    )

                    # check that the length of the weights matches the number of particles
                    self.assertEqual(len(weights), len(lower_node.particles))

        for particle in node.particles:
            # check that particle has an index that matches its index in the list
            self.assertTrue(particle.id is not None)
            self.assertTrue(particle.added_to_node)
            self.assertEqual(node.particles[particle.id], particle)

            # check that previous particle id is correct
            if not is_root_node:
                self.assertEqual(
                    particle.previous_particle.id, node.prev_particle_ids[particle.id]
                )

            # check that particles in root nodes don't have references to previous
            # particles
            if is_root_node:
                self.assertTrue(particle.previous_particle is None)

            # check that the agent action history of particle matches that of node
            self.assertEqual(
                particle.agent_action_history(agent), node.agent_action_history
            )

        for agent_action, child_node in node.child_nodes.items():
            # check that agent action history of child node is correct
            self.assertEqual(
                child_node.agent_action_history,
                node.agent_action_history + (agent_action,),
            )

        # check that n_expansions and n_expansions_act are correct
        n_expansions = node.n_expansions
        n_expansions_act = node.n_expansions_act

        # check shapes
        self.assertEqual(n_expansions.shape, (n_particles,))
        self.assertEqual(n_expansions_act.shape, (n_particles, 3))  # 3 actions

        # check that n_expansions_act sums to n_expansions
        self.assertTrue((n_expansions_act.sum(axis=1) == n_expansions).all())

        # if this is not a root node, n_expansions should all at most 1 after
        # one step of planning
        if not is_root_node and single_planning_step:
            self.assertEqual((n_expansions > 1).sum(), 0)

        #### check child nodes recursively
        for child_node in node.child_nodes.values():
            self._check_node(node=child_node, single_planning_step=single_planning_step)
