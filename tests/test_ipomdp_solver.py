import unittest

import numpy as np
from model import growth, ipomdp_solver, action
from tests import helpers
from collections import Counter
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
            agent_growth=growth.sigmoid_growth,
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


class TestParticleReinvigoration(unittest.TestCase):
    def test_uniform_random_particles(self):
        model = helpers.create_small_universe(
            n_agents=3,
            agent_growth=growth.sigmoid_growth,
            agent_growth_params={
                "speed_range": (0.3, 1.0),
                "takeoff_time_range": (20, 40),
            },
            init_age_range=(10, 20),
            init_age_belief_range=(10, 20),
        )

        n_particles = 10

        # create node to add particles to
        tree = model.agents[0].forest.top_level_tree
        agent_action_history = (action.NO_TURN, action.NO_TURN, action.NO_ACTION)
        node = ipomdp_solver.Node(
            tree=tree,
            agent_action_history=agent_action_history,
            parent_node=None,
        )

        # generate particles
        random_particles = ipomdp_solver.generate_random_particles(
            n_particles=n_particles,
            node=node,
            model=model,
        )

        # check that the number of generated particles is correct
        self.assertEqual(len(random_particles), n_particles)

        # check the state in each particle
        for particle in random_particles:
            self.assertIsInstance(particle.state, np.ndarray)

            # check shapes
            self.assertEqual(particle.state.shape, (3, 4))

            # check that values are in the correct ranges
            for agent_id in range(3):
                agent_state = particle.state[agent_id]

                # check age
                allowed_age_range = (0, 20 + len(agent_action_history))
                self.assertGreaterEqual(agent_state[0], allowed_age_range[0])
                self.assertLessEqual(agent_state[0], allowed_age_range[1])

                # check visibility
                allowed_visibility_range = (0, 1)
                self.assertGreaterEqual(agent_state[1], allowed_visibility_range[0])
                self.assertLessEqual(agent_state[1], allowed_visibility_range[1])

                # check growth parameters
                allowed_speed_range = model.agent_growth_params["speed_range"]
                self.assertGreaterEqual(agent_state[2], allowed_speed_range[0])
                self.assertLessEqual(agent_state[2], allowed_speed_range[1])

                allowed_takeoff_time_range = model.agent_growth_params[
                    "takeoff_time_range"
                ]
                self.assertGreaterEqual(agent_state[3], allowed_takeoff_time_range[0])
                self.assertLessEqual(agent_state[3], allowed_takeoff_time_range[1])

        # check that there is some variance between the generated particles
        all_states = np.stack(tuple(p.state for p in random_particles), axis=0)

        # variance between particles
        variance = all_states.var(axis=0)
        self.assertTrue((variance > 0).all())


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
        n_expansions = np.array([10]).astype(np.float_)
        n_expansions_act = np.array([[2, 3, 5]]).astype(np.float_)
        act_value = np.array([[-1, -0.5, -0.1]])

        # correct action qualities
        correct_action_qualities = np.array([-1, -0.5, -0.1])

        action_qualities = ipomdp_solver.calculate_action_qualities(
            belief=belief,
            n_expansions=n_expansions,
            n_expansions_act=n_expansions_act,
            act_value=act_value,
            explore=False,
            softargmax=False,
            exploration_coef=0,
        )

        self.assertTrue((action_qualities == correct_action_qualities).all())

    def test_action_qualities_2(self):
        # calculates the action values for three particles simultaneously
        n_particles = 3
        n_actions = 3

        belief = np.array([0.4, 0.6, 0.0])
        n_expansions = np.array([10, 12, 5]).astype(np.float_)
        n_expansions_act = np.array(
            [
                [2, 3, 5],
                [6, 6, 0],
                [0, 2, 3],
            ]
        ).astype(np.float_)
        act_value = np.array(
            [
                [-1, -0.5, -0.1],
                [0, -0.5, 0],
                [0, -1.5, -0.8],
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
            softargmax=False,
            exploration_coef=0,
        )

        self.assertTrue((action_qualities == Q).all())

    def test_action_qualities_3(self):
        # testing unexplored actions
        # calculates the action values for three particles simultaneously
        n_particles = 4
        n_actions = 5

        belief = np.array([0.3, 0.3, 0.0, 0.4])
        n_expansions = np.array([1, 1, 1, 1]).astype(np.float_)
        n_expansions_act = np.array(
            [
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
            ]
        ).astype(np.float_)
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
            softargmax=False,
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

        belief = np.array([0.4, 0.6, 0.0])
        n_expansions = np.array([10, 12, 5]).astype(np.float_)
        n_expansions_act = np.array(
            [
                [2, 3, 5],
                [6, 6, 0],
                [0, 2, 3],
            ]
        ).astype(np.float_)
        act_value = np.array(
            [
                [-1, -0.5, -0.1],
                [0, -0.5, 0],
                [0, -1.5, -0.8],
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

        # this is a slightly different, but equivalent way to calculate the action
        # probabilities
        action_probabilities = np.exp((W_a / W) * np.sqrt(N))
        action_probabilities /= action_probabilities.sum()

        action_qualities = ipomdp_solver.calculate_action_qualities(
            belief=belief,
            n_expansions=n_expansions,
            n_expansions_act=n_expansions_act,
            act_value=act_value,
            explore=False,
            softargmax=True,
            exploration_coef=0,
        )

        self.assertTrue(np.allclose(action_qualities, action_probabilities))

    def test_action_qualities_5(self):
        # testing exploration term
        # calculates the action values for three particles simultaneously
        n_particles = 4
        n_actions = 4
        exploration_coef = 0.5

        belief = np.array([0.3, 0.2, 0.1, 0.4])
        n_expansions = np.array([1, 1, 1, 1]).astype(np.float_)
        n_expansions_act = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        ).astype(np.float_)
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
            softargmax=False,
            exploration_coef=exploration_coef,
        )

        self.assertTrue((action_qualities == Q).all())
