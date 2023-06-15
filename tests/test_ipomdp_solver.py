import unittest

import numpy as np
from model import growth, ipomdp_solver, action
from tests import helpers


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
