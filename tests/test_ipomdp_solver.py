import unittest

import numpy as np
from model import growth
from tests import helpers


class TestIPOMDPSolver(unittest.TestCase):

    def test_forest_structure_1(self):
        
        # create model with two agents and level 2 reasoning
        model = helpers.create_small_universe(n_agents=2, reasoning_level=2)
        agent_0, agent_1 = model.agents
        
        ### check the trees of agent 0
        correct_trees = {(agent_0,), 
                         (agent_0, agent_1), 
                         (agent_0, agent_1, agent_0)}
        actual_trees = set(agent_0.forest.trees.keys())
        self.assertSetEqual(correct_trees, actual_trees)
        
        ### check the trees of agent 1
        correct_trees = {(agent_1,),
                         (agent_1, agent_0),
                         (agent_1, agent_0, agent_1)}
        actual_trees = set(agent_1.forest.trees.keys())
        self.assertSetEqual(correct_trees, actual_trees)

    def test_forest_structure_2(self):
        
        # create model with three agents and level 2 reasoning
        model = helpers.create_small_universe(n_agents=3, reasoning_level=2)
        agent_0, agent_1, agent_2 = model.agents
        
        ### check the trees of agent 0
        correct_trees = {(agent_0,), 
                         (agent_0, agent_1), 
                         (agent_0, agent_1, agent_0),
                         (agent_0, agent_1, agent_2),
                         (agent_0, agent_2),
                         (agent_0, agent_2, agent_0),
                         (agent_0, agent_2, agent_1)}
        actual_trees = set(agent_0.forest.trees.keys())
        self.assertSetEqual(correct_trees, actual_trees)
        
        ### check the trees of agent 1
        correct_trees = {(agent_1,),
                         (agent_1, agent_0),
                         (agent_1, agent_0, agent_1),
                         (agent_1, agent_0, agent_2),
                         (agent_1, agent_2),
                         (agent_1, agent_2, agent_0),
                         (agent_1, agent_2, agent_1)}
        actual_trees = set(agent_1.forest.trees.keys())
        self.assertSetEqual(correct_trees, actual_trees)

        ### check the trees of agent 2
        correct_trees = {(agent_2,),
                         (agent_2, agent_0),
                         (agent_2, agent_0, agent_1),
                         (agent_2, agent_0, agent_2),
                         (agent_2, agent_1),
                         (agent_2, agent_1, agent_0),
                         (agent_2, agent_1, agent_2)}
        actual_trees = set(agent_2.forest.trees.keys())
        self.assertSetEqual(correct_trees, actual_trees)

    def test_root_node_initial_belief(self):

        # create model with two agents and level 2 reasoning
        model = helpers.create_small_universe(n_agents=2, reasoning_level=2, 
                                              agent_growth=growth.sigmoid_growth,
                                              n_root_belief_samples=5)
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
            self.assertEqual(len(node.belief), 5)

            # check the shape of samples
            for particle in node.belief:
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
            self.assertEqual(len(node.belief), 5)

            # check the shape of samples
            for particle in node.belief:
                self.assertIsInstance(particle.state, np.ndarray)
                correct_shape = (2, 4)
                self.assertEqual(correct_shape, particle.state.shape)
