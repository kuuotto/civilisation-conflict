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
            self.assertIn((), tree.root_nodes)
            node = tree.root_nodes[()]

            # check that belief is initiated
            self.assertIsInstance(node.belief, np.ndarray)

            # check the shape of belief array
            correct_shape = (5, 2, 4)
            self.assertEqual(correct_shape, node.belief.shape)

        ### check trees for agent 1
        trees_to_check = ((a1,), (a1, a0), (a1, a0, a1))

        for tree_signature in trees_to_check:
            # check that the tree exists
            self.assertIn(tree_signature, a1.forest.trees)
            tree = a1.forest.trees[tree_signature]

            # check that the tree has a root node
            self.assertIn((), tree.root_nodes)
            node = tree.root_nodes[()]

            # check the shape of belief array
            correct_shape = (5, 2, 4)
            self.assertEqual(correct_shape, node.belief.shape)
