import unittest
from tests.helpers import create_small_universe

class TestIPOMDPSolver(unittest.TestCase):

    def test_forest_structure_1(self):
        
        # create model with two agents and level 2 reasoning
        model = create_small_universe(n_agents=2, reasoning_level=2)
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
        model = create_small_universe(n_agents=3, reasoning_level=2)
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