import unittest
import numpy as np
from model import ipomdp
from tests import helpers
from model import growth

class TestIPOMDP(unittest.TestCase):

    def test_transition_1(self):

        model = helpers.create_small_universe(visibility_multiplier=0.5)

        agent_0_state = np.array([10, 0.5, 0.6, 20]) # weak agent
        agent_1_state = np.array([53, 1.0, 0.9, 30]) # strong agent
        agent_2_state = np.array([0, 1.0, 0.4, 40]) # recently destroyed agent
        agent_3_state = np.array([12, 0.9, 0.5, 12]) # growing agent
        agent_4_state = np.array([5, 0.1, 0.3, 20]) # hidden weak agent

        model_state = np.stack((agent_0_state, 
                                agent_1_state,
                                agent_2_state, 
                                agent_3_state, 
                                agent_4_state), axis=0)

        ### strong attacking a weak agent
        destroy_action = {'actor': 1, 
                          'type': 0}

        destroyed_state = ipomdp.transition(state=model_state,
                                            action=destroy_action,
                                            model=model)

        # correct agent states
        c_agent_0_state = np.array([0, 1.0, 0.6, 20]) # weak agent
        c_agent_1_state = np.array([54, 1.0, 0.9, 30]) # strong agent
        c_agent_2_state = np.array([1, 1.0, 0.4, 40]) # recently destroyed agent
        c_agent_3_state = np.array([13, 0.9, 0.5, 12]) # growing agent
        c_agent_4_state = np.array([6, 0.1, 0.3, 20]) # hidden weak agent
        correct_state = np.stack((c_agent_0_state,
                                  c_agent_1_state,
                                  c_agent_2_state,
                                  c_agent_3_state,
                                  c_agent_4_state,), axis=0)

        self.assertTrue((destroyed_state == correct_state).all())

        ### weak agent attacking a strong agent
        failed_attack_action = {'actor': 0, 
                                'type': 1}

        intact_state = ipomdp.transition(state=model_state,
                                         action=failed_attack_action,
                                         model=model)

        # correct agent states
        c_agent_0_state = np.array([11, 0.5, 0.6, 20]) # weak agent
        c_agent_1_state = np.array([54, 1.0, 0.9, 30]) # strong agent
        c_agent_2_state = np.array([1, 1.0, 0.4, 40]) # recently destroyed agent
        c_agent_3_state = np.array([13, 0.9, 0.5, 12]) # growing agent
        c_agent_4_state = np.array([6, 0.1, 0.3, 20]) # hidden weak agent
        correct_state = np.stack((c_agent_0_state,
                                  c_agent_1_state,
                                  c_agent_2_state,
                                  c_agent_3_state,
                                  c_agent_4_state,), axis=0)

        self.assertTrue((intact_state == correct_state).all())

        ### hiding action
        hiding_action = {'actor': 4, 
                         'type': 'hide'}

        hidden_state = ipomdp.transition(state=model_state,
                                         action=hiding_action,
                                         model=model)

        # correct agent states
        c_agent_0_state = np.array([11, 0.5, 0.6, 20]) # weak agent
        c_agent_1_state = np.array([54, 1.0, 0.9, 30]) # strong agent
        c_agent_2_state = np.array([1, 1.0, 0.4, 40]) # recently destroyed agent
        c_agent_3_state = np.array([13, 0.9, 0.5, 12]) # growing agent
        c_agent_4_state = np.array([6, 0.05, 0.3, 20]) # hidden weak agent
        correct_state = np.stack((c_agent_0_state,
                                  c_agent_1_state,
                                  c_agent_2_state,
                                  c_agent_3_state,
                                  c_agent_4_state,), axis=0)

        self.assertTrue((hidden_state == correct_state).all())

        ### skip action
        skip_action = {'actor': 3, 
                       'type': '-'}

        skipped_state = ipomdp.transition(state=model_state,
                                          action=skip_action,
                                          model=model)

        # correct agent states
        c_agent_0_state = np.array([11, 0.5, 0.6, 20]) # weak agent
        c_agent_1_state = np.array([54, 1.0, 0.9, 30]) # strong agent
        c_agent_2_state = np.array([1, 1.0, 0.4, 40]) # recently destroyed agent
        c_agent_3_state = np.array([13, 0.9, 0.5, 12]) # growing agent
        c_agent_4_state = np.array([6, 0.1, 0.3, 20]) # hidden weak agent
        correct_state = np.stack((c_agent_0_state,
                                  c_agent_1_state,
                                  c_agent_2_state,
                                  c_agent_3_state,
                                  c_agent_4_state,), axis=0)

        self.assertTrue((skipped_state == correct_state).all())

    def test_initial_belief(self):

        model = helpers.create_small_universe(n_agents=5,
                                              agent_growth=growth.sigmoid_growth,
                                              agent_growth_params=
                                                {'speed_range': (0.5, 1),
                                                 'takeoff_time_range': (10, 20)},
                                              init_age_belief_range=(50, 100),
                                              init_visibility_belief_range=(0, 0.2),
                                              rng_seed=None)
        agent = model.agents[0]

        # generate belief
        init_belief = ipomdp.sample_init(n_samples=3, 
                                         model=model,
                                         agent=agent)

        # check shape
        correct_shape = (3, 5, 4)
        self.assertEqual(correct_shape, init_belief.shape)

        # check range of ages
        ages = init_belief[:, 1:, 0]
        self.assertTrue(((ages >= 50) & (ages <= 100)).all())

        # check range of visibilities
        visibilities = init_belief[:, 1:, 1]
        self.assertTrue(((visibilities >= 0) & (visibilities <= 0.2)).all())

        # check range of growth speeds
        speeds = init_belief[:, 1:, 2]
        self.assertTrue(((speeds >= 0.5) & (speeds <= 1)).all())

        # check range of takeoff times
        takeoff_times = init_belief[:, 1:, 3]
        self.assertTrue(((takeoff_times >= 10) & (takeoff_times <= 20)).all())

        # check that agent's own state is correct
        correct_state = agent.get_state()
        belief_agent_state = init_belief[:, 0, :]
        self.assertTrue((correct_state == belief_agent_state).all())


if __name__ == '__main__':
    unittest.main()