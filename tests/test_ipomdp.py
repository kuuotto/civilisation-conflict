import unittest
import numpy as np
from model import ipomdp
from model.model import Universe, sigmoid_growth

class TestIPOMDP(unittest.TestCase):

    def test_transition_1(self):

        params = {'n_agents': 5,
                  'agent_growth': sigmoid_growth,
                  'agent_growth_params' : {'speed_range': (0.3, 1),
                                           'takeoff_time_range': (10, 100)},
                  'obs_noise_sd': 0.05,
                  'action_dist_0': 'random',
                  'discount_factor': 0.9,
                  'decision_making': 'targeted',
                  'visibility_multiplier': 0.5}

        model = Universe(**params)

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
        destroy_action = {'actor': model.schedule.agents[1], 
                          'type': model.schedule.agents[0]}

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
        failed_attack_action = {'actor': model.schedule.agents[0], 
                                'type': model.schedule.agents[1]}

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
        hiding_action = {'actor': model.schedule.agents[4], 
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
        skip_action = {'actor': model.schedule.agents[3], 
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

if __name__ == '__main__':
    unittest.main()