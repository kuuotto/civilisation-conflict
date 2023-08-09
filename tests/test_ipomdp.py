import unittest
import numpy as np
from scipy.stats import norm
from tests import helpers
from model import growth, ipomdp, action


class TestTransition(unittest.TestCase):
    def test_transition_1(self):
        model = helpers.create_small_universe(n_agents=5, visibility_multiplier=0.5)

        agent_0_state = np.array([10, 0.5, 0.6, 20])  # weak agent
        agent_1_state = np.array([53, 1.0, 0.9, 30])  # strong agent
        agent_2_state = np.array([0, 1.0, 0.4, 40])  # recently destroyed agent
        agent_3_state = np.array([12, 0.9, 0.5, 12])  # growing agent
        agent_4_state = np.array([5, 0.1, 0.3, 20])  # hidden weak agent

        model_state = np.stack(
            (agent_0_state, agent_1_state, agent_2_state, agent_3_state, agent_4_state),
            axis=0,
        )

        ### strong attacking a weak agent
        destroy_action = (model.agents[1], model.agents[0])

        destroyed_state = ipomdp.transition(
            state=model_state, action_=destroy_action, model=model
        )

        # correct agent states
        c_agent_0_state = np.array([0, 1.0, 0.6, 20])  # weak agent
        c_agent_1_state = np.array([54, 1.0, 0.9, 30])  # strong agent
        c_agent_2_state = np.array([1, 1.0, 0.4, 40])  # recently destroyed agent
        c_agent_3_state = np.array([13, 0.9, 0.5, 12])  # growing agent
        c_agent_4_state = np.array([6, 0.1, 0.3, 20])  # hidden weak agent
        correct_state = np.stack(
            (
                c_agent_0_state,
                c_agent_1_state,
                c_agent_2_state,
                c_agent_3_state,
                c_agent_4_state,
            ),
            axis=0,
        )

        self.assertTrue((destroyed_state == correct_state).all())

        ### weak agent attacking a strong agent
        failed_attack_action = (model.agents[0], model.agents[1])

        intact_state = ipomdp.transition(
            state=model_state, action_=failed_attack_action, model=model
        )

        # correct agent states
        c_agent_0_state = np.array([11, 0.5, 0.6, 20])  # weak agent
        c_agent_1_state = np.array([54, 1.0, 0.9, 30])  # strong agent
        c_agent_2_state = np.array([1, 1.0, 0.4, 40])  # recently destroyed agent
        c_agent_3_state = np.array([13, 0.9, 0.5, 12])  # growing agent
        c_agent_4_state = np.array([6, 0.1, 0.3, 20])  # hidden weak agent
        correct_state = np.stack(
            (
                c_agent_0_state,
                c_agent_1_state,
                c_agent_2_state,
                c_agent_3_state,
                c_agent_4_state,
            ),
            axis=0,
        )

        self.assertTrue((intact_state == correct_state).all())

        ### hiding action
        hiding_action = (model.agents[4], action.HIDE)

        hidden_state = ipomdp.transition(
            state=model_state, action_=hiding_action, model=model
        )

        # correct agent states
        c_agent_0_state = np.array([11, 0.5, 0.6, 20])  # weak agent
        c_agent_1_state = np.array([54, 1.0, 0.9, 30])  # strong agent
        c_agent_2_state = np.array([1, 1.0, 0.4, 40])  # recently destroyed agent
        c_agent_3_state = np.array([13, 0.9, 0.5, 12])  # growing agent
        c_agent_4_state = np.array([6, 0.05, 0.3, 20])  # hidden weak agent
        correct_state = np.stack(
            (
                c_agent_0_state,
                c_agent_1_state,
                c_agent_2_state,
                c_agent_3_state,
                c_agent_4_state,
            ),
            axis=0,
        )

        self.assertTrue((hidden_state == correct_state).all())

        ### skip action
        skip_action = (model.agents[3], action.NO_ACTION)

        skipped_state = ipomdp.transition(
            state=model_state, action_=skip_action, model=model
        )

        # correct agent states
        c_agent_0_state = np.array([11, 0.5, 0.6, 20])  # weak agent
        c_agent_1_state = np.array([54, 1.0, 0.9, 30])  # strong agent
        c_agent_2_state = np.array([1, 1.0, 0.4, 40])  # recently destroyed agent
        c_agent_3_state = np.array([13, 0.9, 0.5, 12])  # growing agent
        c_agent_4_state = np.array([6, 0.1, 0.3, 20])  # hidden weak agent
        correct_state = np.stack(
            (
                c_agent_0_state,
                c_agent_1_state,
                c_agent_2_state,
                c_agent_3_state,
                c_agent_4_state,
            ),
            axis=0,
        )

        self.assertTrue((skipped_state == correct_state).all())


class TestInitialBelief(unittest.TestCase):
    def test_initial_belief(self):
        model = helpers.create_small_universe(
            n_agents=5,
            agent_growth="sigmoid",
            agent_growth_params={
                "speed_range": (0.5, 1),
                "takeoff_time_range": (10, 20),
            },
            init_age_belief_range=(50, 100),
            init_visibility_belief_range=(0, 0.2),
            seed=None,
        )
        agent = model.agents[0]

        # generate belief
        init_belief = ipomdp.sample_init(n_samples=3, model=model, agent=agent)

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


class TestObservationProbability(unittest.TestCase):
    def test_observation_probability_1(self):
        mdl = helpers.create_small_universe(
            n_agents=2,
        )

        ag0, ag1 = mdl.agents

        states = np.array(
            [
                [
                    [10, 1.0, 0.5, 5],  # agent 0 is strong
                    [0, 1.0, 0.5, 5],  # agent 1 is weak
                ],
                [
                    [20, 1.0, 0.5, 5],  # agent 0 is strong
                    [0, 1.0, 0.5, 10],  # agent 1 is weak
                ],
            ]
        )

        # actions = ({ag0: action.NO_ACTION}, {ag0: action.NO_ACTION})
        actor_ids = np.array([0, 0])
        target_ids = np.array([-1, -1])

        # array of size (n_states, n_agents)
        tech_levels = growth.tech_level(state=states, model=mdl)

        ### check observation weights of agent 0
        ag0_obs = (0.9, -0.1, None, None)
        ag0_exp_obs_1 = (
            tech_levels[0, 0],
            tech_levels[0, 1] * states[0, 1, 1],
            None,
            None,
        )
        ag0_exp_obs_2 = (
            tech_levels[1, 0],
            tech_levels[1, 1] * states[1, 1, 1],
            None,
            None,
        )

        weights = ipomdp.prob_observation(
            observation=ag0_obs,
            states=states,
            prev_action_actor_ids=actor_ids,
            prev_action_target_ids=target_ids,
            observer=ag0,
            model=mdl,
        )

        correct_weights = np.array(
            [
                norm.pdf(
                    x=ag0_obs[0], loc=ag0_exp_obs_1[0], scale=mdl.obs_self_noise_sd
                )
                * norm.pdf(x=ag0_obs[1], loc=ag0_exp_obs_1[1], scale=mdl.obs_noise_sd),
                norm.pdf(
                    x=ag0_obs[0], loc=ag0_exp_obs_2[0], scale=mdl.obs_self_noise_sd
                )
                * norm.pdf(x=ag0_obs[1], loc=ag0_exp_obs_2[1], scale=mdl.obs_noise_sd),
            ]
        )

        self.assertTrue(np.allclose(weights, correct_weights))

        ### check observation weights of agent 1
        ag1_obs = (0.6, -0.1, None, None)
        ag1_exp_obs_1 = (
            ag1_obs[0],
            tech_levels[0, 1],
            None,
            None,
        )
        ag1_exp_obs_2 = (
            ag1_obs[0],
            tech_levels[1, 1],
            None,
            None,
        )

        weights = ipomdp.prob_observation(
            observation=ag1_obs,
            states=states,
            prev_action_actor_ids=actor_ids,
            prev_action_target_ids=target_ids,
            observer=ag1,
            model=mdl,
        )

        correct_weights = np.array(
            [
                norm.pdf(x=ag1_obs[0], loc=ag1_exp_obs_1[0], scale=mdl.obs_noise_sd)
                * norm.pdf(
                    x=ag1_obs[1], loc=ag1_exp_obs_1[1], scale=mdl.obs_self_noise_sd
                ),
                norm.pdf(x=ag1_obs[0], loc=ag1_exp_obs_2[0], scale=mdl.obs_noise_sd)
                * norm.pdf(
                    x=ag1_obs[1], loc=ag1_exp_obs_2[1], scale=mdl.obs_self_noise_sd
                ),
            ]
        )

        self.assertTrue(np.allclose(weights, correct_weights))

    def test_observation_probability_2(self):
        mdl = helpers.create_small_universe(
            n_agents=2,
        )

        ag0, ag1 = mdl.agents

        states = np.array(
            [
                [
                    [20, 1.0, 0.5, 5],  # agent 0 is strong
                    [0, 1.0, 0.5, 5],  # agent 1 is weak (just destroyed)
                ],
                [
                    [20, 1.0, 0.5, 5],  # agent 0 is strong
                    [19, 1.0, 0.5, 5],  # agent 1 is strong, but not as strong as 0
                ],
            ]
        )

        # actions = ({ag0: ag1}, {ag1: ag0})
        actor_ids = np.array([0, 1])
        target_ids = np.array([1, 0])

        ### test observations of agent 0

        observation1_ag0 = (0.9, 0.0, True, None)
        weights = ipomdp.prob_observation(
            observation=observation1_ag0,
            states=states,
            prev_action_actor_ids=actor_ids,
            prev_action_target_ids=target_ids,
            observer=ag0,
            model=mdl,
        )

        self.assertGreater(weights[0], 0)  # agent 0 can reach 1 and is stronger
        self.assertEqual(weights[1], 0)  # agent 0 did not attack

        observation2_ag0 = (0.9, 0.0, None, False)
        weights = ipomdp.prob_observation(
            observation=observation2_ag0,
            states=states,
            prev_action_actor_ids=actor_ids,
            prev_action_target_ids=target_ids,
            observer=ag0,
            model=mdl,
        )

        self.assertEqual(
            weights[0], 0
        )  # agent 1 is not strong enough to cause a failed attack
        self.assertGreater(weights[1], 0)  # agent 1 is strong enough to attack

        ### test observations of agent 1

        observation1_ag1 = (0.9, 0.0, None, True)
        weights = ipomdp.prob_observation(
            observation=observation1_ag1,
            states=states,
            prev_action_actor_ids=actor_ids,
            prev_action_target_ids=target_ids,
            observer=ag1,
            model=mdl,
        )

        self.assertGreater(weights[0], 0)  # agent 0 can reach 1 and is stronger
        self.assertEqual(weights[1], 0)  # agent 0 was not attacked

        observation2_ag1 = (0.9, 0.0, False, None)
        weights = ipomdp.prob_observation(
            observation=observation2_ag1,
            states=states,
            prev_action_actor_ids=actor_ids,
            prev_action_target_ids=target_ids,
            observer=ag1,
            model=mdl,
        )

        self.assertEqual(
            weights[0], 0
        )  # agent 1 is not strong enough to cause a failed attack
        self.assertGreater(weights[1], 0)  # agent 1 is strong enough to attack


class TestObservationSample(unittest.TestCase):
    def test_observation_sample_success_bits(self):
        mdl = helpers.create_small_universe(
            n_agents=2,
            visibility_multiplier=0.5,
        )

        strong, weak = mdl.agents

        # agent 0 is strong, agent 1 is weak
        propagated_states = np.array(
            [
                [
                    [11, 1.0, 0.5, 5],
                    [0, 1.0, 0.5, 5],
                ],
                [
                    [11, 1.0, 0.5, 5],
                    [2, 1.0, 0.5, 5],
                ],
                [
                    [11, 0.5, 0.5, 5],
                    [2, 1.0, 0.5, 5],
                ],
                [
                    [11, 1.0, 0.5, 5],
                    [2, 1.0, 0.5, 5],
                ],
                [
                    [11, 1.0, 0.5, 5],
                    [2, 1.0, 0.5, 5],
                ],
                [
                    [11, 1.0, 0.5, 5],
                    [2, 0.5, 0.5, 5],
                ],
                [
                    [12, 1.0, 0.5, 5],
                    [11, 1.0, 0.5, 5],  # here agent 1 can reach 0 but is still weaker
                ],
            ]
        )
        prev_actions = (
            (strong, weak),
            (strong, action.NO_ACTION),
            (strong, action.HIDE),
            (weak, strong),
            (weak, action.NO_ACTION),
            (weak, action.HIDE),
            (weak, strong),
        )

        ### strong attacks weak
        # observation of strong
        obs = ipomdp.sample_observation(
            state=propagated_states[0],
            action=prev_actions[0],
            observer=strong,
            model=mdl,
        )

        self.assertEqual(obs[-2], True)
        self.assertEqual(obs[-1], None)

        # observation of weak
        obs = ipomdp.sample_observation(
            state=propagated_states[0],
            action=prev_actions[0],
            observer=weak,
            model=mdl,
        )

        self.assertEqual(obs[-2], None)
        self.assertEqual(obs[-1], True)

        ### strong does nothing
        # observation of strong
        obs = ipomdp.sample_observation(
            state=propagated_states[1],
            action=prev_actions[1],
            observer=strong,
            model=mdl,
        )

        self.assertEqual(obs[-2], None)
        self.assertEqual(obs[-1], None)

        # observation of weak
        obs = ipomdp.sample_observation(
            state=propagated_states[1],
            action=prev_actions[1],
            observer=weak,
            model=mdl,
        )

        self.assertEqual(obs[-2], None)
        self.assertEqual(obs[-1], None)

        ### strong hides
        # observation of strong
        obs = ipomdp.sample_observation(
            state=propagated_states[2],
            action=prev_actions[2],
            observer=strong,
            model=mdl,
        )

        self.assertEqual(obs[-2], None)
        self.assertEqual(obs[-1], None)

        # observation of weak
        obs = ipomdp.sample_observation(
            state=propagated_states[2],
            action=prev_actions[2],
            observer=weak,
            model=mdl,
        )

        self.assertEqual(obs[-2], None)
        self.assertEqual(obs[-1], None)

        ### weak attacks
        # observation of strong
        obs = ipomdp.sample_observation(
            state=propagated_states[3],
            action=prev_actions[3],
            observer=strong,
            model=mdl,
        )

        self.assertEqual(obs[-2], None)
        self.assertEqual(obs[-1], None)

        # observation of weak
        obs = ipomdp.sample_observation(
            state=propagated_states[3],
            action=prev_actions[3],
            observer=weak,
            model=mdl,
        )

        self.assertEqual(obs[-2], None)
        self.assertEqual(obs[-1], None)

        ### weak does nothing
        # observation of strong
        obs = ipomdp.sample_observation(
            state=propagated_states[4],
            action=prev_actions[4],
            observer=strong,
            model=mdl,
        )

        self.assertEqual(obs[-2], None)
        self.assertEqual(obs[-1], None)

        # observation of weak
        obs = ipomdp.sample_observation(
            state=propagated_states[4],
            action=prev_actions[4],
            observer=weak,
            model=mdl,
        )

        self.assertEqual(obs[-2], None)
        self.assertEqual(obs[-1], None)

        ### weak hides
        # observation of strong
        obs = ipomdp.sample_observation(
            state=propagated_states[5],
            action=prev_actions[5],
            observer=strong,
            model=mdl,
        )

        self.assertEqual(obs[-2], None)
        self.assertEqual(obs[-1], None)

        # observation of weak
        obs = ipomdp.sample_observation(
            state=propagated_states[5],
            action=prev_actions[5],
            observer=weak,
            model=mdl,
        )

        self.assertEqual(obs[-2], None)
        self.assertEqual(obs[-1], None)

        ### weak (who can reach strong) attacks
        # first check that the weaker agent was able to reach the stronger agent in
        # the previous timestep
        prev_state = propagated_states[6].copy()
        prev_state[:, 0] -= 1
        prev_weak_tech_level = growth.tech_level(state=prev_state, model=mdl)[1]
        self.assertGreater(prev_weak_tech_level, mdl._distances_tech_level[0, 1])

        # observation of strong
        obs = ipomdp.sample_observation(
            state=propagated_states[6],
            action=prev_actions[6],
            observer=strong,
            model=mdl,
        )

        self.assertEqual(obs[-2], None)
        self.assertEqual(obs[-1], False)

        # observation of weak
        obs = ipomdp.sample_observation(
            state=propagated_states[6],
            action=prev_actions[6],
            observer=weak,
            model=mdl,
        )

        self.assertEqual(obs[-2], False)
        self.assertEqual(obs[-1], None)


class TestReward(unittest.TestCase):
    def test_reward(self):
        mdl = helpers.create_small_universe(
            n_agents=2,
            rewards={"destroyed": -1, "hide": -0.01, "attack": -0.1},
            agent_growth="sigmoid",
        )

        strong, weak = mdl.agents

        self.assertEqual(strong.id, 0)
        self.assertEqual(weak.id, 1)

        states = np.array(
            [
                [
                    [11, 1.0, 0.5, 5],
                    [0, 1.0, 0.5, 5],
                ],
                [
                    [11, 1.0, 0.5, 5],
                    [10, 1.0, 0.5, 5],
                ],
            ]
        )

        ### strong attacks weak
        # check strong's reward
        reward = ipomdp.reward(
            state=states[0],
            action_=(strong, weak),
            agent=strong,
            model=mdl,
        )
        self.assertEqual(reward, -0.1)

        # check weak's reward
        reward = ipomdp.reward(
            state=states[0],
            action_=(strong, weak),
            agent=weak,
            model=mdl,
        )
        self.assertEqual(reward, -1)

        ### weak attacks strong (weak cannot reach strong)
        # check strong's reward
        reward = ipomdp.reward(
            state=states[0],
            action_=(weak, strong),
            agent=strong,
            model=mdl,
        )
        self.assertEqual(reward, 0)

        # check weak's reward
        reward = ipomdp.reward(
            state=states[0],
            action_=(weak, strong),
            agent=weak,
            model=mdl,
        )
        self.assertEqual(reward, 0)

        ### weak attacks strong (weak can reach strong)

        # check that in the second state weak can in fact reach strong
        weak_tech_level = growth.tech_level(state=states[1], model=mdl)[weak.id]
        self.assertGreater(
            weak_tech_level, mdl._distances_tech_level[weak.id, strong.id]
        )

        # check strong's reward
        reward = ipomdp.reward(
            state=states[1],
            action_=(weak, strong),
            agent=strong,
            model=mdl,
        )
        self.assertEqual(reward, 0)

        # check weak's reward
        reward = ipomdp.reward(
            state=states[1],
            action_=(weak, strong),
            agent=weak,
            model=mdl,
        )
        self.assertEqual(reward, -0.1)

        ### strong hides
        reward = ipomdp.reward(
            state=states[0],
            action_=(strong, action.HIDE),
            agent=strong,
            model=mdl,
        )
        self.assertEqual(reward, -0.01)

        ### strong skips
        reward = ipomdp.reward(
            state=states[0],
            action_=(strong, action.NO_ACTION),
            agent=strong,
            model=mdl,
        )
        self.assertEqual(reward, 0)


if __name__ == "__main__":
    unittest.main()
