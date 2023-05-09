# %%
from model import universe, growth

def create_small_universe(**kwargs):
    params = {'n_agents': 3,
              'agent_growth': growth.sigmoid_growth,
              'agent_growth_params': {'speed_range': (0.3, 1),
                                      'takeoff_time_range': (10, 100)},
              'rewards': {'destroyed': -1, 'hide': -0.01, 'attack': 0},
              'n_root_belief_samples': 10,
              'obs_noise_sd': 0.05,
              'belief_update_time_horizon': 1,
              'planning_time_horizon': 2,
              'reasoning_level': 2,
              'action_dist_0': 'random',
              'discount_factor': 0.9,
              'visibility_multiplier': 0.5,
              'decision_making': 'random',
              'init_age_belief_range': (10, 100),
              'init_age_range': (10, 100),
              'init_visibility_belief_range': (0, 1),
              'init_visibility_range': (0, 1)}

    params.update(**kwargs)

    return universe.Universe(**params)