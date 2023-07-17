# %%
from model import universe


def create_small_universe(**kwargs):
    params = {
        "n_agents": 3,
        "agent_growth": "sigmoid",
        "agent_growth_params": {
            "speed_range": (0.3, 1),
            "takeoff_time_range": (10, 100),
            "speed_noise_scale": 0.03,
            "speed_noise_dist": "normal",
            "takeoff_time_noise_scale": 3,
        },
        "rewards": {"destroyed": -1, "hide": -0.01, "attack": 0},
        "n_root_belief_samples": 1000,
        "n_tree_simulations": 200,
        "n_reinvigoration_particles": 100,
        "obs_noise_sd": 0.1,
        "obs_self_noise_sd": 0.03,
        "reasoning_level": 2,
        "action_dist_0": "random",
        "discount_factor": 0.9,
        "discount_epsilon": 0.05,
        "exploration_coef": 1,
        "softargmax_coef": 1 / 10,
        "visibility_multiplier": 0.5,
        "decision_making": "ipomdp",
        "init_age_belief_range": (10, 100),
        "init_age_range": (10, 100),
        "init_visibility_belief_range": (0, 1),
        "init_visibility_range": (0, 1),
    }

    params.update(**kwargs)

    return universe.Universe(**params)
