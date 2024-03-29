# %%
import pickle

import matplotlib.pyplot as plt
from tqdm import tqdm

from model import analyse, universe, visualise

# %% Run model and gather data

# parameters
params = {
    "n_agents": 2,
    "agent_growth": "sigmoid",
    "agent_growth_params": {
        "speed_range": (0.3, 0.5),
        "takeoff_time_range": (20, 40),
        "speed_noise_scale": 0.03,
        "speed_noise_dist": "normal",
        "takeoff_time_noise_scale": 3,
    },
    "rewards": {"destroyed": -1, "hide": -0.01, "attack": 0},
    "prob_indifferent": 0,
    "n_root_belief_samples": 1000,
    "n_tree_simulations": 1000,
    "obs_noise_sd": 0.15,
    "obs_self_noise_sd": 0.15,
    "reasoning_level": 1,
    "action_dist_0": "random",  # can be "random", "passive" or "aggressive"
    "initial_belief": "uniform",  # can be "uniform" or "surpass_scenario"
    "initial_belief_params": {
        "time_until_surpass": 3,
        "prob_surpass_0": 0,
        "prob_surpass_1": 0.5,
    },
    "discount_factor": 0.6,
    "discount_epsilon": 0.10,
    "exploration_coef": 0.6,
    "softargmax_coef": 0.1,
    "visibility_multiplier": 0.5,
    "decision_making": "ipomdp",
    "init_age_belief_range": (0, 50),
    "init_age_range": (0, 50),
    "init_visibility_belief_range": (1, 1),
    "init_visibility_range": (1, 1),
}
n_steps = 100

# create a universe
mdl = universe.Universe(debug=1, seed=0, **params)
# simulate
for i in tqdm(range(n_steps)):
    mdl.step()

# retrieve data
agent_data = mdl.datacollector.get_agent_vars_dataframe()
action_data = mdl.datacollector.get_table_dataframe("actions")
reward_data = mdl.datacollector.get_table_dataframe("rewards")
action_quality_data = mdl.datacollector.get_table_dataframe("action_qualities")

# %%  save data
with open("output/data.pickle", "wb") as f:
    pickle.dump(
        {
            "agent_data": agent_data,
            "action_data": action_data,
            "reward_data": reward_data,
            "action_quality_data": action_quality_data,
        },
        f,
    )

# %% Visualise model run
vis = visualise.draw_universe(
    data=agent_data,
    action_data=action_data,
    anim_filename="output/output.gif",
    anim_length=60,
)
plt.show()

# %% Diagnostic plots
visualise.plot_technology_distribution(agent_data)
visualise.plot_streak_length_distribution(analyse.count_attack_streaks(action_data))
