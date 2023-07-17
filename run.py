# %%
import matplotlib.pyplot as plt
from model import universe, visualise
from tqdm import tqdm
import pickle

# %% Run model and gather data

# parameters
params = {
    "n_agents": 2,
    "agent_growth": "sigmoid",
    "agent_growth_params": {
        "speed_range": (0.5, 0.5),
        "takeoff_time_range": (20, 40),
        "speed_noise_scale": 0.03,
        "speed_noise_dist": "normal",
        "takeoff_time_noise_scale": 3,
    },
    "rewards": {"destroyed": -1, "hide": -0.01, "attack": 0},
    "n_root_belief_samples": 1000,
    "n_tree_simulations": 10000,
    "n_reinvigoration_particles": 0,
    "obs_noise_sd": 0.1,
    "obs_self_noise_sd": 0.03,
    "reasoning_level": 2,
    "action_dist_0": "random",  # can be "random", "passive" or "aggressive"
    "discount_factor": 0.7,
    "discount_epsilon": 0.10,
    "exploration_coef": 0.1,
    "softargmax_coef": 0.01,
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
data = mdl.datacollector.get_agent_vars_dataframe()
action_data = mdl.datacollector.get_table_dataframe("actions")

# %%  save data
with open("output/data.pickle", "wb") as f:
    pickle.dump(data, f)

with open("output/action_data.pickle", "wb") as f:
    pickle.dump(action_data, f)

# %% Visualise model run
vis = visualise.draw_universe(
    data=data,
    action_data=action_data,
    anim_filename="output/output.mp4",
    anim_length=60,
)
plt.show()

# %% Diagnostic plots
visualise.plot_technology_distribution(data, **params)
visualise.plot_streak_length_distribution(action_data, **params)
