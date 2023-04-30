# %%
from model.model import Universe
from model.growth import sigmoid_growth
from model.visualise import (draw_universe, plot_technology_distribution, 
                       plot_streak_length_distribution)
import matplotlib.pyplot as plt
from tqdm import tqdm

# %% Run model and gather data

# parameters
params = {'n_agents': 3,
          'agent_growth': sigmoid_growth,
          'agent_growth_params': {'speed_range': (0.3, 1),
                                  'takeoff_time_range': (10, 100)},
          'rewards': {'destroyed': -1, 'hide': -0.01, 'attack': 0},
          'n_belief_samples': 10,
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
          'init_visibility_belief_range': (1, 1),
          'init_visibility_range': (1, 1)}
n_steps = 100

# create a universe
model = Universe(debug=False, **params)
# simulate
for id in tqdm(range(n_steps)):
    model.step()

# retrieve data
data = model.datacollector.get_agent_vars_dataframe() 
action_data = model.datacollector.get_table_dataframe("actions")

# %% Visualise model run
vis = draw_universe(data=data, action_data=action_data, 
                    anim_filename="output/output.mp4", 
                    anim_length=60)
plt.show()

# %% Diagnostic plots
plot_technology_distribution(data, **params)
plot_streak_length_distribution(action_data, **params)

