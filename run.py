# %%
from model.model import Universe, sigmoid_growth
from model.visualise import (draw_universe, plot_technology_distribution, 
                       plot_streak_length_distribution)
import matplotlib.pyplot as plt
from tqdm import tqdm

# %% Run model and gather data

# parameters
params = {'n_agents': 50,
          'agent_growth': sigmoid_growth,
          'agent_growth_params': {'speed_range': (0.3, 1),
                                  'takeoff_time_range': (10, 100)},
          'obs_noise_sd': 0.05,
          'action_dist_0': 'random',
          'discount_factor': 0.9,
          'decision_making': 'targeted',
          'visibility_multiplier': 0.5}
n_steps = 100

# create a universe
model = Universe(debug=False, **params)
# simulate
for i in tqdm(range(n_steps)):
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

