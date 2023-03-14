# %%
from model import Universe
from analyse import count_streaks
from visualise import (draw_universe, plot_technology_distribution, 
                       plot_streak_length_distribution)
import matplotlib.pyplot as plt
from tqdm import tqdm

# %% Run model and gather data

# parameters
params = {'num_steps': 500,
          'num_agents': 20,
          'decision_making': 'targeted',
          'hostility_belief_prior': 0.1,
          'speed_range': (0.3, 1),
          'takeoff_time_range': (10, 100)}

# create a universe
model = Universe(debug=False, **params)
# simulate
for i in tqdm(range(params["num_steps"])):
    model.step()

# retrieve data
data = model.datacollector.get_agent_vars_dataframe() 
attack_data = model.datacollector.get_table_dataframe("attacks")

# %% Visualise model run
vis = draw_universe(data=data, attack_data=attack_data)
plt.show()

# %% Diagnostic plots
plot_technology_distribution(data, **params)
plot_streak_length_distribution(attack_data, **params)

