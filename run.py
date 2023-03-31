# %%
from model.model import Universe
from model.visualise import (draw_universe, plot_technology_distribution, 
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
action_data = model.datacollector.get_table_dataframe("actions")

# %% Visualise model run
vis = draw_universe(data=data, action_data=action_data, 
                    anim_filename="output.mp4", 
                    anim_length=60)
plt.show()

# %% Diagnostic plots
plot_technology_distribution(data, **params)
plot_streak_length_distribution(action_data, **params)

