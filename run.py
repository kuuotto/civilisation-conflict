# %%
from model import Universe
from analyse import count_streaks
from visualise import draw_universe, plot_technology_distribution
import matplotlib.pyplot as plt
from tqdm import tqdm

# %% Run model and gather data

# parameters
params = {'decision_making': 'targeted',
          'hostility_belief_prior': 0.1,
          'num_agents': 20, 
          'speed_range': (0.3, 1),
          'takeoff_time_range': (10, 100)}
num_steps = 500

# create a universe
model = Universe(debug=False, **params)
# simulate
for i in tqdm(range(num_steps)):
    model.step()

# retrieve and visualise data
data = model.datacollector.get_agent_vars_dataframe() 
attack_data = model.datacollector.get_table_dataframe("attacks")

# %% Visualise model run
vis = draw_universe(data=data, attack_data=attack_data)
plt.show()

# %% Diagnostic plots
plot_technology_distribution(data)

# visualise streak length distribution
streaks = count_streaks(attack_data['time'])
fig, ax = plt.subplots()
ax.scatter(x=list(streaks.keys()), y=list(streaks.values()))
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel("Attack streak length")
ax.set_ylabel("Frequency")
ax.set_title((f"{params['decision_making']}, " + 
              f"{params['num_agents']} civilisations, " + 
              f"{num_steps} steps, " 
              f"hostility belief prior {params['hostility_belief_prior']}"))
ax.grid()
#fig.savefig("fig.pdf")
plt.show()

