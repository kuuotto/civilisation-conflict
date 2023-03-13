# %%
from model import Universe
from analyse import count_streaks
from visualise import draw_universe
import matplotlib.pyplot as plt
from tqdm import tqdm

# %%

# parameters
params = {'decision_making': 'targeted',
          'hostility_belief_prior': 0.1,
          'num_agents': 10}
num_steps = 50

# create a universe
model = Universe(debug=False, **params)
# simulate
for i in tqdm(range(num_steps)):
    model.step()

# retrieve and visualise data
data = model.datacollector.get_agent_vars_dataframe() 
attack_data = model.datacollector.get_table_dataframe("attacks")
vis = draw_universe(data=data, attack_data=attack_data)
plt.show()


# %%

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

