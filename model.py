# %%

%matplotlib widget

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import ArtistAnimation

import mesa

# %%

def influence_radius(tech_level):
    """
    Gives the radius of influence as a function of the technology level
    of the civilisation.

    TODO: define more rigorously
    """
    return 0.1*np.tan(tech_level * (np.pi / 2))

def sigmoid_growth(time, speed, takeoff_time):
    """
    Gives the current technology level for the agent. Assumes a
    sigmoid-shaped growing curve for the technological capabilities of
    civilisations.
    """
    return 1/(1+np.exp(-speed*(time - takeoff_time)))

class Civilisation(mesa.Agent):
    """An agent represeting a single civilisation in the universe"""

    def __init__(self, unique_id, model, growth, **growth_kwargs) -> None:
        super().__init__(unique_id, model)

        # by default, choose growth parameters randomly
        if len(growth_kwargs) < 1 and growth==sigmoid_growth:
            growth_kwargs = {'speed': self.model.random.uniform(2, 4),
                             'takeoff_time': self.model.random.randrange(1, 20)}
        
        # save the growth function
        self.growth = lambda time: growth(time, **growth_kwargs)

        # get initial tech level
        self.tech_level = self.growth(0)
        self.influence_radius = influence_radius(self.tech_level)

    def step(self):
        new_tech_level = self.growth(self.model.schedule.time)
        #print(f"{self.unique_id}: {self.tech_level:.4f} -> {new_tech_level:.4f}")
        self.tech_level = new_tech_level
        self.influence_radius = influence_radius(self.tech_level)

class Universe(mesa.Model):

    def __init__(self, num_agents, toroidal_space=False, agent_growth=sigmoid_growth) -> None:
        
        self.schedule = mesa.time.RandomActivation(self)
        self.space = mesa.space.ContinuousSpace(x_max=1, y_max=1, 
                                                torus=toroidal_space)

        # add agents
        for i in range(num_agents):
            agent = Civilisation(i, self, agent_growth)
            self.schedule.add(agent)
            x, y = self.random.random(), self.random.random() # todo
            self.space.place_agent(agent, (x, y))

        # initialise data collection
        self.datacollector = mesa.DataCollector(
            model_reporters={}, agent_reporters={
                "Technology":  "tech_level", 
                "Radius of Influence": "influence_radius",
                "Position": "pos"
            })

    def step(self):
        """Advance the model by one step."""
        self.datacollector.collect(self)
        self.schedule.step()


# %%

def draw_universe(model=None, data=None, colormap=mpl.colormaps['Dark2']):
    """
    If given a model, draw the current configuration of the universe
    in the model.
    If given data with a single timestep, draw the configuration of the
    universe in the data.
    If given data with multiple timesteps, draw an animation of the 
    configurations in the data.
    """

    steps = (1,)

    if data is not None and isinstance(data, pd.DataFrame):

        # if there are multiple steps, we animate
        steps = data.index.get_level_values('Step').unique()


    if model:
        # if we are given a model, turn its current state into a DataFrame
        
        steps = (model.schedule.time,)
        agents = model.schedule.agents
        ids = [agent.unique_id for agent in agents]
        tech_levels = [agent.tech_level for agent in agents]
        influence_radii = [agent.influence_radius for agent in agents]
        positions = [agent.pos for agent in agents]

        data = pd.DataFrame({'Technology': tech_levels,
                             'Radius of Influence': influence_radii,
                             'Position': positions},
                            index=pd.MultiIndex.from_tuples(
                                [(steps[0], id) for id in ids], 
                                names=['Step', 'AgentID']))


    # initialise plot
    plt.style.use('dark_background')
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')

    agents = data.index.get_level_values('AgentID').unique()

    artists = []

    for step in steps:

        step_artists = []

        step_data = data.xs(step, level="Step")

        # first draw universal agents (infinite vision)
        universal_agent_data = step_data[step_data['Radius of Influence'] >= 1]
        ids = universal_agent_data.index.get_level_values('AgentID')
        positions = universal_agent_data['Position']
        x = [pos[0] for pos in positions]
        y = [pos[1] for pos in positions]
        colors = [colormap(id % 5) for id in ids]

        paths = ax.scatter(x, y, c=colors, s=7**2, marker="d")
        step_artists.append(paths)

        # draw other agents, showing their radius of influence
        normal_agent_data = step_data[step_data['Radius of Influence'] < 1]
        ids = normal_agent_data.index.get_level_values('AgentID')
        positions = normal_agent_data['Position']
        x = [pos[0] for pos in positions]
        y = [pos[1] for pos in positions]
        colors = [colormap(id % 5) for id in ids]
        influence_radii = normal_agent_data['Radius of Influence']

        paths = ax.scatter(x, y, c=colors, s=3**2)
        step_artists.append(paths)

        # draw circles indicating influence radii
        for position, influence_radius, color in zip(positions, influence_radii, colors):
            patch = ax.add_patch(Circle(position, influence_radius, 
                                        alpha=0.1, color=color))
            step_artists.append(patch)

        artists.append(step_artists)

    if len(steps) > 1:
        # create animation
        ani = ArtistAnimation(fig=fig, artists=artists, interval=200, repeat=True)
        return ani

    return fig, ax


# %%
model = Universe(5)
for i in range(50):
    model.step()

print(f"Final results after {model.schedule.time} steps")
for agent in model.schedule.agents:
    print(f"{agent.unique_id}: tech {agent.tech_level:.4f}, radius {agent.influence_radius:.4f}")

data = model.datacollector.get_agent_vars_dataframe() 
vis = draw_universe(data=data)
plt.show()
