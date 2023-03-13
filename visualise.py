import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import ArtistAnimation

def draw_universe(model=None, data=None, attack_data=None, colormap=mpl.colormaps['Dark2']):
    """
    If given a model, draw the current configuration of the universe
    in the model.
    If given data with a single timestep, draw the configuration of the
    universe in the data.
    If given data with multiple timesteps, draw an animation of the 
    configurations in the data.
    """

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

    # steps and agents to animate
    steps = data.index.get_level_values('Step').unique()
    agents = data.index.get_level_values('AgentID').unique()

    # initialise plot
    plt.style.use('dark_background')
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')

    # artists will store lists corresponding to elements draw at each step
    artists = []

    for step in steps:

        # list containing all elements to draw at this time step
        step_artists = []

        # add text indicating time step
        text = ax.text(0.45, 1.05, f"t = {step}")
        step_artists.append(text)

        # choose data just from this time step
        step_data = data.xs(step, level="Step")

        ### first draw universal agents (infinite vision)
        universal_agent_data = step_data[step_data['Radius of Influence'] >= 1]
        ids = universal_agent_data.index.get_level_values('AgentID')
        positions = universal_agent_data['Position']
        x = [pos[0] for pos in positions]
        y = [pos[1] for pos in positions]
        colors = [colormap(id % 5) for id in ids]

        paths = ax.scatter(x, y, c=colors, s=7**2, marker="d")
        step_artists.append(paths)

        ### draw other agents, showing their radius of influence
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

        # draw arrows indicating attacks
        if isinstance(attack_data, pd.DataFrame) and step in attack_data.time.values:

            step_attack_data = attack_data[attack_data.time == step]
            attacker_id, target_id = step_attack_data['attacker'], step_attack_data['target']

            a_x, a_y = step_data.loc[attacker_id].Position.values[0]
            t_x, t_y = step_data.loc[target_id].Position.values[0]

            arrow = ax.arrow(x=a_x, y=a_y, dx=t_x - a_x, dy=t_y - a_y, 
                             length_includes_head=True, width=0.005,
                             color="white")
            step_artists.append(arrow)

    # revert back to default style
    plt.style.use("default")

    # if there are multiple steps, animate
    if len(steps) > 1:
        ani = ArtistAnimation(fig=fig, artists=artists, interval=800, repeat=True)
        return ani

    return fig, ax
