# %%

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import ArtistAnimation
from model import analyse


def draw_universe(
    model=None,
    data=None,
    action_data=None,
    colormap=mpl.colormaps["Dark2"],
    anim_filename=None,
    anim_length=60,
):
    """
    Visualise the model simulation.

    If given a model, draw the current configuration of the universe
    in the model.
    If given data with a single timestep, draw the configuration of the
    universe in the data.
    If given data with multiple timesteps, draw an animation of the
    configurations in the data.

    Keyword arguments:
    model: a Universe object
    data: data (a pandas DataFrame) from the model datacollector
    action_data: action data (a pandas DataFrame) from the model datacollector
    colormap: color scheme for coloring the agent symbols (different colors
              don't have any meaning besides making it easier to distinguish
              agents)
    anim_filename: if a string is supplied, the animation is saved to this path
    anim_length: desired length of animation (in seconds)
    """

    if model:
        # if we are given a model, turn its current state into a DataFrame

        steps = (model.schedule.time,)
        agents = model.schedule.agents
        ids = [agent.unique_id for agent in agents]
        tech_levels = [agent.tech_level for agent in agents]
        influence_radii = [agent.influence_radius for agent in agents]
        visibility_factors = [agent.visibility_factor for agent in agents]
        positions = [agent.pos for agent in agents]

        data = pd.DataFrame(
            {
                "Technology": tech_levels,
                "Radius of Influence": influence_radii,
                "Visibility Factor": visibility_factors,
                "Position": positions,
            },
            index=pd.MultiIndex.from_tuples(
                [(steps[0], id) for id in ids], names=["Step", "AgentID"]
            ),
        )

    # steps and agents to animate
    steps = data.index.get_level_values("Step").unique()
    agents = data.index.get_level_values("AgentID").unique()

    # initialise plot
    plt.style.use("dark_background")
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")

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
        universal_agent_data = step_data[step_data["Radius of Influence"] >= 1]
        ids = universal_agent_data.index.get_level_values("AgentID")
        positions = universal_agent_data["Position"]
        x = [pos[0] for pos in positions]
        y = [pos[1] for pos in positions]
        colors = [colormap(id % 5) for id in ids]
        facecolors = [
            c[:3] + (v,)
            for c, v in zip(colors, universal_agent_data["Visibility Factor"])
        ]

        paths = ax.scatter(
            x, y, edgecolors=colors, s=50, marker="d", facecolor=facecolors
        )
        step_artists.append(paths)

        ### draw other agents, showing their radius of influence
        normal_agent_data = step_data[step_data["Radius of Influence"] < 1]
        ids = normal_agent_data.index.get_level_values("AgentID")
        positions = normal_agent_data["Position"]
        x = [pos[0] for pos in positions]
        y = [pos[1] for pos in positions]
        colors = [colormap(id % 5) for id in ids]
        facecolors = [
            c[:3] + (v,) for c, v in zip(colors, normal_agent_data["Visibility Factor"])
        ]
        influence_radii = normal_agent_data["Radius of Influence"]

        paths = ax.scatter(x, y, edgecolors=colors, facecolors=facecolors, s=20)
        step_artists.append(paths)

        # draw circles indicating influence radii
        for position, influence_radius, color in zip(
            positions, influence_radii, colors
        ):
            patch = ax.add_patch(
                Circle(position, influence_radius, alpha=0.1, color=color)
            )
            step_artists.append(patch)

        # if action data is supplied, draw arrows indicating attacks
        # and circles indicating turns to move
        if isinstance(action_data, pd.DataFrame) and step in action_data.time.values:
            step_action_data = action_data[action_data.time == step]

            # draw a circle around the actor
            actor_id = step_action_data.actor
            actor_pos = step_data.loc[actor_id].Position.values[0]
            circle = ax.add_patch(
                Circle(
                    actor_pos,
                    radius=0.03,
                    alpha=0.5,
                    linestyle="dashed",
                    edgecolor="white",
                    facecolor="none",
                )
            )
            step_artists.append(circle)

            # if action is an attack, draw an arrow
            if step_action_data["action"].values[0] == "a":
                target_id = step_action_data["attack_target"]

                a_x, a_y = actor_pos
                t_x, t_y = step_data.loc[target_id].Position.values[0]

                arrow = ax.arrow(
                    x=a_x,
                    y=a_y,
                    dx=t_x - a_x,
                    dy=t_y - a_y,
                    length_includes_head=True,
                    width=0.005,
                    head_width=0.03,
                    head_length=0.03,
                    color="white",
                )
                step_artists.append(arrow)

        artists.append(step_artists)

    # revert back to default style
    plt.style.use("default")

    # if there are multiple steps, animate
    if len(steps) > 1:
        # determine interval from desired animation length
        interval = int(anim_length * 1000 / len(steps))

        # create animation
        ani = ArtistAnimation(fig=fig, artists=artists, interval=interval, repeat=True)

        # save to a file if requested
        if anim_filename:
            ani.save(anim_filename)

        return ani

    return fig, ax


def get_technology_distribution_step(data, step, bins, normalise=False):
    """
    Determine the technology level distribution at the given time step.

    Parameters:
    data: an agent data Pandas DataFrame collected by the model datacollector
    step: an integer, pointing the step in the data which to use
    normalise: whether to return a properly normalised probability distribution
               (True) or counts (False)

    Returns:
    a dictionary where keys are technology values and values are either
    absolute or relative frequencies (depending on the value of the normalise
    parameter)
    """

    # collect data and update distribution
    step_data = data.xs(step, level="Step")["Technology"]

    # bin the data
    freqs, _ = np.histogram(step_data, bins=bins, range=(0, 1))

    if normalise and freqs.sum() > 0:
        freqs = freqs / freqs.sum()

    return freqs


def _get_caption(**params):
    """Generate a caption from model parameters"""
    return "\n".join([f"{k}: {v}" for k, v in params.items()])


def plot_technology_distribution_step(data, step, n_bins, **params):
    """
    Plot the technology level distribution at the given time step.

    Parameters:
    data: an agent data Pandas DataFrame collected by the model datacollector
    step: an integer, pointing the step in data which to visualise
    normalise: whether to normalise the frequency distribution
    **params: model parameter values used for the simulation. These will be
              displayed under the plot as a caption.
    """
    # initialise figure
    fig, ax = plt.subplots(constrained_layout=True, figsize=(5, 5))

    # calculate distribution
    step_tech_levels = data.xs(step, level="Step")["Technology"]

    # draw
    ax.hist(x=step_tech_levels, bins=n_bins, range=(0, 1))
    ax.set_xlabel("Technology Level")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Technology Level Distribution at t={step}")
    fig.supxlabel(_get_caption(**params))
    plt.show()


def plot_technology_distribution(data, n_bins=10, **params):
    """
    Plot the technology level distribution over time as a heat map.

    Parameters:
    data: an agent data Pandas DataFrame collected by the model datacollector
    n_bins: number of bins to divide technology levels (in the range [0,1]) to
    **params: model parameter values used for the simulation. These will be
              displayed under the plot as a caption.
    """
    fig, ax = plt.subplots(constrained_layout=True, figsize=(5, 5))

    bin_edges = np.linspace(0, 1, n_bins + 1, endpoint=True)
    steps = data.index.get_level_values("Step").unique()

    # initialise array
    dist_array = np.zeros((n_bins, len(steps)))

    for step in steps:
        # calculate distribution on this step
        dist = get_technology_distribution_step(
            data=data, step=step, bins=bin_edges, normalise=True
        )
        dist_array[:, step] = dist

    im = ax.imshow(
        dist_array,
        interpolation="nearest",
        origin="lower",
        extent=(0, max(steps), 0, 1),
        aspect="auto",
    )
    fig.colorbar(im, label="frequency")

    ax.set_title("Distribution of Technology Levels")
    ax.set_xlabel("Time")
    ax.set_ylabel("Technology Level")
    fig.supxlabel(_get_caption(**params))

    plt.show()


def plot_streak_length_distribution(action_data, **params):
    """
    Visualise distribution of attack streak lengths on a log-log scale.
    An attack streak is defined as successive time steps when an attack occurs
    (whether successful or not).

    Parameters:
    action_data: a pandas DataFrame collected by the model datacollector
    **params: model parameter values used for the simulation. These will be
              displayed under the plot as a caption.
    """
    fig, ax = plt.subplots(constrained_layout=True, figsize=(5, 5))

    # count streaks
    streaks = analyse.count_streaks(
        action_data[action_data.action == "a"]["time"].values
    )

    if len(streaks) == 0:
        print("There were no attacks in the data.")
        plt.show()
        return

    # visualise
    ax.scatter(x=list(streaks.keys()), y=list(streaks.values()))

    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("Attack Streak Length")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Attack Streak Lengths")
    ax.grid()
    fig.supxlabel(_get_caption(**params))

    plt.show()
