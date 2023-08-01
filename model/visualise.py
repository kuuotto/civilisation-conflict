# %%

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Patch
from matplotlib.animation import ArtistAnimation
from matplotlib.ticker import MultipleLocator
from model import analyse, ipomdp_solver, universe, growth
from typing import List


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
        universal_agent_data = step_data[step_data["Radius of Influence"] >= np.sqrt(2)]
        ids = universal_agent_data.index.get_level_values("AgentID")
        positions = universal_agent_data["Position"]
        x = [pos[0] for pos in positions]
        y = [pos[1] for pos in positions]
        colors = [colormap(id % 5) for id in ids]
        facecolors = [
            c[:3] + (v,)  # add transparency to color depending on visibility factor
            for c, v in zip(colors, universal_agent_data["Visibility Factor"])
        ]

        # draw symbols
        paths = ax.scatter(
            x, y, edgecolors=colors, s=50, marker="d", facecolor=facecolors
        )
        step_artists.append(paths)

        ### draw other agents, showing their radius of influence
        normal_agent_data = step_data[step_data["Radius of Influence"] < np.sqrt(2)]
        ids = normal_agent_data.index.get_level_values("AgentID")
        positions = normal_agent_data["Position"]
        x = [pos[0] for pos in positions]
        y = [pos[1] for pos in positions]
        colors = [colormap(id % 5) for id in ids]
        facecolors = [
            c[:3] + (v,)  # add transparency to color depending on visibility factor
            for c, v in zip(colors, normal_agent_data["Visibility Factor"])
        ]
        influence_radii = normal_agent_data["Radius of Influence"]

        # draw symbols
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

                # whether target was within the radius of influence of the attacker
                attack_possible = not np.isnan(
                    step_action_data["attack_successful"].iat[0]
                )

                arrow = ax.arrow(
                    x=a_x,
                    y=a_y,
                    dx=t_x - a_x,
                    dy=t_y - a_y,
                    length_includes_head=True,
                    width=0.005,
                    head_width=0.03,
                    head_length=0.03,
                    ls="solid" if attack_possible else "dashed",
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


def plot_streak_length_distribution(streaks, ax=None, **params):
    """
    Visualise distribution of attack streak lengths on a log-log scale.
    An attack streak is defined as successive time steps when an attack occurs
    (whether successful or not).

    Parameters:
    streaks: calculated streaks, as returned by analyse.count_attack_streaks
    ax: a Matplotlib Axes. If provided, plotting will be done on the Axes
    **params: model parameter values used for the simulation. These will be
              displayed under the plot as a caption.
    """
    create_new_axes = ax is None
    if create_new_axes:
        fig, ax = plt.subplots(constrained_layout=True, figsize=(5, 5))

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


def plot_particles(particles: List[ipomdp_solver.Particle], model: universe.Universe):
    """
    Plots the particles in the list. This creates n_agents subplots, each depicting the
    state of the corresponding agent. The horizontal axes show the age of the
    civilisations and the vertical axis corresponds to the visibility factor.

    If particles have weights, they are visually represented by the size of the points.

    If the model uses sigmoid growth for agents, the horizontal axis corresponds to
    time until takeoff (takeoff_time - age) as this is what determines technology
    levels.
    """

    n_agents = particles[0].state.shape[0]
    fig, axs = plt.subplots(
        nrows=1, ncols=n_agents, constrained_layout=True, figsize=(n_agents * 4, 4)
    )

    states = np.stack(tuple(p.state for p in particles), axis=0)
    weights = tuple(p.weight for p in particles)

    show_weights = sum(p.weight is None for p in particles) == 0

    if model.agent_growth == growth.sigmoid_growth:
        x_vals = growth.tech_level(state=states, model=model)
        # x_vals = states[..., 0] - states[..., 3]
        y_vals = states[..., 1]

    else:
        raise NotImplementedError()

    for ag_id, ax in enumerate(axs):
        ag_x = x_vals[:, ag_id]
        ag_y = y_vals[:, ag_id]

        # add some noise
        # ag_x += model.rng.uniform(low=-0.5, high=0.5, size=len(ag_x))
        ag_y += model.rng.uniform(low=-0.01, high=0.01, size=len(ag_y))

        if show_weights:
            min_size, max_size = 1, 40
            if max(weights) == 0:
                point_sizes = (min_size,) * len(particles)
            else:
                point_sizes = tuple(
                    min_size + (max_size - min_size) / max(weights) * w for w in weights
                )
            ax.scatter(
                x=ag_x,
                y=ag_y,
                s=point_sizes,
                c=tuple(
                    "red" if w == 0 else ("blue" if w > 1e-6 else "green")
                    for w in weights
                ),
            )
        else:
            ax.scatter(x=ag_x, y=ag_y)

        ax.set_xlim(0 - 0.02, 1 + 0.02)
        ax.set_ylim(y_vals.min() - 0.02, y_vals.max() + 0.02)
        ax.set_title(f"Agent {ag_id}")
        ax.grid()

    fig.supxlabel(
        "Technology level" if model.agent_growth == growth.sigmoid_growth else "Age"
    )
    fig.supylabel("Visibility factor")
    fig.suptitle("Particle states")

    plt.show()


def plot_particles_n2(
    particles: List[ipomdp_solver.Particle], model: universe.Universe
):
    """
    Plots the particles in the list. This creates n_agents subplots, each depicting the
    state of the corresponding agent. The horizontal axes show the age of the
    civilisations and the vertical axis corresponds to the visibility factor.

    If particles have weights, they are visually represented by the size of the points.

    If the model uses sigmoid growth for agents, the horizontal axis corresponds to
    time until takeoff (takeoff_time - age) as this is what determines technology
    levels.
    """

    n_agents = particles[0].state.shape[0]

    assert n_agents == 2

    fig, ax = plt.subplots(constrained_layout=True, figsize=(4, 4))

    states = np.stack(tuple(p.state for p in particles), axis=0)
    weights = tuple(p.weight for p in particles)

    show_weights = sum(p.weight is None for p in particles) == 0

    tech_levels = growth.tech_level(state=states, model=model)
    x_vals = tech_levels[:, 0]
    y_vals = tech_levels[:, 1]

    if show_weights:
        min_size, max_size = 1, 40
        if max(weights) == 0:
            point_sizes = (min_size,) * len(particles)
        else:
            point_sizes = tuple(
                min_size + (max_size - min_size) / max(weights) * w for w in weights
            )
        ax.scatter(
            x=x_vals,
            y=y_vals,
            s=point_sizes,
            c=tuple(
                "red" if w == 0 else ("blue" if w > 1e-6 else "green") for w in weights
            ),
        )
    else:
        ax.scatter(x=x_vals, y=y_vals)

    ax.set_xlim(0 - 0.02, 1 + 0.02)
    ax.set_ylim(0 - 0.02, 1 + 0.02)
    ax.set_title(f"Particle states")
    ax.set_xlabel("Technology level of agent 0")
    ax.set_ylabel("Technology level of agent 1")
    ax.grid()

    plt.show()


def plot_tree_particle_counts(
    data: pd.DataFrame,  # columns "repetition_id", "depth", "n_particles"
    ax: plt.Axes = None,
    show_individual_counts=True,
    summary_metrics=["avg", "max"],
    label: str = "",
    color=None,
):
    """
    Plot particle counts in the different nodes of the tree.
    Horizontal axis shows node depth while vertical axis displays the
    number of particles as
    i) individual node counts and
    ii) average at that depth
    iii) maximum at that depth
    """
    create_new_axes = ax is None
    if create_new_axes:
        fig, ax = plt.subplots(constrained_layout=True)

    n_repetitions = len(data["repetition_id"].unique())

    if show_individual_counts:
        # plot all values
        sctr = ax.scatter(
            x="depth",
            y="n_particles",
            data=data,
            marker="_",
            color=color,
            label="node particle count",
        )

    if "avg" in summary_metrics:
        # calculate the average for every depth
        avg_count = (
            data.groupby(["repetition_id", "depth"])
            .mean()
            .groupby("depth")
            .mean()
            .iloc[:, 0]
        )

        color = sctr.get_facecolor() if show_individual_counts else color

        avg_line = ax.plot(
            avg_count,
            label=f"average ({label})",
            color=color,
        )

        # add error bars if applicable
        if n_repetitions > 1:
            # determine error margins
            mean_error_margins = (
                data.groupby(["repetition_id", "depth"])
                .mean()
                .groupby("depth")
                .aggregate(lambda x: analyse.t_confidence_interval(x)[1])
                .iloc[:, 0]
            )

            ax.errorbar(
                x=mean_error_margins.index,
                y=avg_count,
                yerr=mean_error_margins,
                fmt="none",
                ecolor="red" if show_individual_counts else avg_line[0].get_color(),
                capsize=3,
            )

    if "max" in summary_metrics:
        # calculate the max for every depth
        max_count = (
            data.groupby(["repetition_id", "depth"])
            .max()
            .groupby("depth")
            .mean()
            .iloc[:, 0]
        )

        color = avg_line[0].get_color() if "avg" in summary_metrics else color

        max_line = ax.plot(
            max_count,
            label=f"maximum ({label})",
            color=color,
            linestyle="dashed",
        )

        # add error bars if applicable
        if n_repetitions > 1:
            # determine error margins
            max_error_margins = (
                data.groupby(["repetition_id", "depth"])
                .max()
                .groupby("depth")
                .aggregate(lambda x: analyse.t_confidence_interval(x)[1])
                .iloc[:, 0]
            )

            ax.errorbar(
                x=max_error_margins.index,
                y=max_count,
                yerr=max_error_margins,
                fmt="none",
                ecolor="red" if show_individual_counts else max_line[0].get_color(),
                capsize=3,
            )

    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.set_xlabel("Node depth")
    ax.set_ylabel("Number of particles")
    ax.set_title(f"Numbers of particles in nodes")
    ax.set_yscale("symlog")
    ax.grid(visible=True)
    ax.legend()

    if create_new_axes:
        plt.show()


def plot_tree_fraction_nodes_searched(
    data: pd.DataFrame,  # columns "repetition_id", "depth", "n_particles"
    n_possible_actions: int,
    ax: plt.Axes = None,
    label: str = "",
):
    """
    Plot fraction of nodes explored at each depth.
    Horizontal axis shows node depth while vertical axis displays the fraction.
    """
    create_new_axes = ax is None
    if create_new_axes:
        fig, ax = plt.subplots(constrained_layout=True)

    n_repetitions = len(data["repetition_id"].unique())

    def calc_frac_nodes(chunk):
        """Calculates the fraction of nodes explored for a chunk at a given depth"""
        depth = chunk.index.get_level_values("depth")[0]
        return chunk / (n_possible_actions + 1) ** depth

    # average fraction of nodes searched
    frac_grouped = (
        data.groupby(["depth", "repetition_id"])
        .size()
        .groupby("depth")
        .transform(calc_frac_nodes)
        .groupby("depth")
    )
    frac_avg = frac_grouped.mean()

    # plot
    frac_line = ax.plot(frac_avg, label=f"{label}")

    if n_repetitions > 1:
        frac_error_margin = frac_grouped.aggregate(
            lambda chunk: analyse.t_confidence_interval(chunk)[1]
        )

        ax.errorbar(
            x=frac_error_margin.index,
            y=frac_avg,
            yerr=frac_error_margin,
            fmt="none",
            capsize=3,
            color=frac_line[0].get_color(),
        )

    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.set_xlabel("Node depth")
    ax.set_ylabel("Fraction of nodes explored")
    ax.set_title("Fraction of nodes explored")
    ax.update_datalim([(0, 0)])
    ax.legend()
    ax.grid(visible=True)

    if create_new_axes:
        plt.show()


def plot_fraction_successful_lower_tree_queries(
    data: pd.DataFrame,
    ax: plt.Axes = None,
    label: str = "x",
    title: str = "Success rate for queries to lower trees",
):
    """
    Plots the fraction of successful queries to the tree

    Keyword arguments:
    data: A Pandas data frame with the following columns:
            - x
            - n_queries
            - prop_successful
            - prop_missing_node
            - prop_diverged_belief
            - prop_some_actions_unexpanded
    ax: The Matplotlib Axes to plot onto
    label: Label for the 'x' column
    title: Title of plot
    """

    create_new_axes = ax is None
    if create_new_axes:
        fig, ax = plt.subplots(constrained_layout=True)

    bar_width = 0.5
    columns = (
        "prop_successful",
        "prop_some_actions_unexpanded",
        "prop_diverged_belief",
        "prop_missing_node",
    )
    colours = ("green", "yellow", "orange", "red")
    labels = ("successful", "unexpanded actions", "diverged belief", "no node")
    x_vals = []

    for i, (x, x_group) in enumerate(data.groupby("x")):
        bar_bottom = 0
        x_vals.append(x)

        for column, colour in zip(columns, colours):
            # calculate confidence interval
            mean, error_margin = analyse.t_confidence_interval(x_group[column])

            # plot bar
            ax.bar(
                x=i,
                height=mean,
                width=bar_width,
                bottom=bar_bottom,
                yerr=None if np.isnan(error_margin) else error_margin,
                capsize=4,
                color=colour,
            )

            # update bar bottom
            bar_bottom += mean

    # adjust plot
    ax.set_xticks(ticks=list(range(len(x_vals))), labels=x_vals)
    ax.set_xlabel(label)
    ax.set_ylabel("proportion")
    ax.grid(visible=True)
    ax.legend(
        handles=[
            Patch(facecolor=colour, label=label)
            for colour, label in zip(colours, labels)
        ]
    )
    ax.set_title(title)

    if create_new_axes:
        plt.show()
