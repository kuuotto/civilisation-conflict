# %%
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import ArtistAnimation
from scipy.stats import binom

import mesa
# %%

def influence_radius(tech_level):
    """
    Gives the radius of influence as a function of the technology level
    of the civilisation.

    TODO: define more rigorously
    """
    return 0.1*np.tan(tech_level * (np.pi / 2))

def inv_influence_radius(inf_radius):
    """
    Gives the tech level corresponding to a given influence radius
    """
    return (2/np.pi) * np.arctan(10*inf_radius)

def sigmoid_growth(time, speed, takeoff_time):
    """
    Gives the current technology level for the agent. Assumes a
    sigmoid-shaped growing curve for the technological capabilities of
    civilisations.
    """
    return 1/(1+np.exp(-speed*(time - takeoff_time)))

def prob_smaller(rv1, rv2, n_samples=1000):
    """
    Uses Monte Carlo sampling to determine the probability for the event
    rv1 < rv2.

    Arguments:
    rv1, rv2: callables that return i.i.d. random samples from the variables,
              should have a keyword argument "size" for number of samples
    n_samples: number of samples to draw from each random variable
    """
    sample1 = rv1(size=n_samples)
    sample2 = rv2(size=n_samples)
    return np.mean(sample1 < sample2)

class Civilisation(mesa.Agent):
    """An agent represeting a single civilisation in the universe"""

    def __init__(self, unique_id, model, growth, **growth_kwargs) -> None:
        super().__init__(unique_id, model)

        # initialise reset time, which is updated if the civilisation is 
        # destroyed
        self.reset_time = 0

        # initialise visibility factor -- the civilisation can choose to hide 
        # which decreases its apparent tech level (=technosignature)
        self.visibility_factor = 1

        # keep track of the last time this civilisation acted
        # this is used for preventing hostility belief updates when this
        # civilisation destroyed another
        self.last_acted = -1

        # by default, choose growth parameters randomly
        if len(growth_kwargs) < 1 and growth==sigmoid_growth:
            growth_kwargs = {'speed': self.model.random.uniform(2, 4),
                             'takeoff_time': self.model.random.randrange(1, 20)}
        
        # save the growth function
        self.growth = lambda time: growth(time, **growth_kwargs)

        # initialise own tech level
        self.step_tech_level()

        # initialise a dictionary of neighbour tech level beliefs
        self.tech_beliefs = dict()

        # initialise a dictionary of neighbour hostility beliefs
        self.hostility_beliefs = dict()
        

    def step_tech_level(self):
        """Update own tech level"""
        # tech level is discretised
        new_tech_level = self.growth(self.model.schedule.time - self.reset_time)
        new_tech_level = np.round(new_tech_level, 1)

        # update tech level and calculate new influence radius
        self.tech_level = new_tech_level
        self.influence_radius = influence_radius(new_tech_level)

    def step_update_beliefs(self):
        """
        Update beliefs regarding technology level and hostility of 
        neighbours.
        """
        neighbours = self.get_neighbouring_agents()

        ### update technology beliefs
        # also estimate whether a civilisation has been destroyed;
        # if there is a civilisation that has been destroyed with a high
        # probability, we update hostility beliefs. If there are multiple,
        # we update based on the one that is the most likely to have been
        # destoryed

        new_tech_beliefs = dict()
        old_tech_beliefs = self.tech_beliefs
        destroyed_civilisation, max_prob_destr = None, 0

        for neighbour in neighbours:

            # add Gaussian noise to the technosignature (which is a product
            # of the technology level and the visibility factor)
            noisy_tech_level = (neighbour.tech_level*neighbour.visibility_factor + 
                                self.random.gauss(mu=0, sigma=0.05))
            noisy_tech_level = np.clip(noisy_tech_level, 0, 1)
            new_tech_level = binom(n=10, p=noisy_tech_level)
            new_tech_beliefs[neighbour] = new_tech_level

            ### next, estimate if this civilisation has been destoryed during
            ### the previous round
            # if we don't have previous beliefs about this neighbour's
            # capabilities, we can't say if it has been destroyed
            if neighbour not in old_tech_beliefs:
                continue

            # don't update hostility beliefs if we acted last round, because
            # we know we were the perpetrator
            if self.last_acted == self.model.schedule.time - 1:
                continue

            # calculate probability that the neighbour was destroyed, i.e.
            # that the new tech level is lower than the old
            old_tech_level = old_tech_beliefs[neighbour]
            prob_destr = prob_smaller(new_tech_level.rvs, old_tech_level.rvs)

            # if this civilisation is the most likely one to have been 
            # destroyed, save it
            if prob_destr > max_prob_destr:
                destroyed_civilisation = neighbour
                max_prob_destr = prob_destr

        # save changes
        self.tech_beliefs = new_tech_beliefs

        ### update hostility beliefs

        # TODO: this is where prior hostility beliefs are currently defined
        old_hostility_beliefs = self.hostility_beliefs
        new_hostility_beliefs = {neighbour: 0.01 
                                            if neighbour not in old_hostility_beliefs 
                                            else old_hostility_beliefs[neighbour] 
                                            for neighbour in neighbours}


        # if there are no civilisations that we are very confident have been
        # destroyed, don't update hostility beliefs
        if not destroyed_civilisation or max_prob_destr < 0.9:
            self.hostility_beliefs = new_hostility_beliefs
            return

        self.dprint(f"Updating hostility beliefs because",
                    f"{destroyed_civilisation.unique_id} was destroyed")

        destr_tech_level = old_tech_beliefs[destroyed_civilisation]

        # calculate the capabilities of all neighbours
        perp_values = dict()
        for perpetrator in neighbours:

            if perpetrator==destroyed_civilisation:
                continue

            # if we don't have information about this possible perpetrator
            # from the last time step, we can't say anything about whether
            # they could've been the culprit
            if perpetrator not in old_tech_beliefs:
                continue

            assert(perpetrator in old_hostility_beliefs)

            perpetrator_tech_level = old_tech_beliefs[perpetrator]

            # tech level required for perpetrator to be able to reach the
            # destoryed civilisation
            distance = self.model.space.get_distance(destroyed_civilisation.pos, 
                                                     perpetrator.pos)
            req_tech_level = inv_influence_radius(distance)

            # probability that perpetrator 
            # i) had a higher tech level than the destroyed neighbour, and
            # ii) could reach the destroyed neighbour
            prob_capable = prob_smaller(lambda size: np.maximum(
                                            destr_tech_level.rvs(size=size), 
                                            req_tech_level),
                                        perpetrator_tech_level.rvs)
            perp_values[perpetrator] = (prob_capable, 
                                        old_hostility_beliefs[perpetrator])

        # calculate the updated hostility beliefs
        if len(perp_values) > 0:
            denominator = np.sum([c*h for perp, (c, h) in perp_values.items()])

            if denominator > 0:
                new_hostility_beliefs.update({perp: h + c*h/denominator - c*h**2 / denominator 
                                              for perp, (c, h) in perp_values.items()})

                self.dprint(f"New hostility beliefs:", 
                    *(f"{agent.unique_id}: {old_hostility_beliefs[agent]:.3f} -> {new_belief:.3f}" 
                      for agent, new_belief in new_hostility_beliefs.items() 
                      if agent in old_hostility_beliefs))
            else:
                self.dprint(f"Cannot update hostility beliefs because no neighbour is deemed capable of destroying {destroyed_civilisation.unique_id}")

        # save changes
        self.hostility_beliefs = new_hostility_beliefs

    def step_act(self):
        """
        The agent chooses an action. Possible actions include attacking a 
        neighbour, decreasing the civilisation's own technosignature (technology
        level perceived by others) and doing nothing.

        Currently, one of these options is chosen randomly. In the future, this
        will change to choosing an action based on the equilibria of 
        hypothetical games.
        """
        neighbours = self.get_neighbouring_agents()
        actions = neighbours + ['no action']
        if self.visibility_factor == 1:
            actions += ['hide']
        action = self.random.choice(actions)

        if isinstance(action, Civilisation):
            self.dprint(f"Attacks {action.unique_id}")
            action.attack(self)
        elif action=="hide":
            # TODO: define hiding more rigorously
            self.visibility_factor = 0.5
            self.dprint(f"Hides")
        elif action=="no action":
            self.dprint("No action")

        # update last acted time
        self.last_acted = self.model.schedule.time


    def attack(self, attacker):
        """
        This is called when the civilisation is attacked. The target is
        destroyed iff the attacker is stronger than the target.

        In case of a tie, the target is destroyed with a 50% probability.
        """
        if (attacker.tech_level > self.tech_level or 
            (attacker.tech_level == self.tech_level and
             self.model.random.random() > 0.5)):

            # civilisation is destroyed
            self.reset_time = self.model.schedule.time
            self.visibility_factor = 1
            self.dprint(f"Attack successful ({attacker.tech_level:.3f}",
                        f"> {self.tech_level:.3f})")
        else:

            # update hostility beliefs after a failed attack
            self.hostility_beliefs[attacker] = 1
            self.dprint(f"Attack failed ({attacker.tech_level:.3f}",
                        f"< {self.tech_level:.3f})")
                

    def get_neighbouring_agents(self):
        return self.model.space.get_neighbors(pos=self.pos, 
                                              radius=self.influence_radius, 
                                              include_center=False)     

    def dprint(self, *message):
        """Prints message to the console if debugging flag is on"""
        if self.model.debug:
            print(f"t={self.model.schedule.time}, {self.unique_id}:", *message)

class Universe(mesa.Model):

    def __init__(self, num_agents, toroidal_space=False, 
                 agent_growth=sigmoid_growth, debug=False) -> None:
        
        self.schedule = SingleActivation(self, 
                                         update_methods=['step_tech_level', 
                                                         'step_update_beliefs'], 
                                         step_method='step_act')
        self.space = mesa.space.ContinuousSpace(x_max=1, y_max=1, 
                                                torus=toroidal_space)

        # add agents
        for i in range(num_agents):
            agent = Civilisation(i, self, agent_growth)
            self.schedule.add(agent)

            # place agent in a randomly chosen position
            # TODO: consider the distribution of stars
            x, y = self.random.random(), self.random.random()
            self.space.place_agent(agent, (x, y))

        # initialise data collection
        self.datacollector = mesa.DataCollector(
            model_reporters={}, agent_reporters={
                "Technology":  "tech_level", 
                "Radius of Influence": "influence_radius",
                "Position": "pos"
            })

        # whether to print debug prints
        self.debug = debug

    def step(self):
        """Advance the model by one step."""
        self.datacollector.collect(self)
        self.schedule.step()

class SingleActivation(mesa.time.BaseScheduler):
    """
    A scheduler which first calls the update method(s) for every agent (if
    there are multiple, this is done in stages: the first update method
    is executed for every agent before moving on to the next). Finally,
    a random agent is activated to perform a step method.
    """

    def __init__(self, model: mesa.Model, update_methods: list[str],
                 step_method: str) -> None:
        """
        Create an empty Single Activation schedule.

        Args:
            model: Model object associated with the schedule.
            update_methods: List of strings of names of stages to run, in the
                            order to run them in.
            step_method: The name of the step method to be activated in a 
                         single randomly chosen agent.
        """
        super().__init__(model)
        self.update_methods = update_methods
        self.step_method = step_method

    def step(self) -> None:
        """
        Executes the update method(s) of all agents (if multiple, in stages)
        and then the step method of a randomly chosen agent.
        """
        # To be able to remove and/or add agents during stepping
        # it's necessary to cast the keys view to a list.
        agent_keys = list(self._agents.keys())

        # run update methods in stages for all agents
        for update_method in self.update_methods:
            for agent_key in agent_keys:
                if agent_key in self._agents:
                    # run update method
                    getattr(self._agents[agent_key], update_method)()

            # We recompute the keys because some agents might have been removed
            # in the previous loop.
            agent_keys = list(self._agents.keys())

        # finally, choose a random agent to step
        agent = self.model.random.choice(agent_keys)
        getattr(self._agents[agent], self.step_method)()

        # increase time
        self.time += 1
        self.steps += 1

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

    # if there are multiple steps, animate
    if len(steps) > 1:
        ani = ArtistAnimation(fig=fig, artists=artists, interval=200, repeat=True)
        return ani

    return fig, ax


# %%

# create a test universe with five civilisations
model = Universe(5, debug=True)
for i in range(50):
    model.step()

# retrieve and visualise data
data = model.datacollector.get_agent_vars_dataframe() 
vis = draw_universe(data=data)
plt.show()

# %%
