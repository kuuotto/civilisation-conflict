# %%
import mesa
import numpy as np
from model.growth import influence_radius, sigmoid_growth
from model.ipomdp import (optimal_action, sample_init, sample_observation,
                          update_beliefs_1)


def prob_smaller(rv1, rv2, n_samples=100, include_equal=False):
    """
    Uses Monte Carlo sampling to determine the probability for the event
    rv1 < rv2.

    Arguments:
    rv1, rv2: callables that return i.i.d. random samples from the variables,
              should have a keyword argument "size" for number of samples.
              Should return a NumPy array.
    n_samples: number of samples to draw from each random variable
    """
    sample1 = rv1(size=n_samples)
    sample2 = rv2(size=n_samples)
    return np.mean(sample1 <= sample2) if include_equal else np.mean(sample1 < sample2)

class TechBelief():
    """Helper class for converting distributions into the range [0,1]"""

    def __init__(self, rng, p, n=10):
        '''
        p is the p parameter of a binomial distribution, which is used to 
        represent technology beliefs
        '''
        self.p = p
        self.n = n
        self.rng = rng

    def rvs(self, size):
        return self.rng.binomial(n=self.n, p=self.p, size=size) / self.n

    def support():
        return np.linspace(0, 1, 11).round(1)

class Civilisation(mesa.Agent):
    """An agent represeting a single civilisation in the universe"""

    def __init__(self, unique_id, model) -> None:
        """
        Initialise a civilisation.

        Keyword arguments:
        unique_id: integer, uniquely identifies this civilisation
        model: a Universe object which this civilisation belongs to
        """
        super().__init__(unique_id, model)

        # add reference to model's rng
        self.rng = model.rng

        # initialise reset time, which is updated if the civilisation is 
        # destroyed
        self.reset_time = 0

        # initialise visibility factor -- the civilisation can choose to hide 
        # which decreases its apparent tech level (=technosignature)
        self.visibility_factor = 1

        # if model was supplied with ranges of values for sigmoid growth,
        # randomly choose civilisation's parameter values from those ranges
        if (model.agent_growth == sigmoid_growth and
                "speed_range" in model.agent_growth_params and
                "takeoff_time_range" in model.agent_growth_params):

            speed_range = model.agent_growth_params["speed_range"]
            takeoff_time_range = model.agent_growth_params["takeoff_time_range"]

            self.agent_growth_params = {
                'speed': self.rng.uniform(*speed_range),
                'takeoff_time': self.rng.integers(*takeoff_time_range,
                                                  endpoint=True)}

        else:
            # save agent growth parameters from the model
            self.agent_growth_params = model.agent_growth_params

        # initialise own tech level
        self.step_tech_level()

        # initialise own state (see a description of the state in the 
        # get_state method)
        self._init_state()

        # initialise beliefs
        self.belief = sample_init(n_samples=model.n_belief_samples,
                                  level=1,
                                  agent=unique_id,
                                  agent_state=self.get_state(),
                                  model=model)
        
        # keep track of previous action
        self.previous_agent_action = None

    def step_tech_level(self):
        """Update own tech level"""
        new_tech_level = self.model.agent_growth(
                            self.model.schedule.time - self.reset_time, 
                            **self.agent_growth_params)

        # update tech level and calculate new influence radius
        self.tech_level = new_tech_level
        self.influence_radius = influence_radius(new_tech_level)

    def step_update_beliefs(self):
        """
        Update beliefs regarding technology level and hostility of 
        neighbours.
        """
        # determine the observation received by agent
        obs = sample_observation(state=self.model.get_state(),
                                 action=self.model.previous_action,
                                 agent=self.unique_id,
                                 model=self.model,
                                 n_samples=1)

        # update beliefs
        self.belief = update_beliefs_1(belief=self.belief,
                                       agent_action=self.previous_agent_action,
                                       agent_observation=obs,
                                       agent=self.unique_id,
                                       model=self.model)

        # previous action is no longer needed
        self.previous_agent_action = None

    def step_act(self):
        """
        The agent chooses an action. Possible actions include attacking a 
        neighbour, decreasing the civilisation's own technosignature (technology
        level perceived by others) and doing nothing.

        Currently, there are two strategies for choosing actions (determined
        by the model's decision_making attribute):
        - "random"
        - "ipomdp"
        """
        ### Choose an action
        if self.model.decision_making == "ipomdp":
            action = optimal_action(belief=self.belief,
                                    agent=self.unique_id,
                                    actor=self.unique_id,
                                    time_horizon=self.model.planning_time_horizon,
                                    level=1,
                                    model=self.model,
                                    return_sample=True)[0]['type']
        elif self.model.decision_making == "random":
            
            neighbours = self.get_neighbouring_agents()
            action = self.rng.choice(['-', 'hide'] + 
                                     [nbr.unique_id for nbr in neighbours])
        else:
            raise NotImplementedError("Only 'random' and 'ipomdp' are supported")

        ### Perform and log action
        if isinstance(action, int):
            self.dprint(f"Attacks {action}")
            self.model.schedule.agents[action].attack(self)

        elif action == "hide":
            self.visibility_factor *= self.model.visibility_multiplier

            self.dprint(f"Hides (tech level {self.tech_level:.3f},",
                        f"visibility {self.visibility_factor:.3f})")
            self.model.datacollector.add_table_row('actions',
                {'time': self.model.schedule.time,
                 'actor': self.unique_id,
                 'action': 'h'}, 
                ignore_missing=True)
                
        elif action == "-":
            self.dprint("-")
            # log
            self.model.datacollector.add_table_row('actions',
                {'time': self.model.schedule.time,
                 'actor': self.unique_id,
                 'action': '-'}, 
                ignore_missing=True)

        self.previous_agent_action = action
        self.model.previous_action = {'actor': self.unique_id,
                                      'type': action}

    def attack(self, attacker):
        """
        This is called when the civilisation is attacked. The target is
        destroyed iff the attacker is stronger than the target.

        In case of a tie, the target is destroyed with a 50% probability.
        """
        if (attacker.tech_level > self.tech_level or
                (attacker.tech_level == self.tech_level and
                 self.rng.random() > 0.5)):
                
            # civilisation is destroyed
            self.reset_time = self.model.schedule.time
            self.step_tech_level()
            self.visibility_factor = 1
            # beliefs will be reset at the beginning of the next round, because
            # we will have no neighbours

            self.dprint(f"Attack successful ({attacker.tech_level:.3f}",
                        f"> {self.tech_level:.3f})")
            result = True
        else:
            # failed attack
            self.dprint(f"Attack failed ({attacker.tech_level:.3f}",
                        f"< {self.tech_level:.3f})")
            result = False

        # log attack
        self.model.datacollector.add_table_row('actions',
                                               {'time': self.model.schedule.time,
                                                'actor': attacker.unique_id,
                                                'action': 'a',
                                                'attack_target': self.unique_id,
                                                'attack_successful': result})

    def get_neighbouring_agents(self):
        return self.model.space.get_neighbors(pos=self.pos, 
                                              radius=self.influence_radius, 
                                              include_center=False)     

    def dprint(self, *message):
        """Prints message to the console if debugging flag is on"""
        if self.model.debug:
            print(f"t={self.model.schedule.time}, {self.unique_id}:", *message)

    def _init_state(self):
        """Initialises the state array"""
        if self.model.agent_growth == sigmoid_growth:
            self._state = np.zeros(4)
        else:
            raise NotImplementedError()

    def get_state(self):
        """
        Updates self._state and returns it.
        
        The state consists of 4 numbers: 
        1. time since last destruction
        2. visibility factor
        3. growth speed
        4. growth takeoff time

        The last two are related to the specific growth model assumed, and
        will therefore be different with different growth types
        """
        if self.model.agent_growth == sigmoid_growth:
            self._state[0] = self.model.schedule.time - self.reset_time
            self._state[1] = self.visibility_factor
            self._state[2] = self.agent_growth_params['speed']
            self._state[3] = self.agent_growth_params['takeoff_time']
        else:
            raise NotImplementedError()

        return self._state

    def __str__(self):
        return f"{self.unique_id}"

class Universe(mesa.Model):

    def __init__(self, n_agents, agent_growth, agent_growth_params, rewards,
                 n_belief_samples, obs_noise_sd, belief_update_time_horizon, 
                 planning_time_horizon, action_dist_0, discount_factor, 
                 visibility_multiplier, decision_making,
                 toroidal_space=False, debug=False, rng_seed=0
                ) -> None:
        """
        Initialise a new Universe model.

        Keyword arguments:
        n_agents: the number of agents in the model
        agent_growth: the type of growth agents can undergo. A callable which
                      accepts a time step and additional keyword arguments 
                      defined in agent_growth_params.
        agent_growth_params: see above
        rewards: a dictionary of rewards with keys 'destroyed', 'hide' and 
                 'attack' and the rewards as values
        n_belief_samples: the number of samples (particles) in the beliefs of 
                          each agent. If n_belief_samples=10, each agent holds
                          10 samples representing their beliefs about the
                          environment state and for each of those, they hold
                          10 samples for each other agent representing their
                          beliefs about the others' beliefs about the 
                          environment state
        obs_noise_sd: standard deviation of technosignature observation noise
                      (which follows an unbiased normal distribution)
        belief_update_time_horizon: how many steps to look ahead when figuring
                                    out what others did when updating beliefs
        planning_time_horizon: how many steps to look ahead when planning our
                               own action
        action_dist_0: the method agents assume others use to figure out which
                       actions other agents choose. "random" means the others'
                       actions are chosen uniformly over the set of possible
                       choices.
        discount_factor: how much future time steps are discounted when 
                         determining the rational actions of agents
        visibility_multiplier: how much a single “hide” action multiplies the
                               current agent visibility factor by
        decision_making: the method used by agents to make decisions. Options
                         include "random" and "ipomdp" ("targeted" is 
                         deprecated)
        hostility_belief_prior: deprecated
        toroidal_space: whether to use a toroidal universe topology
        debug: whether to print detailed debug information while model is run
        rng_seed: seed of the random number generator. Fixing the seed allows
                  for reproducibility
        """
        # save parameters
        self.n_agents = n_agents
        self.agent_growth = agent_growth
        self.agent_growth_params = agent_growth_params
        self.rewards = rewards
        self.n_belief_samples = n_belief_samples
        self.obs_noise_sd = obs_noise_sd
        self.belief_update_time_horizon = belief_update_time_horizon
        self.planning_time_horizon = planning_time_horizon
        self.action_dist_0 = action_dist_0
        self.discount_factor = discount_factor
        self.visibility_multiplier = visibility_multiplier
        self.decision_making = decision_making
        self.debug = debug
        
        # initialise random number generator
        self.rng = np.random.default_rng(rng_seed)

        # initialise schedule and space
        self.schedule = SingleActivation(self, 
                                         update_methods=['step_tech_level', 
                                                         'step_update_beliefs'], 
                                         step_method='step_act')
        self.space = mesa.space.ContinuousSpace(x_max=1, y_max=1, 
                                                torus=toroidal_space)

        # add agents
        for i in range(n_agents):
            agent = Civilisation(i, self)
            self.schedule.add(agent)

            # place agent in a randomly chosen position
            # TODO: consider the distribution of stars
            x, y = self.rng.random(size=2)
            self.space.place_agent(agent, (x, y))

        # initialise data collection
        self.datacollector = mesa.DataCollector(
            agent_reporters={
                "Technology": "tech_level", 
                "Radius of Influence": "influence_radius",
                "Visibility Factor": "visibility_factor",
                "Position": "pos"
            }, 
            tables={'actions': ['time', 'actor', 'action', 'attack_target', 
                                'attack_successful']})

        # initialise model state (see a detailed description in the get_state
        # method)
        self._init_state()

        # keep track of the last action (initial value doesn't really matter)
        self.previous_action = {'actor': self.schedule.agents[0], 'type': '-'}

    def step(self):
        """Advance the model by one step."""
        self.datacollector.collect(self)
        self.schedule.step()

    def _init_state(self):
        """Initialise model state"""
        # get length of state of individual agent
        agent_state_len = self.schedule.agents[0]._state.size
        num_agents = len(self.schedule.agents)
        self._state = np.zeros((num_agents, agent_state_len))

    def get_state(self):
        """
        Update and return the current model state.

        Returns:
        a NumPy array of shape (n, k), where k is the length of the state
        description of a single agent
        """
        for i, agent in enumerate(self.schedule.agents):
            self._state[i] = agent.get_state()

        return self._state

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
        agent = self.model.rng.choice(agent_keys)
        getattr(self._agents[agent], self.step_method)()

        # increase time
        self.time += 1
        self.steps += 1

