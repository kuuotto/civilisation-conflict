# %%
import numpy as np
import mesa

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
    exponent = -speed * (time - takeoff_time)
    # avoid overflows in exp
    if exponent > 10:
        return 0
    return 1/(1+np.exp(exponent))

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

    def __init__(self, unique_id, model, decision_making, growth,
                 **growth_kwargs) -> None:
        """
        Initialise a civilisation.

        Keyword arguments:
        unique_id: integer, uniquely identifies this civilisation
        model: a Universe object which this civilisation belongs to
        decision_making: either "random" or "targeted" (see step_act for their
                         meaning)
        growth: a callable of the form growth(time, **growth_kwargs)
        **growth_kwargs: keyword arguments to the growth function. Can also
                         supply speed_range and takeoff_time_range in case 
                         growth if sigmoid_growth, in which case the speed
                         and takeoff time are randomly chosen from these ranges
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

        # keep track of the last time this civilisation acted
        # this is used for preventing hostility belief updates when this
        # civilisation destroyed another
        self.last_acted = -1

        if (growth == sigmoid_growth and 
            "speed" not in growth_kwargs and 
            "takeoff_time" not in growth_kwargs):
            # for sigmoid growth, parameters are chosen randomly from given
            # ranges

            if len(growth_kwargs) == 0:
                speed_range = (2, 4)
                takeoff_time_range = (1, 20)
            elif "speed_range" in growth_kwargs and "takeoff_time_range" in growth_kwargs:
                speed_range = growth_kwargs["speed_range"]
                takeoff_time_range = growth_kwargs["takeoff_time_range"]
            else:
                raise Exception("Sigmoid growth parameters are incorrect")

            growth_kwargs = {'speed': self.rng.uniform(*speed_range),
                             'takeoff_time': self.rng.integers(
                                        *takeoff_time_range, endpoint=True)}
        
        # save parameters
        self.growth = growth
        self.growth_kwargs = growth_kwargs
        self.decision_making = decision_making
    
        # initialise own tech level
        self.step_tech_level()

        # initialise a dictionary of neighbour tech level beliefs
        self.tech_beliefs = dict()

        # initialise a dictionary of neighbour hostility beliefs
        self.hostility_beliefs = dict()

        # initialise own state (see a description of the state in the 
        # get_state method)
        self._init_state()
        
    def step_tech_level(self):
        """Update own tech level"""
        # tech level is discretised
        new_tech_level = self.growth(self.model.schedule.time - self.reset_time, 
                                     **self.growth_kwargs)
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
                                self.rng.normal(loc=0, scale=0.05))
            noisy_tech_level = np.clip(noisy_tech_level, 0, 1)
            new_tech_level = TechBelief(p=noisy_tech_level, rng=self.model.rng)
            new_tech_beliefs[neighbour] = new_tech_level

            ### next, estimate if this civilisation has been destroyed during
            ### the previous round

            # don't update hostility beliefs if we acted last round, because
            # we know we were the perpetrator (and we already reset the
            # hostility belief regarding the target)
            if self.last_acted == self.model.schedule.time - 1:
                continue

            # if we don't have previous beliefs about this neighbour's
            # capabilities, we can't say if it has been destroyed
            if neighbour not in old_tech_beliefs:
                continue

            # calculate probability that the neighbour was destroyed, i.e.
            # that the new tech level is lower than the old (and lower than 0.1)
            old_tech_level = old_tech_beliefs[neighbour]
            prob_destr = prob_smaller(new_tech_level.rvs, 
                                      lambda size: np.minimum(
                                        old_tech_level.rvs(size=size), 0.1))

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
        new_hostility_beliefs = {neighbour: self.model.hostility_belief_prior 
                                 if neighbour not in old_hostility_beliefs 
                                 else old_hostility_beliefs[neighbour] 
                                 for neighbour in neighbours}


        # the civilisation with the highest probability of having been
        # destroyed is deemed destroyed stochastically with that probability
        if not destroyed_civilisation or self.rng.random() > max_prob_destr:
            self.hostility_beliefs = new_hostility_beliefs
            return

        self.dprint(f"Updating hostility beliefs because",
                    f"{destroyed_civilisation} was destroyed",
                    f"(prob. {max_prob_destr:.3f})")

        # reset our belief about the hostility of the destroyed civilisation
        new_hostility_beliefs[destroyed_civilisation] = self.model.hostility_belief_prior

        # tech level of destroyed civilisation
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
                                         perpetrator_tech_level.rvs, 
                                         include_equal=True)
            perp_values[perpetrator] = (prob_capable, 
                                        old_hostility_beliefs[perpetrator])

        # calculate the updated hostility beliefs
        if len(perp_values) > 0:
            denominator = sum([c*h for perp, (c, h) in perp_values.items()])

            if denominator > 0:
                new_hostility_beliefs.update({perp: h + c*h/denominator - c*h**2 / denominator 
                                              for perp, (c, h) in perp_values.items()})

                self.dprint(f"New hostility beliefs:", 
                    *(f"{agent}: {old_hostility_beliefs[agent]:.3f} -> {new_belief:.3f}" 
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

        Currently, there are two strategies for choosing actions (determined
        by self.decision_making):
        - "random"
        - "targeted": only attack if we are sure a neighbour is hostile and 
                      we believe we have higher than 50% chance of destroying 
                      it. Also attack randomly with a 10% probability. 
                      Otherwise choose randomly between hiding and no action

        In the future, there will be an option to choose actions based on the 
        equilibria of hypothetical games.
        """
        neighbours = self.get_neighbouring_agents()

        ### Choose an action
        if self.decision_making == "random":

            actions = neighbours + ['no action']
            if self.visibility_factor == 1:
                actions += ['hide']
            action = self.rng.choice(actions)

        elif self.decision_making == "targeted":

            # neighbours we are sure are hostile
            hostile_neighbours = {nbr for nbr in neighbours 
                                  if self.hostility_beliefs[nbr] == 1}

            # perceived chance of successfully destroying them
            prob_successful_attack = {hnbr: prob_smaller(
                                                self.tech_beliefs[hnbr].rvs, 
                                                lambda size: self.tech_level) 
                                      for hnbr in hostile_neighbours}

            # if (len(hostile_neighbours) > 0 and 
            #     max(prob_successful_attack.values()) >= 0.5):
            if len(hostile_neighbours) > 0:
                # if we are relatively sure we are technologically superior,
                # attack
                action = max(prob_successful_attack, 
                             key=lambda nbr: prob_successful_attack[nbr])
                self.dprint(f"Targeted attack at {action}")
            else:

                # random attack with 10% probability
                if len(neighbours) > 0 and self.rng.random() < 0.1:
                    action = self.rng.choice(neighbours)
                else:
                    action = self.rng.choice(['hide', 'no action'])

        ### Perform and log action
        if isinstance(action, Civilisation):
            self.dprint(f"Attacks {action.unique_id}")
            action.attack(self)

        elif action=="hide":
            # TODO: define hiding more rigorously
            self.visibility_factor *= self.model.visibility_multiplier

            self.dprint(f"Hides (tech level {self.tech_level},",
                        f"visibility {self.visibility_factor:.3f})")
            self.model.datacollector.add_table_row('actions',
                {'time': self.model.schedule.time,
                 'actor': self.unique_id,
                 'action': 'h'}, 
                ignore_missing=True)
                
        elif action == "no action" or action == "hide":
            self.dprint("-")
            # log
            self.model.datacollector.add_table_row('actions',
                {'time': self.model.schedule.time,
                 'actor': self.unique_id,
                 'action': '-'}, 
                ignore_missing=True)

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
             self.rng.random() > 0.5)):

            # civilisation is destroyed
            self.reset_time = self.model.schedule.time
            self.step_tech_level()
            self.visibility_factor = 1
            # tech and hostility beliefs will be reset at the beginning
            # of the next round, because we will have no neighbours

            # attacker gets to know that the target was destroyed.
            # in particular, it resets its hostility beliefs
            # regarding the target
            attacker.hostility_beliefs[self] = self.model.hostility_belief_prior

            self.dprint(f"Attack successful ({attacker.tech_level:.3f}",
                        f"> {self.tech_level:.3f})")
            result = True
        else:
            # update hostility beliefs after a failed attack
            self.hostility_beliefs[attacker] = 1
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
        if self.growth == sigmoid_growth:
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
        if self.growth == sigmoid_growth:
            self._state[0] = self.model.schedule.time - self.reset_time
            self._state[1] = self.visibility_factor
            self._state[2] = self.growth_kwargs['speed']
            self._state[3] = self.growth_kwargs['takeoff_time']
        else:
            raise NotImplementedError()

        return self._state

    def __str__(self):
        return f"{self.unique_id}"

class Universe(mesa.Model):

    def __init__(self, num_agents, toroidal_space=False, 
                 agent_growth=sigmoid_growth, decision_making="random", 
                 hostility_belief_prior=0.01, visibility_multiplier=0.7, 
                 debug=False, rng_seed=0, 
                 **agent_growth_kwargs) -> None:
        
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
        for i in range(num_agents):
            agent = Civilisation(i, self, growth=agent_growth, 
                                 decision_making=decision_making,
                                 **agent_growth_kwargs)
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

        # save parameters
        self.debug = debug
        self.hostility_belief_prior = hostility_belief_prior
        self.visibility_multiplier = visibility_multiplier

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

