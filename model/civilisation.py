import mesa
import numpy as np
from model import growth, ipomdp, ipomdp_solver, action

class Civilisation(mesa.Agent):
    """An agent represeting a single civilisation in the universe"""

    def __init__(self, unique_id, model, reasoning_level: int, age: int, 
                 visibility_factor: float, agent_growth_params) -> None:
        """
        Initialise a civilisation.

        Keyword arguments:
        unique_id: integer, uniquely identifies this civilisation
        model: a Universe object which this civilisation belongs to
        reasoning_level: 
        """
        super().__init__(unique_id, model)

        # “id” is nicer than “unique_id”
        self.id = unique_id

        # add reference to model's rng
        self.rng = model.rng

        # initialise reset time, which is the “zero time” for this civilisation.
        # It is set to the current model time if the civilisation is destroyed.
        self.reset_time = -age

        # initialise visibility factor -- the civilisation can choose to hide 
        # which decreases its apparent tech level (=technosignature)
        self.visibility_factor = visibility_factor

        # save growth parameters
        self.agent_growth_params = agent_growth_params

        # save ipomdp reasoning level
        self.level = reasoning_level

        # initialise own tech level
        self.step_tech_level()

        # initialise own state (see a description of the state in the 
        # get_state method)
        self._init_state()
        
        # keep track of previous action
        self.previous_agent_action = action.NO_ACTION

    def initialise_forest(self):
        """Create belief trees"""
        self.forest = ipomdp_solver.BeliefForest(owner=self)

    def step_tech_level(self):
        """Update own tech level"""
        new_tech_level = self.model.agent_growth(
                            self.model.schedule.time - self.reset_time, 
                            **self.agent_growth_params)

        # update tech level and calculate new influence radius
        self.tech_level = new_tech_level
        self.influence_radius = growth.influence_radius(new_tech_level)

    def step_update_beliefs(self):
        """
        Update beliefs regarding technology level and hostility of 
        neighbours.
        """
        # no need to update on the first step
        if self.model.schedule.time == 0:
            return

        if self.model.decision_making != "ipomdp":
            return

        # determine the observation received by agent
        obs = ipomdp.sample_observation(state=self.model.get_state(),
                                        action=self.model.previous_action,
                                        agent=self,
                                        model=self.model,
                                        n_samples=1)

        # update beliefs
        self.forest.update_beliefs(owner_action=self.previous_agent_action,
                                   owner_observation=obs)

        # previous action is no longer needed
        self.previous_agent_action = action.NO_ACTION

    def step_plan(self):
        """
        Plan if decisions are made using the I-POMDP-based algorithm.
        """
        if self.model.decision_making == "ipomdp":
            self.forest.plan()


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
            agent_action = self.forest.optimal_action()

        elif self.model.decision_making == "random":
            possible_actions = tuple(ipomdp_solver.possible_actions(
                model=self.model,
                agent=self))
            agent_action = self.rng.choice(possible_actions)

        else:
            raise NotImplementedError("Only 'random' and 'ipomdp' are supported")

        ### Perform and log action
        if isinstance(agent_action, Civilisation):
            self.dprint(f"Attacks {agent_action}")
            agent_action.attack(self)

        elif agent_action == action.HIDE:
            self.visibility_factor *= self.model.visibility_multiplier

            self.dprint(f"Hides (tech level {self.tech_level:.3f},",
                        f"visibility {self.visibility_factor:.3f})")
            self.model.datacollector.add_table_row('actions',
                {'time': self.model.schedule.time,
                 'actor': self.unique_id,
                 'action': action.HIDE}, 
                ignore_missing=True)
                
        elif agent_action == action.NO_ACTION:
            self.dprint("-")
            # log
            self.model.datacollector.add_table_row('actions',
                {'time': self.model.schedule.time,
                 'actor': self.unique_id,
                 'action': action.NO_ACTION}, 
                ignore_missing=True)

        else:
            raise Exception("Unrecognised action")

        self.previous_agent_action = agent_action
        self.model.previous_action = {self: agent_action}

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
            self.reset_time = self.model.schedule.time + 1
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
        if self.model.agent_growth == growth.sigmoid_growth:
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
        if self.model.agent_growth == growth.sigmoid_growth:
            self._state[0] = self.model.schedule.time - self.reset_time
            self._state[1] = self.visibility_factor
            self._state[2] = self.agent_growth_params['speed']
            self._state[3] = self.agent_growth_params['takeoff_time']
        else:
            raise NotImplementedError()

        return self._state

    def __str__(self):
        return f'Civ {self.id}'

    def __repr__(self):
        return f'Civ {self.id}'