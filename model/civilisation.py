import mesa
import numpy as np
from model import growth, ipomdp, ipomdp_solver, action


class Civilisation(mesa.Agent):
    """An agent represeting a single civilisation in the universe"""

    def __init__(
        self,
        unique_id,
        model,
        reasoning_level: int,
        age: int,
        visibility_factor: float,
        agent_growth_params,
    ) -> None:
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

        # initialise age and previous age
        self.age = age
        self.previous_age = age - 1

        # initialise visibility factor -- the civilisation can choose to hide
        # which decreases its apparent tech level (=technosignature)
        self.visibility_factor = visibility_factor

        # save growth parameters
        self.agent_growth_params = agent_growth_params

        # save ipomdp reasoning level
        self.level = reasoning_level

        # agent state will be initialised elsewhere
        self._state = None

        # store the set of possible actions for this agent (will be determined once
        # it is requested)
        self._action_set = None

    def initialise_forest(self):
        """Create belief trees"""
        self.forest = ipomdp_solver.BeliefForest(owner=self)

    def possible_actions(self, state=None):
        """
        Return all possible actions of agent. State can be supplied to
        constrain the actions to only those that the agent is currently
        capable of.
        """
        if self._action_set is None:
            self._action_set = tuple(ag for ag in self.model.agents if ag != self) + (
                action.HIDE,
                action.NO_ACTION,
            )

        if state is None:
            return self._action_set

        # determine possible actions depending on state
        agent_tech_level = growth.tech_level(state=state[self.id], model=self.model)

        return self.model.get_agent_neighbours(
            agent=self, tech_level=agent_tech_level
        ) + (action.HIDE, action.NO_ACTION)

    @property
    def tech_level(self) -> float:
        return growth.tech_level(state=self.state, model=self.model)

    @property
    def influence_radius(self) -> float:
        return growth.influence_radius(self.tech_level)

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
        obs = ipomdp.sample_observation(
            state=self.model.get_state(),
            action=self.model.previous_action,
            agent=self,
            model=self.model,
        )

        # update beliefs
        self.forest.update_beliefs(
            owner_action=self.model.previous_action[self.id], owner_observation=obs
        )

    def step_plan(self):
        """
        Plan if decisions are made using the I-POMDP-based algorithm.
        """
        if self.model.decision_making != "ipomdp":
            return

        self.forest.plan()

    def choose_action(self):
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
            agent_action = self.random.choice(self.possible_actions())

        else:
            raise NotImplementedError("Only 'random' and 'ipomdp' are supported")

        return agent_action

    def step_log_estimated_action_qualities(self):
        """
        Stores:
        - the estimated action qualities of agent's own actions
        - the estimated action qualities of others' actions
        - number of times simulated for each
        """
        if self.model.decision_making != "ipomdp":
            return

        # determine own estimated action qualities
        root_node = self.forest.top_level_tree_root_node
        action_qualities = ipomdp_solver.calculate_action_qualities(
            belief=root_node.belief,
            n_expansions=root_node.n_expansions,
            n_expansions_act=root_node.n_expansions_act,
            act_value=root_node.act_value,
            explore=False,
            exploration_coef=0,
        )

        n_expansions = root_node.n_expansions_act.sum(axis=0)

        # store
        self.model.datacollector.add_table_row(
            table_name="action_qualities",
            row={
                "time": self.model.schedule.time,
                "estimator": self.id,
                "actor": self.id,
                "qualities": action_qualities,
                "n_expansions": n_expansions,
            },
        )

        if self.forest.top_level_tree.level == 0:
            return

        # determine estimated action qualities for other agents
        for other_agent in self.model.agents:
            if other_agent == self:
                continue

            action_qualities = np.zeros(len(other_agent.possible_actions()))

            for particle in root_node.particles:
                if particle.weight == 0:
                    continue

                # initialise weights in the level L-1 tree
                self.forest.initialise_simulation(particle)

                # determine the appropriate node in the lower tree
                lower_node = self.forest.get_matching_lower_node(
                    particle=particle, agent=other_agent
                )

                # determine action qualities for other_agent
                particle_action_qualities = ipomdp_solver.calculate_action_qualities(
                    belief=lower_node.belief,
                    n_expansions=lower_node.n_expansions,
                    n_expansions_act=lower_node.n_expansions_act,
                    act_value=lower_node.act_value,
                    explore=False,
                    exploration_coef=0,
                )

                # if there are unexpanded actions (np.infty), ignore them but include
                # the other estimates
                particle_action_qualities[particle_action_qualities == np.infty] = 0

                # add to weighted sum
                action_qualities += particle.weight * particle_action_qualities

            # store
            self.model.datacollector.add_table_row(
                table_name="action_qualities",
                row={
                    "time": self.model.schedule.time,
                    "estimator": self.id,
                    "actor": other_agent.id,
                    "qualities": action_qualities,
                    "n_expansions": None,
                },
            )

    @property
    def state(self):
        """
        Updates self._state and returns it.

        The state consists of 5 numbers:
        1. previous age (age = time since last destruction)
        1. current age
        2. visibility factor
        3. growth speed
        4. growth takeoff time

        The last two are related to the specific growth model assumed, and
        will therefore be different with different growth types
        """
        if self._state is None:
            self._state = np.zeros(self.model.agent_state_size)

        if self.model.agent_growth == growth.sigmoid_growth:
            self._state[0] = self.previous_age
            self._state[1] = self.age
            self._state[2] = self.visibility_factor
            self._state[3] = self.agent_growth_params["speed"]
            self._state[4] = self.agent_growth_params["takeoff_time"]
        else:
            raise NotImplementedError()

        return self._state

    def __str__(self):
        return f"Civ {self.id}"

    def __repr__(self):
        return f"Civ {self.id}"
