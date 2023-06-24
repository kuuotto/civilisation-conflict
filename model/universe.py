import mesa
import numpy as np

from model import civilisation, growth, ipomdp
from typing import Tuple, List, Any


class Universe(mesa.Model):
    def __init__(
        self,
        n_agents,
        agent_growth,
        agent_growth_params,
        rewards,
        n_root_belief_samples,
        n_tree_simulations,
        n_reinvigoration_particles,
        obs_noise_sd,
        reasoning_level,
        action_dist_0,
        discount_factor,
        discount_epsilon,
        exploration_coef,
        visibility_multiplier,
        decision_making,
        init_age_belief_range,
        init_age_range,
        init_visibility_belief_range,
        init_visibility_range,
        toroidal_space=False,
        debug=False,
        log_events=True,
        seed=0,
    ) -> None:
        """
        Initialise a new Universe model.

        Keyword arguments:
        n_agents: the number of agents in the model
        agent_growth: the type of growth agents can undergo. A callable which \
                      accepts a time step and additional keyword arguments \
                      defined in agent_growth_params.
        agent_growth_params: see agent_growth
        rewards: a dictionary of rewards with keys 'destroyed', 'hide' and \
                 'attack' and the rewards as values
        n_root_belief_samples: the number of samples (particles) used in \
                               representing beliefs at root nodes of trees
        n_tree_simulations: the number of simulations to perform on each tree \
                            when planning
        n_reinvigoration_particles: number of additional particles to create \
                                    for each tree when updating beliefs
        obs_noise_sd: standard deviation of technosignature observation noise \
                      (which follows an unbiased normal distribution)
        reasoning_level: the level of ipomdp reasoning all civilisations use \
        action_dist_0: the default action distribution at level 0. Level 0 \
                       trees use this to determine the actions of others when \
                       planning. "random" means the others' actions are \
                       chosen uniformly over the set of possible choices.
        discount_factor: how much future time steps are discounted when \
                         determining the rational actions of agents
        discount_epsilon: how small the value discount_factor ** time has to \
                          be to stop looking forward when planning. Determines\
                          the planning time horizon.
        exploration_coef: used in the MCTS (Monte Carlo Tree Search) based \
                          algorithm to adjust how much exploration of seldomly\
                          visited agent actions is emphasised
        visibility_multiplier: how much a single “hide” action multiplies the \
                               current agent visibility factor by
        decision_making: the method used by agents to make decisions. Options \
                         include "random" and "ipomdp".
        init_age_belief_range: the range in which agents initially believe the\
                               ages of others are uniformly distributed
        init_age_range: the range in which the ages of agents are initially\
                        uniformly distributed. Typically (0, 0)
        init_visibility_belief_range: the range in which agents initially \
                                      believe the visibility factors of others\
                                      are uniformly distributed
        init_visibility_range: the range in which the visibility factors of \
                               agents are initialy uniformly distributed. \
                               Typically (1, 1)
        toroidal_space: whether to use a toroidal universe topology
        debug: whether to print detailed debug information while model is run
        log_events: whether to keep a log of model events
        seed: seed of the random number generator. Fixing the seed allows \
              for reproducibility of results.
        """
        # save parameters
        self.n_agents = n_agents
        self.agent_growth = agent_growth
        self.agent_growth_params = agent_growth_params
        self.rewards = rewards
        self.n_root_belief_samples = n_root_belief_samples
        self.n_tree_simulations = n_tree_simulations
        self.n_reinvigoration_particles = n_reinvigoration_particles
        self.obs_noise_sd = obs_noise_sd
        self.reasoning_level = reasoning_level
        self.action_dist_0 = action_dist_0
        self.discount_factor = discount_factor
        self.discount_epsilon = discount_epsilon
        self.exploration_coef = exploration_coef
        self.visibility_multiplier = visibility_multiplier
        self.decision_making = decision_making
        self.init_age_belief_range = init_age_belief_range
        self.init_age_range = init_age_range
        self.init_visibility_belief_range = init_visibility_belief_range
        self.init_visibility_range = init_visibility_range
        self.debug = debug
        self.log_events = log_events

        # initialise random number generator
        self.rng = np.random.default_rng(seed)

        # initialise schedule and space
        self.schedule = SingleActivation(
            self,
            update_methods=["step_tech_level", "step_update_beliefs", "step_plan"],
            step_method="step_act",
        )
        self.space = mesa.space.ContinuousSpace(x_max=1, y_max=1, torus=toroidal_space)

        # keep a list of agents in the model. The schedule also keeps a list,
        # but it is re-generated every time it is accessed which is not very
        # efficient
        self.agents: List[civilisation.Civilisation] = []

        # add agents
        for id in range(n_agents):
            # choose the age of the civilisation
            age = self.rng.integers(*init_age_range, endpoint=True)

            # choose visibility factor of the civilisation
            visibility_factor = self.rng.uniform(*init_visibility_range)

            # choose the growth parameters of the civilisation
            if (
                agent_growth == growth.sigmoid_growth
                and "speed_range" in agent_growth_params
                and "takeoff_time_range" in agent_growth_params
            ):
                speed_range = agent_growth_params["speed_range"]
                takeoff_time_range = agent_growth_params["takeoff_time_range"]

                growth_params = {
                    "speed": self.rng.uniform(*speed_range),
                    "takeoff_time": self.rng.integers(
                        *takeoff_time_range, endpoint=True
                    ),
                }
            else:
                growth_params = agent_growth_params

            agent = civilisation.Civilisation(
                unique_id=id,
                model=self,
                reasoning_level=reasoning_level,
                age=age,
                visibility_factor=visibility_factor,
                agent_growth_params=growth_params,
            )
            self.schedule.add(agent)
            self.agents.append(agent)

            # place agent in a randomly chosen position
            # TODO: consider the distribution of stars
            x, y = self.rng.random(size=2)
            self.space.place_agent(agent, (x, y))

        # initialise distance cache
        self._init_distance_cache()

        # initialise model state
        self._init_state()

        # initialise constants used
        self._init_constants()

        # after all agents have been created, initialise their trees
        for agent in self.agents:
            agent.initialise_forest()

        # initialise data collection
        self.datacollector = mesa.DataCollector(
            agent_reporters={
                "Technology": "tech_level",
                "Radius of Influence": "influence_radius",
                "Visibility Factor": "visibility_factor",
                "Position": "pos",
            },
            tables={
                "actions": [
                    "time",
                    "actor",
                    "action",
                    "attack_target",
                    "attack_successful",
                ]
            },
        )

        # keep track of the last action
        self.previous_action = None

        # initialise a log for events
        self.log = []

    def step(self):
        """Advance the model by one step."""
        self.datacollector.collect(self)
        self.schedule.step()

    def add_log_event(self, event_type: int, event_data: Any) -> None:
        if self.log_events:
            self.log.append(LogEvent(event_type=event_type, event_data=event_data))

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
        for i, agent in enumerate(self.agents):
            self._state[i] = agent.get_state()

        return self._state

    def _init_constants(self) -> None:
        """
        Pre-calculates some constants used in the I-POMDP solver.
        """
        self.obs_prob_const = ipomdp._norm_pdf(x=0, mean=0, sd=self.obs_noise_sd)

    def _init_distance_cache(self) -> None:
        """
        Calculates distances between all agents and stores these. This is used
        to speed up finding neighbours of agents.

        These distances are valid as long as agents don't move.

        Toroidal space is taken into account.
        """
        self._distances = np.zeros((self.n_agents, self.n_agents))

        for i, ag_i in enumerate(self.agents):
            for j, ag_j in enumerate(self.agents):
                if ag_j == ag_i:
                    continue

                distance = self.space.get_distance(ag_i.pos, ag_j.pos)
                self._distances[i, j] = distance
                self._distances[j, i] = distance

        # stores distances as equivalent technology levels -- using these means that
        # influence radii do not have to be calculated during the simulations
        self._distances_tech_level = growth.inv_influence_radius(self._distances)

    def get_agent_neighbours(
        self,
        agent: civilisation.Civilisation,
        radius: float = None,
        tech_level: float = None,
    ) -> Tuple[civilisation.Civilisation]:
        """
        Find neighbours of agent given a radius or a technology level.

        If both a radius and a technology level are supplied, the former takes precedence.

        This is more efficient than the method of the mesa space module,
        because this uses the pre-generated array of agent distances. We can
        do this because agents do not move in our model.
        """
        if radius is not None:
            return tuple(
                ag
                for ag in self.agents
                if self._distances[agent.id, ag.id] < radius and ag != agent
            )
        elif tech_level is not None:
            return tuple(
                ag
                for ag in self.agents
                if self._distances_tech_level[agent.id, ag.id] < tech_level
                and ag != agent
            )

        raise Exception("Either a radius or a technology level must be supplied.")

    def is_neighbour(
        self,
        agent1: civilisation.Civilisation,
        agent2: civilisation.Civilisation,
        radius: float = None,
        tech_level: float = None,
    ) -> bool:
        """
        Checks if distance between the agents is less than radius.
        """
        if radius is not None:
            return self._distances[agent1.id, agent2.id] < radius
        if tech_level is not None:
            return self._distances_tech_level[agent1.id, agent2.id] < tech_level

        raise Exception("Either a radius or a technology level must be supplied.")


class SingleActivation(mesa.time.BaseScheduler):
    """
    A scheduler which first calls the update method(s) for every agent (if
    there are multiple, this is done in stages: the first update method
    is executed for every agent before moving on to the next). Finally,
    a random agent is activated to perform a step method.
    """

    def __init__(
        self, model: mesa.Model, update_methods: list[str], step_method: str
    ) -> None:
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


class LogEvent(object):
    def __init__(self, event_type: int, event_data: Any):
        self.event_type = event_type
        self.event_data = event_data

    def __repr__(self):
        return f"({self.event_type}, {self.event_data})"
