import mesa
import numpy as np

from model import civilisation, growth, ipomdp, action
from typing import Tuple, List, Any


class Universe(mesa.Model):
    def __init__(
        self,
        n_agents,
        agent_growth,
        agent_growth_params,
        rewards,
        prob_indifferent,
        n_root_belief_samples,
        n_tree_simulations,
        obs_noise_sd,
        obs_self_noise_sd,
        reasoning_level,
        action_dist_0,
        initial_belief,
        initial_belief_params,
        discount_factor,
        discount_epsilon,
        exploration_coef,
        softargmax_coef,
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
        ignore_exceptions=False,
    ) -> None:
        """
        Initialise a new Universe model.

        Keyword arguments:
        n_agents: the number of agents in the model
        agent_growth: the type of growth agents can undergo. Currently supports \
                      "sigmoid".
        agent_growth_params: see agent_growth
        rewards: a dictionary of rewards with keys 'destroyed', 'hide' and \
                 'attack' and the rewards as values
        prob_indifferent: probability that civilisations believe each other agent has \
                          an attack reward of 0. Called “indifference” because an agent \
                          with an attack reward of 0 does not include the well-being \
                          of others in its moral considerations.
        n_root_belief_samples: the number of samples (particles) used in \
                               representing beliefs at root nodes of trees
        n_tree_simulations: the number of simulations to perform on each tree \
                            when planning
        obs_noise_sd: standard deviation of technosignature observation noise \
                      (which follows an unbiased normal distribution)
        obs_self_noise_sd: standard deviation of own technosignature observation noise \
                           (which follows an unbiased normal distribution)
        reasoning_level: the level of ipomdp reasoning all civilisations use \
        action_dist_0: the default action distribution at level 0. Level 0 \
                       trees use this to determine the actions of others when \
                       planning. "random" means the others' actions are \
                       chosen uniformly over the set of possible choices.
        initial_belief: defines the function used to generate initial beliefs. \
                        Currently supports "uniform".
        initial_belief_params: a dictionary of possible parameters for the initial \
                               belief generator
        discount_factor: how much future time steps are discounted when \
                         determining the rational actions of agents
        discount_epsilon: how small the value discount_factor ** time has to \
                          be to stop looking forward when planning. Determines\
                          the planning time horizon.
        exploration_coef: used in the MCTS (Monte Carlo Tree Search) based \
                          algorithm to adjust how much exploration of seldomly\
                          visited agent actions is emphasised
        softargmax_coef: the coefficient used in softargmax. A lower value means \
                         higher-quality actions are weighted more strongly, whereas \
                         a higher value means actions are chosen more uniformly and \
                         less based on the qualities of actions. Analogous to the \
                         Boltzmann constant k in the Boltzmann distribution.
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
        ignore_exceptions: whether to ignore exceptions (and instead just stop
                           running)
        """
        # save parameters
        self.n_agents = n_agents
        self.agent_growth_params = agent_growth_params
        self.rewards = rewards
        self.prob_indifferent = prob_indifferent
        self.n_root_belief_samples = n_root_belief_samples
        self.n_tree_simulations = n_tree_simulations
        self.obs_noise_sd = obs_noise_sd
        self.obs_self_noise_sd = obs_self_noise_sd
        self.reasoning_level = reasoning_level
        self.action_dist_0 = action_dist_0
        self.initial_belief = initial_belief
        self.initial_belief_params = initial_belief_params
        self.discount_factor = discount_factor
        self.discount_epsilon = discount_epsilon
        self.exploration_coef = exploration_coef
        self.softargmax_coef = softargmax_coef
        self.visibility_multiplier = visibility_multiplier
        self.decision_making = decision_making
        self.init_age_belief_range = init_age_belief_range
        self.init_age_range = init_age_range
        self.init_visibility_belief_range = init_visibility_belief_range
        self.init_visibility_range = init_visibility_range
        self.debug = debug
        self.log_events = log_events
        self.ignore_exceptions = ignore_exceptions

        if agent_growth == "sigmoid":
            self.agent_growth = growth.sigmoid_growth
            self.agent_state_size = 4
        else:
            raise NotImplementedError

        # initialise random number generator
        self.rng = np.random.default_rng(seed)

        # initialise schedule and space
        self.schedule = JointActivation(
            self,
            update_methods=["step_update_beliefs", "step_plan"],
            step_method="step_act",
            log_methods=["step_log_reward", "step_log_estimated_action_qualities"],
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
                agent_growth == "sigmoid"
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
            x, y = self.rng.random(size=2)
            self.space.place_agent(agent, (x, y))

        # initialise distance cache
        self._init_distance_cache()

        # initialise state
        self._state = None

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
                ],
                "rewards": ["time", "agent", "reward"],
                "action_qualities": [
                    "time",
                    "estimator",
                    "actor",
                    "qualities",
                    "n_expansions",
                ],
            },
        )

        # keep track of the last action
        self.previous_action = None

        # keep track of the previous model state
        self.previous_state = None

        # initialise a log for events
        self.log = []

        # run indefinitely (this is used by mesa.batch_run)
        self.running = True

    def step(self):
        """Advance the model by one step."""
        if not self.running:
            return

        try:
            self.datacollector.collect(self)
            self.schedule.step()
        except Exception as e:
            if self.ignore_exceptions:
                print("Stopping due to:", e)
                self.running = False
                return

            raise e

    def step_act(self):
        """
        Activated once every time step. Asks each agent for an action and progresses
        the model state.
        """
        # store current model state (needed for calculating reward later)
        self.previous_state = self.get_state()
        previous_tech_levels = growth.tech_level(state=self.previous_state, model=self)

        # determine action
        action_ = tuple(agent.choose_action() for agent in self.agents)

        # determine result of action
        new_state = ipomdp.transition(self.get_state(), action_=action_, model=self)

        # destroyed agents
        result_description = ["-" for agent in self.agents]

        # store the values in the new state
        for agent, agent_action, new_agent_state in zip(
            self.agents, action_, new_state
        ):
            ### Update agent's state

            # new agent attributes
            new_agent_age = new_agent_state[0]
            new_agent_visibility_factor = new_agent_state[1]
            new_agent_growth_speed = new_agent_state[2]
            new_agent_takeoff_time = new_agent_state[3]

            # growth speed and takeoff time should not change
            assert new_agent_growth_speed == agent.get_state()[2]
            assert new_agent_takeoff_time == agent.get_state()[3]

            # agent got destroyed
            if new_agent_age == 0:
                agent.reset_time = self.schedule.time + 1

                # when an agent is destroyed, its visibility factor should be reset
                assert new_agent_visibility_factor == 1

                # update description
                result_description[agent.id] = "d"

            agent.visibility_factor = new_agent_visibility_factor

            ### Log agent's own action
            if agent_action in (action.NO_ACTION, action.HIDE):
                self.datacollector.add_table_row(
                    "actions",
                    {
                        "time": self.schedule.time,
                        "actor": agent.id,
                        "action": agent_action,
                    },
                    ignore_missing=True,
                )

            else:
                # agent attacked
                target = agent_action

                # determine result. Attack only takes place if agent can reach target
                result = None
                if self.is_neighbour(
                    agent1=agent,
                    agent2=target,
                    tech_level=previous_tech_levels[agent.id],
                ):
                    result = new_state[target.id, 0] == 0

                self.datacollector.add_table_row(
                    "actions",
                    {
                        "time": self.schedule.time,
                        "actor": agent.id,
                        "action": "a",
                        "attack_target": target.id,
                        "attack_successful": result,
                    },
                )

        if self.debug >= 1:
            print(
                f"t = {self.schedule.time}: actions {action_}, result {tuple(result_description)}"
            )

        # store previous action (will be used to generate observations on next time step)
        self.previous_action = action_

    def add_log_event(self, event_type: int, event_data: Any) -> None:
        """
        Codes 1x: Determining others' action during a simulation
        Codes 2x: Lower node belief update
        Codes 3x: Updating beliefs during a simulation
        """
        # printing settings
        debug_print_warnings = self.debug >= 1
        debug_print_info = self.debug >= 2

        # add event to log
        if self.log_events:
            self.log.append(
                LogEvent(
                    event_type=event_type,
                    event_data=tuple(str(d).strip("()") for d in event_data),
                    event_time=self.schedule.time,
                )
            )

        # code 10 means prediction of others' action when simulating a tree was successful

        if event_type == 11 and debug_print_info:
            print(
                f"Could not find a matching node in child tree when simulating {event_data}"
            )
        elif event_type == 12 and debug_print_info:
            print(f"Belief in lower tree has diverged when simulating {event_data}")
        elif event_type == 13 and debug_print_info:
            print(
                f"All actions in lower node have not been expanded when simulating {event_data}"
            )
        elif event_type == 21 and debug_print_info:
            print(
                f"Lower node belief update {event_data[0]} : {event_data[1]}",
                f"-> {event_data[2]} : {event_data[3]} saw beliefs diverge",
            )
        elif event_type == 22 and debug_print_info:
            print(
                f"Lower node belief update {event_data[0]} : {event_data[1]}",
                f"-> {event_data[2]} : {event_data[3]} could not find the node in",
                "the lower tree. An empty node was created.",
            )
        elif event_type == 22 and debug_print_info:
            print(
                f"Lower node belief update {event_data[0]} : {event_data[1]}",
                f"-> {event_data[2]} : {event_data[3]} found an empty node in",
                "the lower tree.",
            )
        elif event_type == 31 and debug_print_info:
            print(
                "Could not create a belief in the next lower node of agent",
                f"{event_data[1]} when simulating tree {event_data[0]} because",
                "its parent does not exist.",
            )
        elif event_type == 32 and debug_print_info:
            print(
                "Could not create a belief in the next lower node of agent",
                f"{event_data[1]} when simulating tree {event_data[0]} because",
                "it does not exist.",
            )

    def get_state(self):
        """
        Update and return the current model state.

        Returns:
        a NumPy array of shape (n, k), where k is the length of the state
        description of a single agent
        """
        if self._state == None:
            self._state = np.zeros((self.n_agents, self.agent_state_size))

        for i, agent in enumerate(self.agents):
            self._state[i] = agent.get_state()

        return self._state

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


class JointActivation(mesa.time.BaseScheduler):
    """
    A scheduler which first calls the update method(s) for every agent (if
    there are multiple, this is done in stages: the first update method
    is executed for every agent before moving on to the next). Then, the step method
    of the Universe is called, which gets an action from every agent and progresses
    the model state. Finally, the log methods of each agent are called (staged, like
    before).
    """

    def __init__(
        self,
        model: mesa.Model,
        update_methods: List[str],
        step_method: str,
        log_methods: List[str],
    ) -> None:
        """
        Create an empty Single Activation schedule.

        Args:
            model: Model object associated with the schedule.
            update_methods: List of strings of names of stages to run, in the
                            order to run them in.
            step_method: The name of the step method to be activated in the model
                         object.
            log_methods: List of strings of names of stages to run after step_method,
                         in the order to run them in.
        """
        super().__init__(model)
        self.update_methods = update_methods
        self.step_method = step_method
        self.log_methods = log_methods

    def step(self) -> None:
        """
        Executes the update method(s) of all agents (if multiple, in stages)
        and then the step method of a randomly chosen agent. Finally, executes
        lof method(s) of all agents (if multiple, in stages)
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

        # call the model's step function
        getattr(self.model, self.step_method)()

        # run log methods in stages for all agents
        for log_method in self.log_methods:
            for agent_key in agent_keys:
                if agent_key in self._agents:
                    # run update method
                    getattr(self._agents[agent_key], log_method)()

            # We recompute the keys because some agents might have been removed
            # in the previous loop.
            agent_keys = list(self._agents.keys())

        # increase time
        self.time += 1
        self.steps += 1


class LogEvent(object):
    def __init__(self, event_type: int, event_data: Any, event_time: int):
        self.event_type = event_type
        self.event_data = event_data
        self.event_time = event_time

    def __repr__(self):
        return f"Event(t={self.event_time}: {self.event_type}, {self.event_data})"
