import sys, os

# add the parent folder of the experiments folder to the path
# this is not very neat, but I want to retain the folder structure
lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(lib_path)

import pickle
from model import universe, analyse
import numpy as np
import argparse
from datetime import datetime


def run(params, max_steps, tables):
    # initialise model
    model = universe.Universe(**params)

    # store information on success rate of lower tree queries
    lower_tree_query_data = []

    while model.running and model.schedule.time <= max_steps:
        model.step()

        # analyse log for this time step
        step_query_data = analyse.prop_successful_lower_tree_queries(model)

        # turn tree signatures into strings to avoid storing the tree objects themselves
        step_query_data = {
            str(signature).strip("()"): props
            for signature, props in step_query_data.items()
        }

        # store
        lower_tree_query_data.append(step_query_data)

        # clear log to save memory
        model.log = []

    agent_df = model.datacollector.get_agent_vars_dataframe()
    tables = tuple(model.datacollector.get_table_dataframe(table) for table in tables)

    return (model.schedule.time, agent_df, *tables, lower_tree_query_data)


# parse arguments given to script
parser = argparse.ArgumentParser()
parser.add_argument("--test", help="a small test run", action="store_true")
parser.add_argument(
    "--output", type=str, help="directory to save output file", required=True
)
parser.add_argument(
    "--run_id", type=int, help="id of run (determines parameters)", required=True
)
args = parser.parse_args()

# extract run arguments
test_run = args.test
run_id = args.run_id
output_dir = args.output

# path to result file
result_file_path = os.path.join(output_dir, f"experiment1_results_{run_id}.pickle")

# check the output directory to see if the result file exists.
# If it does, this run has already been completed and can be stopped.
if os.path.isfile(result_file_path):
    print(f"Run {run_id} has already been completed, exiting.", flush=True)
    sys.exit()

# number of repetitions
n_repetitions = 10

# maximum number of model steps
max_steps = 100

# choose appropriate parameters for this run
all_params = [
    (seed, attack_reward)
    for seed in range(n_repetitions)
    # -1.1, -1.0, ..., 0, 0.1 (13 values)
    for attack_reward in np.linspace(-1.1, 0.1, 13).round(1)
]
seed, attack_reward = all_params[run_id]

params = {
    "seed": seed,
    "n_agents": 2,
    "agent_growth": "sigmoid",
    "agent_growth_params": {
        "speed_range": (0.3, 0.5),
        "takeoff_time_range": (20, 40),
        "speed_noise_scale": 0.03,
        "speed_noise_dist": "normal",
        "takeoff_time_noise_scale": 3,
    },
    "rewards": {"destroyed": -1, "hide": -0.01, "attack": attack_reward},
    "n_root_belief_samples": 1000,
    "n_tree_simulations": 100 if test_run else 10000,
    "n_reinvigoration_particles": 0,
    "obs_noise_sd": 0.15,
    "obs_self_noise_sd": 0.15,
    "reasoning_level": 2,
    "action_dist_0": "random",  # can be "random", "passive" or "aggressive"
    # a discount horizon of 4
    "discount_factor": 0.6,
    "discount_epsilon": 0.10,
    "exploration_coef": 0.1,
    "softargmax_coef": 0.01,
    "visibility_multiplier": 0.5,
    "decision_making": "ipomdp",
    "init_age_belief_range": (0, 50),
    "init_age_range": (0, 50),
    "init_visibility_belief_range": (1, 1),
    "init_visibility_range": (1, 1),
    "ignore_exceptions": True,
}

start_time = datetime.now()

print(
    f"Starting run {run_id} at {start_time} in {'test' if test_run else 'full'} mode",
    flush=True,
)

# run
data = run(
    params=params,
    max_steps=max_steps,
    tables=["actions", "rewards", "action_qualities"],
)

end_time = datetime.now()
duration = end_time - start_time

# add run information to data
data = (run_id, duration, seed, attack_reward, *data)

# save
with open(result_file_path, "wb") as f:
    pickle.dump(data, f)

print(f"Finished run {run_id} at {end_time}, took {duration}", flush=True)
