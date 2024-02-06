### EXPERIMENT 7
import sys, os

# add the parent folder of the experiments folder to the path
# this is not very neat, but I want to retain the folder structure
lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(lib_path)

import pickle
from model import universe, analyse
import numpy as np
import argparse
import datetime
from SALib.sample import sobol

# parse arguments given to script
parser = argparse.ArgumentParser()
parser.add_argument("--test", help="a small test run", action="store_true")
parser.add_argument(
    "--output", type=str, help="directory to save output file", required=True
)
parser.add_argument(
    "--run_id", type=int, help="id of run (determines parameters)", required=True
)
parser.add_argument("--time_limit", type=int, help="time limit in hours")
args = parser.parse_args()

# extract run arguments
test_run = args.test
run_id = args.run_id
output_dir = args.output
time_limit = args.time_limit

# path to result file
result_file_path = os.path.join(output_dir, f"experiment7_results_{run_id}.pickle")

# check if the output directory exists
if not os.path.isdir(output_dir):
    print(
        f"Cancelling run {run_id}; output directory {output_dir} doesn't exist",
        flush=True,
    )
    sys.exit()

# check the output directory to see if the result file exists.
# If it does, this run has already been completed and can be stopped.
if os.path.isfile(result_file_path):
    print(f"Run {run_id} has already been completed, exciting.", flush=True)
    sys.exit()


# create a Saltelli sample
problem = {
    "num_vars": 2,
    "names": ["prob_surpass_0", "prob_surpass_1"],
    "bounds": [[0, 1], [0, 1]],
}
samples = sobol.sample(
    problem=problem,
    N=128,
    calc_second_order=False,
    seed=42,
)

# there are a total of 2 * 128 * (2 + 2) = 1024 runs
# these will be run individually, which is not the most efficient as numba functions
# are compiled for each one separately. However, the experiment is quite short either way.


times_until_surpass = (2, 4)
all_params = [
    (*params, time_until_surpass)
    for params in samples
    for time_until_surpass in times_until_surpass
]

# choose appropriate parameters for this run
prob_surpass_0, prob_surpass_1, time_until_surpass = all_params[run_id]

# the important parameter
prob_indifferent = 0.5

# other parameters
exploration_coef = 0.6
softargmax_coef = 0.1
attack_reward = -0.1
action_dist_0 = "random"
reasoning_level = 1


run_description = (
    f"run {run_id} (prob_surpass_0={prob_surpass_0}, prob_surpass_1={prob_surpass_1})"
)

print(
    f"Starting {run_description} at {datetime.datetime.now()} in {'test' if test_run else 'full'} mode",
    flush=True,
)

start_time = datetime.datetime.now()

params = {
    "seed": run_id,
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
    "prob_indifferent": prob_indifferent,
    "n_root_belief_samples": 1000,
    "n_tree_simulations": 1000 if test_run else 10000,
    "obs_noise_sd": 0.15,
    "obs_self_noise_sd": 0.15,
    "reasoning_level": reasoning_level,
    "action_dist_0": action_dist_0,  # can be "random", "passive" or "aggressive"
    "initial_belief": "surpass_scenario",  # can be "uniform" or "surpass_scenario"
    "initial_belief_params": {
        "time_until_surpass": time_until_surpass,
        "prob_surpass_0": prob_surpass_0,
        "prob_surpass_1": prob_surpass_1,
    },
    # a discount horizon of 4
    "discount_factor": 0.6,
    "discount_epsilon": 0.10,
    "exploration_coef": exploration_coef,
    "softargmax_coef": softargmax_coef,
    "visibility_multiplier": 0.5,
    "decision_making": "ipomdp",
    "init_age_belief_range": (0, 50),
    "init_age_range": (0, 50),
    "init_visibility_belief_range": (1, 1),
    "init_visibility_range": (1, 1),
    "ignore_exceptions": True,
}

# initialise model
model = universe.Universe(**params)

stop_reason = "reached max_steps or crashed"

# run one planning step
model.step()

### analyse log for this time step
step_query_data = analyse.prop_successful_lower_tree_queries(model)

# turn tree signatures into strings to avoid storing the tree objects themselves
step_query_data = {
    str(signature).strip("()"): props for signature, props in step_query_data.items()
}

agent_df = model.datacollector.get_agent_vars_dataframe()
tables = {
    table: model.datacollector.get_table_dataframe(table)
    for table in ("actions", "rewards", "action_qualities")
}

end_time = datetime.datetime.now()
duration = end_time - start_time

data = {
    "run_id": run_id,
    "duration": duration,
    "run_length": model.schedule.time,
    "params": params,
    "agent_df": agent_df,
    "lower_tree_query_data": step_query_data,
    **tables,
}

# save
with open(result_file_path, "wb") as f:
    pickle.dump(data, f)

print(
    f"Finished {run_description} at {end_time} (reason: {stop_reason})",
    flush=True,
)
