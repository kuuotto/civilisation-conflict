### EXPERIMENT 4
import os
import sys

# add the parent folder of the experiments folder to the path
# this is not very neat, but I want to retain the folder structure
lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(lib_path)

import argparse
import datetime
import pickle

from SALib.sample import sobol

from model import analyse, universe

start_time = datetime.datetime.now()

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
result_file_path = os.path.join(output_dir, f"experiment4_results_{run_id}.pickle")

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

# maximum number of model steps
max_steps = 100

# create a Saltelli sample
problem = {
    "num_vars": 1,
    "names": ["attack_reward"],
    "bounds": [[-0.2, 0.1]],
}
samples = sobol.sample(
    problem=problem,
    N=32,
    calc_second_order=False,
    seed=42,
)

exploration_coef = 0.5
softargmax_coef = 0.1
prob_indifferent = 0.5

# there are a total of 1 * 32 * (1 + 2) = 96 possible samples.

# choose appropriate parameters for this run
attack_reward = samples[run_id][0]

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
    "n_tree_simulations": 100 if test_run else 9000,
    "obs_noise_sd": 0.15,
    "obs_self_noise_sd": 0.15,
    "reasoning_level": 1,
    "action_dist_0": "random",  # can be "random", "passive" or "aggressive"
    "initial_belief": "uniform",  # can be "uniform" or "surpass_scenario"
    "initial_belief_params": {},
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

run_description = (
    f"run {run_id} (reward={attack_reward}, prob_indifferent={prob_indifferent})"
)

print(
    f"Starting {run_description} at {start_time} in {'test' if test_run else 'full'} mode",
    flush=True,
)

# initialise model
model = universe.Universe(**params)

# store information on success rate of lower tree queries
lower_tree_query_data = []

stop_reason = "reached max_steps or crashed"

while model.running and model.schedule.time <= max_steps:
    model.step()

    ### analyse log for this time step
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

    # if we are less than an hour from the time limit, stop execution
    if (
        time_limit is not None
        and datetime.datetime.now() - start_time
        > datetime.timedelta(hours=time_limit - 1, minutes=30)
    ):
        stop_reason = "time limit"
        break

    # stop if we are instructed to do so
    if os.path.isfile(os.path.join(output_dir, "stop.txt")):
        stop_reason = "requested"
        break

agent_df = model.datacollector.get_agent_vars_dataframe()
tables = {
    table: model.datacollector.get_table_dataframe(table)
    for table in ("actions", "rewards", "action_qualities")
}

data = {
    "run_length": model.schedule.time,
    "agent_df": agent_df,
    "lower_tree_query_data": lower_tree_query_data,
    **tables,
}

end_time = datetime.datetime.now()
duration = end_time - start_time

# add run information to data
data = {
    "run_id": run_id,
    "duration": duration,
    "attack_reward": attack_reward,
    "prob_indifferent": prob_indifferent,
    "params": params,
    **data,
}

# save
with open(result_file_path, "wb") as f:
    pickle.dump(data, f)

print(
    f"Finished {run_description} at {end_time} (reason: {stop_reason}), took {duration}",
    flush=True,
)
