### EXPERIMENT 2
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


def jensen_shannon_divergence(dist1, dist2):
    # pad with zeros to reach the same length
    max_length = max(len(dist1), len(dist2))
    dist1 = np.pad(
        dist1,
        pad_width=(0, max_length - len(dist1)),
        mode="constant",
        constant_values=0,
    )
    dist2 = np.pad(
        dist2,
        pad_width=(0, max_length - len(dist2)),
        mode="constant",
        constant_values=0,
    )

    # average distribution
    dist_m = 0.5 * (dist1 + dist2)

    # Kullback-Leibler divergence
    kld = lambda dist_p, dist_q: sum(
        p * np.log2(p / q) for p, q in zip(dist_p, dist_q) if p > 0
    )

    jsd = 0.5 * kld(dist1, dist_m) + 0.5 * kld(dist2, dist_m)

    return jsd


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
result_file_path = os.path.join(output_dir, f"experiment2_results_{run_id}.pickle")

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

# number of repetitions
n_repetitions = 10

# maximum number of model steps
max_steps = 1000

# choose appropriate parameters for this run
all_params = [
    (seed, n_agents) for seed in range(n_repetitions) for n_agents in (3, 5, 7)
]
seed, n_agents = all_params[run_id]

params = {
    "seed": seed,
    "n_agents": n_agents,
    "agent_growth": "sigmoid",
    "agent_growth_params": {
        "speed_range": (0.3, 0.5),
        "takeoff_time_range": (20, 40),
        "speed_noise_scale": 0.03,
        "speed_noise_dist": "normal",
        "takeoff_time_noise_scale": 3,
    },
    "rewards": {"destroyed": -1, "hide": -0.01, "attack": 0},
    "n_root_belief_samples": 1000,
    "n_tree_simulations": 100 if test_run else 10000,
    "n_reinvigoration_particles": 0,
    "obs_noise_sd": 0.15,
    "obs_self_noise_sd": 0.15,
    "reasoning_level": 1,
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

run_description = f"run {run_id} (seed={seed}, n_agents={n_agents})"

print(
    f"Starting {run_description} at {start_time} in {'test' if test_run else 'full'} mode",
    flush=True,
)

# initialise model
model = universe.Universe(**params)

# store information on success rate of lower tree queries
lower_tree_query_data = []

# keep track of attack streak length distribution
prev_attack_streak_length_dist = None

stop_reason = "crashed or reached max_steps"

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

    ### analyse and keep track of attack streak length distribution
    action_data = model.datacollector.get_table_dataframe("actions")
    streak_counts = analyse.count_attack_streaks(action_data)
    if len(streak_counts) > 0:
        # calculate attack streak length distribution
        attack_streak_length_dist = np.array(
            [
                streak_counts[length] if length in streak_counts else 0
                for length in range(1, max(streak_counts.keys()) + 1)
            ]
        )
        attack_streak_length_dist = (
            attack_streak_length_dist / attack_streak_length_dist.sum()
        )

        # check if the attack streak length distribution has converged
        if model.schedule.time % 50 == 0:
            if prev_attack_streak_length_dist is not None and model.schedule.time > 300:
                # calculate Jensen-Shannon divergence
                jsd = jensen_shannon_divergence(
                    attack_streak_length_dist, prev_attack_streak_length_dist
                )

                print(
                    f"{run_description}: At time {model.schedule.time} the distribution is {attack_streak_length_dist}. The JSD to the distribution 50 steps back ({prev_attack_streak_length_dist}) is {jsd:.4f}",
                    flush=True,
                )
                if jsd < 1e-4:
                    stop_reason = "distribution converged"
                    break

            prev_attack_streak_length_dist = attack_streak_length_dist

    # report progress every now and then
    if model.schedule.time % 10 == 0 and model.schedule.time > 0:
        print(
            f"{run_description}: now at {model.schedule.time} steps ({datetime.datetime.now()})",
            flush=True,
        )

    # if we are less than an hour from the time limit, stop execution
    if (
        time_limit is not None
        and datetime.datetime.now() - start_time
        > datetime.timedelta(hours=time_limit - 1)
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
    "seed": seed,
    "n_agents": n_agents,
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
