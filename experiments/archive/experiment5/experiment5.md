# Experiment 5

The goal of this experiment is to determine the optimal policy for a civilisation in a specific scenario: when a weak civilisation is about to surpass a stronger civilisation in technological capability. 

We are interested in whether the stronger civilisation will engage in pre-emptive hostility, creating a Hobbesian trap. We also investigate whether it is optimal for the weaker civilisation to spend effort hiding its technosignature.

## Hypothesis

We expect to demonstrate a Hobbesian trap.

## Independent Variable(s)

At reasoning level 0 (=1):
- $p_\text{surpass}^1$ in the range [0, 1]

At reasoning level 1 (=2):
- $p_\text{surpass}^2$ in the range [0, 1]
- $p_\text{surpass}^1$ in the range [0, 1]

## Dependent Variable(s)

Utilities of the different action options in the first time step

## Other Parameters

- time until surpass: 2 or 4
- two (2) civilisations
- attack reward: 0 or -0.1
- reasoning level: 0 or 1 (=1 or 2 in the new notation)
- level 0 policy: random or passive

## Solver Parameters

- exploration coefficient: 0.3 or 0.5
- softargmax coefficient: 0.01
- 10k simulations per forest

## Simulation Details

The values of the independent variables are from a Saltelli sample with 128 samples.

In the script, reasoning level 0 (=1) is called experiment 5a and reasoning level 1 (=2) is called experiment 5b.

In total, there are $2 * 2 * 2 * 2 * 128 * (1 + 2) = 6144$ (5a) + $2 * 2 * 2 * 2 * 128 * (2 + 2) = 8192$ (5b) model evaluations. One evaluation consists of only one time step.

## Results

See **`experiment5_analysis.ipynb`**.

This experiment was superceded by experiment 5R.
