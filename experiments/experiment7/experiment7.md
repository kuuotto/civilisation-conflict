# Experiment 7

The goal of this experiment is to determine the optimal policy for a civilisation in a specific scenario: when a weak civilisation is about to surpass a stronger civilisation in technological capability.

This expands on experiment 5R by adding a twist: while all civilisations are weakly universalist (have an attack reward of -0.1), we assume that with a probability of 50% they believe the other civilisation is selfish (has an attack reward of 0).

We are interested in whether the stronger civilisation will engage in pre-emptive hostility, creating a Hobbesian trap. We also investigate whether it is optimal for the weaker civilisation to spend effort hiding its technosignature.

## Hypothesis

We expect to demonstrate a Hobbesian trap.

## Independent Variable(s)

- $p_\text{surpass}^2$ in the range [0, 1]
- $p_\text{surpass}^1$ in the range [0, 1]

## Dependent Variable(s)

Utilities of the different action options in the first time step

## Other Parameters

- time until surpass: 2 or 4
- two (2) civilisations
- attack reward: -0.1
- reasoning level: 1 (=2 in the new notation)
- level 0 policy: random

## Solver Parameters

- exploration coefficient: 0.6
- softargmax coefficient: 0.1
- 10k simulations per forest

## Simulation Details

The values of the independent variables are from a Saltelli sample with 128 samples.

In total, there are $2 * 128 * (2 + 2) = 1024$ model evaluations. One evaluation consists of only one time step.

## Results

See **`experiment7_analysis.ipynb`**.
