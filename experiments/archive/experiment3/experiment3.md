# Experiment 3

The goal of this experiment is to establish the relationship between the attack reward and the overall behaviour of the system. It also serves as a sensitivity analysis: two system parameters (attack reward, discount factor) and one solver parameter (softargmax coefficient) are varied in a Saltelli sample. This allows for calculating sensitivity indices.

## Independent Variable(s)

Attack reward; varies in the range [-0.2, 0.1]

## Dependent Variable(s)

- proportions of the different actions (no action, hide, attack)

## Other Parameters

- two (2) civilisations
- reasoning level: 0 or 1 (=1 or 2 in the new notation)
- discount factor: [0.5, 0.7]

## Solver Parameters
- exploration coefficient: 0.1
- softargmax coefficient: [0.01, 1]
- 10k simulations per forest

## Simulation Details

The script supports a variable number of Saltelli samples (see instructions in the code). We did 128 samples, resulting in a total of 1280 model evaluations.

Runs last for 100 time steps.

## Results

This experiment was replaced by the more comprehensive experiment 3R.
