# Experiment 3R

The goal of this experiment is to establish the relationship between the attack reward and the overall behaviour of the system. It also serves as a sensitivity analysis: two system parameters (attack reward, discount factor) and two solver parameters (exploration coefficient, softargmax coefficient) are varied in a Saltelli sample. This allows for calculating sensitivity indices.

The experiment is a Re-do of experiment 3 with the addition of the exploration coefficient to the Saltelli sample and testing for two different level 0 policies.

## Independent Variable(s)

Attack reward; varies in the range [-0.2, 0.1]

## Dependent Variable(s)

- proportions of the different actions (no action, hide, attack)

## Other Parameters

- two (2) civilisations
- reasoning level: 0 or 1 (=1 or 2 in the new notation)
- level 0 policy: random or passive
- discount factor: [0.5, 0.7]

## Solver Parameters
- exploration coefficient: [0.1, 1]
- softargmax coefficient: [0.01, 1]
- 10k simulations per forest

## Simulation Details

The script supports a variable number of Saltelli samples (see instructions in the code). We did 64 samples, resulting in a total of 1536 model evaluations.

Runs last for 100 time steps.

## Results

See **`experiment3R_analysis.ipynb`**.
