# Experiment 2AS

The goal of this experiment is to investigate the behaviour of the system with three civilisations. Instead of the prior random activation schedule where a randomly chosen civilisations gets to act each time step, here all civilisations get to act at every step (joint Activation Schedule).

## Independent Variable(s)

None

## Dependent Variable(s)

- proportions of the different actions (no action, hide, attack)
- attack streak length distribution
- average reward per timestep (how “collectively optimal” the system is)
- prediction error for other civilisations' action utilities

## Other Parameters

- three (3) civilisations
- attack reward 0
- reasoning level 1 (=2 in the new notation)

## Solver Parameters
- exploration coefficient: 0.3 or 0.6
- softargmax coefficient: 0.1 or 1
- 8000 simulations per forest (+20% faster than the default of 10k)

## Simulation Details

We will perform 8 repetitions for each combination of solver parameters, which means that there will be a total of $2 * 2 * 8 = 32$ simulations.

Runs stop after 200 time steps.

## Results


