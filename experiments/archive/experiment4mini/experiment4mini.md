# Experiment 4 mini

The goal of this experiment is to investigate the overall nature of the universe when civilisations have uncertainty about the attack rewards (moralities) of other civilisations.

This is called mini because it is a smaller version of an earlier experiment, called experiment 4.

## Independent Variable(s)

attack reward: [-0.2, 0.1]

## Dependent Variable(s)

proportions of different actions (no action, hide, attack)

## Other Parameters

- probability that civilisations think the other is selfish (p_indifferent): 0.5
- two (2) civilisations
- reasoning level: 1 (=2 in the new notation)
- level 0 policy: random

## Solver Parameters

- exploration coefficient: 0.5
- softargmax coefficient: 0.1
- 9k simulations per forest (+10% faster)

## Simulation Details

The values of the attack reward parameter are from a Saltelli sample with 32 samples.

In total, there are $1 * 32 * (1 + 2) = 96$ model evaluations. One evaluation consists of 100 time steps.

## Results

See **`experiment4mini_analysis.ipynb`**.
