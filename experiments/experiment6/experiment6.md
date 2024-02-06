# Experiment 6

The goal of this experiment is to investigate the statistical properties of attacks at the “critical point” of zero attack reward. This is essentially a copy of experiment 2, but with two agents.

## Independent Variable(s)

None

## Dependent Variable(s)

Distribution of streaks of attacks and non-attacks

## Other Parameters

- two (2) civilisations
- attack reward 0
- reasoning level: 0 or 1 (=1 or 2 in the new notation)
- level 0 policy: random or passive

## Solver Parameters

- exploration coefficient: 0.3 or 0.6
- softargmax coefficient: 0.1 or 1
- 10k simulations per forest

## Simulation Details

Each parameter combination is simulated 8 times, resulting in a total of $2*2*2*2*8 = 128$ model evaluations.

A single run is 200 time steps long.

## Results

See **`experiment6_analysis.ipynb`**.
