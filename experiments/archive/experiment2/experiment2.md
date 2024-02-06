# Experiment 2

The goal of this experiment is to investigate the behaviour of the system with
many (either 3, 5, or 7) civilisations.

## Independent Variable(s)

The number of agents (3 / 5 / 7)

## Dependent Variable(s)

- proportions of the different actions (no action, hide, attack)
- attack streak length distribution
- average reward per timestep (how “collectively optimal” the system is)
- prediction error for other civilisations' action utilities

## Other Parameters

- attack reward 0
- reasoning level 1 (=2 in the new notation)

## Solver Parameters
- exploration coefficient 0.1
- softargmax coefficient 0.01

## Simulation Details

We will perform 10 repetitions for each agent count, which means that there will be a total of 30 simulations.

To decide when to stop a single run, we keep simulating until the attack streak distribution doesn't change significantly. We measure the change every 50 steps using Jensen-Shannon divergence. Specifically, the attack streak length distribution is considered unchanging when 
$$
\begin{align*}
& \text{JSD} \left( P^{(t+50)} \mid \mid P^{(t)} \right) \\ &= \frac{1}{2} D(P^{(t)} \mid \mid M) + \frac{1}{2} D (P^{(t+50)} \mid \mid M) \\
& < \varepsilon
\end{align*}
$$
where $M=(1/2)(P^{(t)} + P^{(t+50)})$ and $D$ is the Kullback-Leibler divergence.

## Results

This experiment was not run to completion due to performance issues.

With five agents, the maximum memory usage is about 12-13GB for a single simulation.

Note: the analysis notebook doesn't seem to fully correspond to this experiment specification -- I must've changed it later.
