# Modelling Interstellar Conflict

This repository contains the code for a research project modelling the interaction between civilisations in the universe.

The research has its origins in the master's thesis of Otto Kuusela at the University of Amsterdam. The thesis was supervised by Debraj Roy.

To learn more, read our paper “Higher-Order Reasoning under Intent Uncertainty Reinforces the Hobbesian Trap”, published at AAMAS 2024.

The code is currently at a “research level”. Refactoring is underway.

## How to Run the Model

The file **`run.py`** can be used to perform a single model run. You can adjust all parameters by changing the `params` dictionary. After the simulation is complete, the script creates a few plots and an animation. Make sure you create a folder named **`output`** in the root directory before you run the script.

## Software Requirements

This project was built using Python 3.11.3. To install the required software, create a new Conda environment with

    $ conda create --name <env> --file requirements.txt

Note that this is simply the state of the development environment and might therefore contain unnecessary packages.
