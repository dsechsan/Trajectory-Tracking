# UCSD ECE276B PR3 

## Overview
In this assignment, you will implement a controller for a car robot to track a trajectory.

## Dependencies
The original starter code was tested with Python 3.7, matplotlib 3.4, and numpy
1.20. The CEC implementation also requires CasADi (including its IPOPT solver):
`python -m pip install numpy matplotlib casadi`.

## Starter code
### 1. main.py
This file contains examples of how to generate control inputs from a simple P controller and apply the control on a car model. This simple controller does not work well. Your task is to replace the P controller with your own controller using CEC and GPI as described in the project statement.

The implemented CEC controller reuses a parameterized solver with warm starts.
See the repository README for its objective, safety limitations, and run commands.

### 2. utils.py
This file contains code to visualize the desired trajectory, robot's trajectory, and obstacles.

