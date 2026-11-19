# UCSD ECE276B PR3 

## Overview
In this assignment, two different controllers were implemented to make a ground differential-drive robot track a lissajous trajectory safely while avoiding obstacles in the way. More details can be found in the ECE276B_PR3.pdf file. 
## Starter code
### 1. main.py
This file contains examples of how to generate control inputs from a simple P controller and apply the control on a car model.The task is to replace the P controller with our own controller using CEC and GPI as described in the project. This file contains the CEC implementation. 

The CEC controller builds its CasADi/IPOPT problem once, updates state and
reference parameters each step, and warm-starts with shifted controls. Its
world-coordinate objective penalizes position error once, uses the periodic
heading cost `1 - cos(heading_error)`, and includes every control stage and
the terminal tracking error. All future predicted states, including the terminal
state, have workspace and obstacle constraints.

Obstacles have a 0.12-unit clearance buffer (three times the simulated position
noise standard deviation). The measured initial state is not constrained, so a
measurement inside the buffer does not itself make the problem infeasible.
This is nominal predictive control, not a guarantee of safety under unbounded
Gaussian noise; an infeasible solve raises an error rather than returning an
unchecked control. The reported tracking error compares states and references
at the same time and wraps heading differences to `[-pi, pi)`.

Install the CEC dependencies with `python -m pip install numpy matplotlib casadi`.
Run the simulation from the repository root with `python starter_code/main.py`.
Run the controller regression tests with
`python -m unittest discover -s starter_code -p 'test_cec.py'`.

<img src="https://github.com/dsechsan/Trajectory-Tracking/blob/09b8302673f58925ed703b42a834186b42e50ee1/fig/animation1686890140.158695.gif" alt="Using CEC" width="400" height="400">

### 2. GPI_main.py

This file contains the generalized policy iteration implementation to track the given trajectory. 