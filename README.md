# UCSD ECE276B PR3 

## Overview
In this assignment, two different controllers were implemented to make a ground differential-drive robot track a lissajous trajectory safely while avoiding obstacles in the way. More details can be found in the ECE276B_PR3.pdf file. 
## Starter code
### 1. main.py
This file contains examples of how to generate control inputs from a simple P controller and apply the control on a car model.The task is to replace the P controller with our own controller using CEC and GPI as described in the project. This file contains the CEC implementation. 

<img src="https://github.com/dsechsan/Trajectory-Tracking/blob/09b8302673f58925ed703b42a834186b42e50ee1/fig/animation1686890140.158695.gif" alt="Using CEC" width="400" height="400">

### 2. GPI_main.py

Part 2 implements both **discounted value iteration** and **policy iteration**
on the same periodic tracking-error MDP, as allowed in `ECE276B_PR3.pdf`.
Run either solver from the repository root:

```sh
python -m pip install numpy numba matplotlib
python starter_code/GPI_main.py --method value --output value_policy.npy
python starter_code/GPI_main.py --method policy --output pi_policy.npy
```

Both use 100 phases, a 20 x 20 x 20 error grid and 10 x 20 controls. Position
bins are finer near zero. The heading and phase wrap; sampled positions beyond
the error grid are clipped to its boundary. Each action uses three Gaussian
quadrature samples per dimension, interpolated onto neighboring grid points.
Their probabilities factor across the three independent noise dimensions. Only
one-dimensional neighbor indices and weights are stored; neither method
constructs the original 10.24 TB transition array or a full state-action cost table.

The discounted objective uses `gamma=0.95`, identity position weights, control
effort weight `0.1`, and the project's `(1 - cos(heading_error))**2` penalty.
An additional 1000 cost applies at unsafe **world positions**, including a
0.15-unit margin around obstacles. This discourages collisions but does not
guarantee avoidance on a noisy continuous rollout. The reference itself enters
an obstacle and sometimes requires turns faster than the robot's limit, so
perfect tracking is not always feasible. Policy iteration evaluates the current
policy to tolerance before a greedy improvement; value iteration uses greedy
backups throughout. Both require a final optimal Bellman residual at most
`1e-4` by default.

Each command saves an integer action-index array, then runs a seeded, noisy
240-step tracking simulation and animates the resulting trajectory (pass
`--no-plot` to skip the animation). Regenerate policies saved by earlier versions
to benefit from the updated transitions and costs. To reuse a saved policy with
the same grid, set
`model = GPI()` and `model.policy = np.load("value_policy.npy")`; call
`model.control(step, world_state)` for the linear and angular velocity.