from time import time
import numpy as np
from utils import visualize
import casadi as ca
# Simulation params
np.random.seed(10)
time_step = 0.5 # time between steps in seconds
sim_time = 120    # simulation time
position_noise_std = 0.04
heading_noise_std = 0.004
obstacle_clearance = 3 * position_noise_std

# Car params
x_init = 1.5
y_init = 0.0
theta_init = np.pi/2
v_max = 1
v_min = 0
w_max = 1
w_min = -1

# This function returns the reference point at time step k
def lissajous(k):
    xref_start = 0
    yref_start = 0
    A = 2
    B = 2
    a = 2*np.pi/50
    b = 3*a
    T = np.round(2*np.pi/(a*time_step))
    k = k % T
    delta = np.pi/2
    xref = xref_start + A*np.sin(a*k*time_step + delta)
    yref = yref_start + B*np.sin(b*k*time_step)
    v = [A*a*np.cos(a*k*time_step + delta), B*b*np.cos(b*k*time_step)]
    thetaref = np.arctan2(v[1], v[0])
    return [xref, yref, thetaref]

# This function implements a simple P controller
def simple_controller(cur_state, ref_state):
    k_v = 0.55
    k_w = 1.0
    v = k_v*np.sqrt((cur_state[0] - ref_state[0])**2 + (cur_state[1] - ref_state[1])**2)
    v = np.clip(v, v_min, v_max)
    angle_diff = ref_state[2] - cur_state[2]
    angle_diff = (angle_diff + np.pi) % (2 * np.pi ) - np.pi
    w = k_w*angle_diff
    w = np.clip(w, w_min, w_max)
    return [v,w]

class CECController:
    def __init__(self, horizon=20):
        self.horizon = horizon
        self.opti = opti = ca.Opti()
        self.x = x = opti.variable(3, horizon + 1)
        self.u = u = opti.variable(2, horizon)
        self.initial_state = opti.parameter(3)
        self.reference = reference = opti.parameter(3, horizon + 1)
        self.previous_controls = None

        # Use world coordinates so each tracking error subtracts the reference once.
        opti.subject_to(x[:, 0] == self.initial_state)
        cost = 0
        for t in range(horizon):
            velocity = ca.vertcat(
                u[0, t] * ca.cos(x[2, t]),
                u[0, t] * ca.sin(x[2, t]),
                u[1, t],
            )
            opti.subject_to(x[:, t + 1] == x[:, t] + time_step * velocity)
            error = x[:, t] - reference[:, t]
            cost += ca.sumsqr(error[:2]) + ca.sumsqr(u[:, t])
            cost += 1 - ca.cos(error[2])

        terminal_error = x[:, horizon] - reference[:, horizon]
        cost += ca.sumsqr(terminal_error[:2]) + 1 - ca.cos(terminal_error[2])
        opti.minimize(cost)

        # The measured state is fixed; constrain only states the controls can change.
        for t in range(1, horizon + 1):
            opti.subject_to(opti.bounded(-3, x[:2, t], 3))
            for center in ([-2, -2], [1, 2]):
                opti.subject_to(
                    ca.sumsqr(x[:2, t] - ca.DM(center)) >= (0.5 + obstacle_clearance)**2
                )
        opti.subject_to(opti.bounded(v_min, u[0, :], v_max))
        opti.subject_to(opti.bounded(w_min, u[1, :], w_max))
        opti.solver('ipopt', {'print_time': False}, {'print_level': 0, 'sb': 'yes'})

    def __call__(self, cur_state, iteration):
        reference = np.array([
            lissajous(iteration + t) for t in range(self.horizon + 1)
        ]).T
        self.opti.set_value(self.initial_state, cur_state)
        self.opti.set_value(self.reference, reference)

        if self.previous_controls is None:
            displacement = np.diff(reference[:2], axis=1)
            heading = reference[2, :-1]
            speed = (displacement[0] * np.cos(heading)
                     + displacement[1] * np.sin(heading)) / time_step
            angle_change = np.diff(reference[2])
            turn_rate = ((angle_change + np.pi) % (2 * np.pi) - np.pi) / time_step
            controls = np.vstack((
                np.clip(speed, v_min, v_max),
                np.clip(turn_rate, w_min, w_max),
            ))
        else:
            controls = np.column_stack((
                self.previous_controls[:, 1:], self.previous_controls[:, -1]
            ))

        # Roll out from the measured state; the previous prediction omits process noise.
        states = np.empty((3, self.horizon + 1))
        states[:, 0] = cur_state
        for t in range(self.horizon):
            theta = states[2, t]
            states[:, t + 1] = states[:, t] + time_step * np.array([
                controls[0, t] * np.cos(theta),
                controls[0, t] * np.sin(theta),
                controls[1, t],
            ])
        self.opti.set_initial(self.x, states)
        self.opti.set_initial(self.u, controls)
        solution = self.opti.solve()
        self.previous_controls = np.asarray(solution.value(self.u)).reshape(2, self.horizon)
        return self.previous_controls[:, 0].copy()


_cec_optimizer = None


def cec_controller(cur_state, iteration):
    global _cec_optimizer
    if _cec_optimizer is None:
        _cec_optimizer = CECController()
    return _cec_optimizer(cur_state, iteration)

# def error_next_state(err_state,iter,cur_state,control,noise):
#     ref_state = lissajous(iter)
#     next_ref_state = lissajous(iter+1)
#     G_tilda = np.array([[np.cos(err_state[2] + ref_state[2]),0],[np.sin(err_state[2] + ref_state[2]),0],[0,1]])
#     f_tilda = G_tilda @ control
#     w_xy = np.random.normal(mu, sigma, 2)
#     mu, sigma = 0, 0.004  # mean and standard deviation for theta
#     w_theta = np.random.normal(mu, sigma, 1)
#     w = np.concatenate((w_xy, w_theta))
#     if(noise):
#         return cur_state + time_step*f_tilda.flatten() + (next_ref_state - ref_state)+ w
#     else:
#         return cur_state + time_step*f_tilda.flatten() + (next_ref_state - ref_state)
    
# This function implement the car dynamics
def car_next_state(time_step, cur_state, control, noise = True):
    theta = cur_state[2]
    rot_3d_z = np.array([[np.cos(theta), 0], [np.sin(theta), 0], [0, 1]])
    f = rot_3d_z @ control
    w_xy = np.random.normal(0, position_noise_std, 2)
    w_theta = np.random.normal(0, heading_noise_std, 1)
    w = np.concatenate((w_xy, w_theta))
    if noise:
        return cur_state + time_step*f.flatten() + w
    else:
        return cur_state + time_step*f.flatten()

if __name__ == '__main__':
    # Obstacles in the environment
    obstacles = np.array([[-2,-2,0.5], [1,2,0.5]])
    # Params
    traj = lissajous
    ref_traj = []
    error = 0.0
    car_states = []
    times = []

    # Start main loop
    main_loop = time()  # return time in sec
    # Initialize state
    cur_state = np.array([x_init, y_init, theta_init])
    cur_iter = 0
    # Main loop
    while (cur_iter * time_step < sim_time):
        t1 = time()
        # Get reference state
        cur_time = cur_iter*time_step
        cur_ref = traj(cur_iter)
        # Save current state and reference state for visualization
        ref_traj.append(cur_ref)
        car_states.append(cur_state)
        tracking_error = cur_state - np.array(cur_ref)
        tracking_error[2] = (tracking_error[2] + np.pi) % (2 * np.pi) - np.pi
        error += np.linalg.norm(tracking_error)

        ################################################################
        # Generate control input
        # TODO: Replace this simple controller with your own controller
        # control = simple_controller(cur_state, cur_ref)
        # print("[v,w]", control)
        
        control = cec_controller(cur_state,cur_iter)
        ################################################################

        # Apply control input
        next_state = car_next_state(time_step, cur_state, control, noise=True)
        # Update current state
        cur_state = next_state
        # Loop time
        t2 = time()
        print(cur_iter)
        print(t2-t1)
        times.append(t2-t1)
        cur_iter = cur_iter + 1

    main_loop_time = time()
    print('\n\n')
    print('Total time: ', main_loop_time - main_loop)
    print('Average iteration time: ', np.array(times).mean() * 1000, 'ms')
    print('Final error: ', error)

    # Visualization
    ref_traj = np.array(ref_traj)
    car_states = np.array(car_states)
    times = np.array(times)
    visualize(car_states, ref_traj, obstacles, times, time_step, save=True)

