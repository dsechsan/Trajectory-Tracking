from time import time
import numpy as np
from utils import visualize
import casadi as ca
# Simulation params
np.random.seed(10)
time_step = 0.5 # time between steps in seconds
sim_time = 120    # simulation time

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

def cec_controller(cur_state,iter):
    gamma = 1
    q = 1
    time_horizon = 20

    T = time_horizon
    ref_state = []
    next_ref_state = []
    
    opti = ca.Opti()
    x = opti.variable(3,T+1)
    Q = ca.diag([1,1])
    u_t = opti.variable(2,T)
    R = ca.diag([1,1])

    iter_in = iter
    for t in range(T):
        ref_state.append(np.array(lissajous(iter_in)).reshape([3,1]))
        next_ref_state.append(np.array(lissajous(iter_in+1)).reshape([3,1]))
        G_tilda = time_step*ca.horzcat(ca.vertcat(ca.cos(x[2,t]+ref_state[t][2,0]), ca.sin(x[2,t]+ref_state[t][2,0]), 0), ca.vertcat(0, 0, 1))
        f_tilda = ca.mtimes(G_tilda,u_t[:,t])
        opti.subject_to(x[:,t+1] == x[:,t] + f_tilda + ca.MX(ref_state[t]-next_ref_state[t]))
        iter_in += 1
    
    cost = 0
    for t in range(T-1):
        # err_state = x[:,t]
        cost += (gamma**t)*(ca.mtimes(ca.mtimes(x[:2,t].T,Q), x[:2,t]) + ca.mtimes(ca.mtimes(u_t[:,t].T,R), u_t[:,t]) + q*(1-ca.cos(x[2,t])**2))
    
    term = x[:,T]-ca.MX(lissajous(iter_in))
    opti.minimize(cost + term[0]**2 + term[1]**2 + (1-np.cos(term[2])**2))
    opti.subject_to(x[:,0] == (cur_state - np.array(lissajous(iter))).reshape([3,1]))
    

    for t in range(T):
        opti.subject_to(ca.norm_2(x[:2,t] + ref_state[t][:2,0] -ca.MX([-2,-2]))>0.5)
        opti.subject_to(ca.norm_2(x[:2,t] + ref_state[t][:2,0] -ca.MX([1,2]))>0.5)
        # if(x[2,t] + ref_state[t][2,0] > np.pi):
            
        # opti.subject_to(x[2,t] + ref_state[t][2,0] >= -ca.pi)
        # opti.subject_to(x[2,t] + ref_state[t][2,0]< ca.pi)
        opti.subject_to(x[0,t]+ ref_state[t][0,0] >=-3)
        opti.subject_to(x[1,t]+ ref_state[t][1,0] >=-3)
        opti.subject_to(x[0,t]+ ref_state[t][0,0] <=3)
        opti.subject_to(x[1,t]+ ref_state[t][1,0] <=3)

        opti.subject_to(u_t[0,t] >= 0)
        opti.subject_to(u_t[0,t] <= 1)
        opti.subject_to(u_t[1,t] >= -1)
        opti.subject_to(u_t[1,t] <= 1)
        
    
    opti.solver('ipopt')
    sol = opti.solve()
    u_opt = sol.value(u_t)
    return u_opt[:,0]
    

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
    mu, sigma = 0, 0.04 # mean and standard deviation for (x,y)
    w_xy = np.random.normal(mu, sigma, 2)
    mu, sigma = 0, 0.004  # mean and standard deviation for theta
    w_theta = np.random.normal(mu, sigma, 1)
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
        error = error + np.linalg.norm(cur_state - cur_ref)
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

