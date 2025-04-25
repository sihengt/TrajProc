from DKBM import DSKinematicBicycleModel
from DKBM_casadi import csDSKBM
from scripts import *

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import time
import casadi as cs

N_STATES = 4
N_ACTIONS = 3
L = 0.3
l_f = 0.1
l_r = 0.2
T = 10          # Time
N = 50          # Control Interval
DT = 0.2        # dt = T/N

MAX_SPEED = 1.5
MAX_STEER = np.radians(30)
MAX_D_ACC = 1.0
MAX_D_STEER = np.radians(30)  # rad/s
MAX_ACC = 1.0
REF_VEL = 1.0

cs_kbm = csDSKBM(N_STATES, N_ACTIONS, L, l_f, l_r, T, N)
kbm = DSKinematicBicycleModel(N_STATES, N_ACTIONS, L, l_f, l_r, T, DT)

# Step 1: Create a sample track
track = generate_path_from_wp(
    [0, 3, 4, 6, 10, 12, 14, 6, 1, 0], [0, 0, 2, 4, 3, 3, -2, -6, -2, -2], 0.05
)

sim_duration = 200  # time steps
opt_time = []

# VARIABLES FOR TRACKING
x_sim = np.zeros((N_STATES, sim_duration))
u_sim = np.zeros((N_ACTIONS, sim_duration - 1))

# Step 2: Create starting conditions x0
x_sim[:, 0] = np.array([0.0, -0.25, 0.0, np.radians(0)]).T

# Step 3: Generate starting guess for u_bar (does not have to be too accurate I suppose.)
u_bar_start = np.zeros((N_ACTIONS, T))
u_bar_start[0, :] = MAX_ACC / 2
u_bar_start[1, :] = 0.0
u_bar_start[2, :] = 0.0

l_a = []
l_df = []
l_dr = []
l_state = []

for sim_time in range(sim_duration - 1):
    iter_start = time.time()

    for i_iter in range(5):
        if i_iter == 0:
            u_bar = u_bar_start
        
        # Step 4: x_bar
        # Setting the very first x_bar to be same as the simulator
        x_bar = np.zeros((N_STATES, T + 1))
        x_bar_compare = np.zeros((N_STATES, T + 1))
        x_bar[:, 0] = x_sim[:, sim_time]
        x_bar_compare[:, 0] = x_sim[:, sim_time]

        # Step 5: Rollout using dynamics to get rest of x_bar
        for t in range(1, T + 1):
            # New implementation
            x_bar_t = x_bar[:, t-1]
            u_bar_t = u_bar[:, t-1]
            x_kp1, A_d, B_d, C_d = cs_kbm.integrate(x_bar_t, u_bar_t)
            x_bar[:, t] = x_kp1.full().flatten()

            # Comparing with previous implementation
            A, B, C = kbm.linearize_model(x_bar_t, u_bar_t)
            x_bar_compare[:, t] = A @ x_bar_t + B @ u_bar_t + C.flatten()
        
        breakpoint()
        
        x = cp.Variable((N_STATES, T + 1))
        u = cp.Variable((N_ACTIONS, T))
        cost = 0
        constr = []

        # Cost Matrices
        Q = np.diag([20, 20, 10, 0])  # state error cost
        Qf = np.diag([30, 30, 30, 0])  # state  final error cost
        R = np.diag([10, 10, 10])  # input cost
        R_ = np.diag([10, 10, 10])  # input rate of change cost

        x_ref, _ = get_reference_trajectory(x_bar[:, 0], track, REF_VEL, 0.05)
        x_ref[3, :] = np.unwrap(x_ref[3, :])

        for t in range(T):
            cost += cp.quad_form(x[:, t] - x_ref[:, t], Q)
            cost += cp.quad_form(u[:, t], R)
            if t < (T - 1):
                cost += cp.quad_form(u[:, t+1] - u[:, t], R_)
                constr += [
                    cp.abs(u[0, t + 1] - u[0, t]) / DT <= MAX_D_ACC
                ]
                constr += [
                    cp.abs(u[1, t + 1] - u[1, t]) / DT <= MAX_D_STEER
                ]
                constr += [
                    cp.abs(u[2, t + 1] - u[2, t]) / DT <= MAX_D_STEER
                ]
            
            A, B, C = kbm.linearize_model(x_bar[:, t], u_bar[:, t])
            constr += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C.flatten()]

        cost += cp.quad_form(x[:, T] - x_ref[:, T], Qf)

        constr += [x[:, 0] == x_bar[:, 0]]
        constr += [x[2, :] <= MAX_SPEED]
        constr += [x[2, :] >= 0.0]
        constr += [cp.abs(u[0, :]) <= MAX_ACC]
        constr += [cp.abs(u[1, :]) <= MAX_STEER]

        prob = cp.Problem(cp.Minimize(cost), constr)
        solution = prob.solve(solver=cp.OSQP, verbose=False)
        a_mpc   = np.array(u.value[0, :]).flatten()
        d_f_mpc = np.array(u.value[1, :]).flatten()
        d_r_mpc = np.array(u.value[2, :]).flatten()
        u_bar_new = np.vstack((a_mpc, d_f_mpc, d_r_mpc))

        delta_u = np.sum(np.sum(np.abs(u_bar_new - u_bar), axis=0), axis=0)
        if delta_u < 0.05:
            break
            
        u_bar = u_bar_new
    
    current_state = x.value[:, 0]
    l_state.append(current_state)
    x_mpc = np.array(x.value[0, :]).flatten()
    y_mpc = np.array(x.value[1, :]).flatten()
    v_mpc = np.array(x.value[2, :]).flatten()
    theta_mpc = np.array(x.value[3, :]).flatten()

    a_mpc = np.array(u.value[0, :]).flatten()
    df_mpc = np.array(u.value[1, :]).flatten()
    dr_mpc = np.array(u.value[2, :]).flatten()

    l_a.append(a_mpc[0])
    l_df.append(df_mpc[0])
    l_dr.append(dr_mpc[0])

    u_bar = np.vstack((a_mpc, df_mpc, dr_mpc))
    
    # Take first action
    u_sim[:, sim_time] = u_bar[:, 0]

    # Measure elpased time to get results from cvxpy
    opt_time.append(time.time() - iter_start)

    # move simulation to t+1
    tspan = [0, DT]
    x_sim[:, sim_time + 1] = kbm.forward_one_step(x_sim[:, sim_time], u_sim[:, sim_time])

    # x_traj = kbm.forward(x0, np.vstack((a_mpc, delta_mpc)))

# Visualization
fig = plt.figure(figsize=(10, 6))
gs = fig.add_gridspec(2, 3)

ax1 = fig.add_subplot(gs[0, :])
ax1.plot(track[0, :], track[1, :], "b")
ax1.scatter(x_sim[0, :], x_sim[1, :], s=0.5, color='red', zorder=1)
# for i in range(x_sim.shape[1]):
#     ax1.text(
#         x_sim[0, i], x_sim[1, i], str(i),
#         fontsize=4, color='black', ha='center', va='center',
#         zorder=2  # Make sure it's above the scatter dots
#     )

ax1.plot(x_sim[0, :], x_sim[1, :], color='green', zorder=-1)
# plt.plot(x_traj[0, :], x_traj[1, :])
ax1.axis("equal")
ax1.set_ylabel("y")
ax1.set_xlabel("x")

ax2 = fig.add_subplot(gs[1, 0]) 
x = np.arange(len(l_a))
ax2.plot(x, l_a)
ax2.set_ylabel("acceleration command")
ax2.set_xlabel("time")

ax3 = fig.add_subplot(gs[1, 1])
x = np.arange(len(l_df))
ax3.plot(x, l_df)
ax3.set_ylabel("Front steering")
ax3.set_xlabel("time")

ax4 = fig.add_subplot(gs[1, 2])
x = np.arange(len(l_dr))
ax4.plot(x, l_dr)
ax4.set_ylabel("Rear steering")
ax4.set_xlabel("time")
plt.show()

# Test
# cs_kbm.integrate(np.random.rand(4, 1), np.random.rand(3, 1))
# cs_kbm.RK4(np.random.rand(4, 1), np.random.rand(3, 1), np.random.rand(4, 4), np.random.rand(4, 3))
