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
T = 10          # MPC horizon
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
        # x_bar_compare[:, 0] = x_sim[:, sim_time]

        # Step 5: Rollout using dynamics to get rest of x_bar
        for t in range(1, T + 1):
            # New implementation
            x_bar_t = x_bar[:, t-1]
            u_bar_t = u_bar[:, t-1]
            x_kp1, A_d, B_d, C_d = cs_kbm.integrate(x_bar_t, u_bar_t)
            x_bar[:, t] = x_kp1.full().flatten()

            # Comparing with previous implementation
            # A, B, C = kbm.linearize_model(x_bar_t, u_bar_t)
            # x_bar_compare[:, t] = A @ x_bar_t + B @ u_bar_t + C.flatten()
        
        g = []      # Constraint
        lbg = []    # Lower bound for constraints
        ubg = []    # Upper bound for constraints
        J = 0       # Cost

        # Initializing decision variables (X, U)
        X = cs.MX.sym('X', N_STATES, T + 1)

        # Initializing lower and upper bounds for state
        X_lb_row = cs.DM([-cs.inf, -cs.inf, 0, -cs.inf],)
        X_ub_row = cs.DM([cs.inf, cs.inf, MAX_SPEED, cs.inf],)
        X_lb = cs.repmat(X_lb_row, T + 1, 1)
        X_ub = cs.repmat(X_ub_row, T + 1, 1)
        
        U = cs.MX.sym('U', N_ACTIONS, T)
        U_lb_row = cs.DM([-MAX_ACC, -MAX_STEER, -MAX_STEER])
        U_ub_row = cs.DM([MAX_ACC, MAX_STEER, MAX_STEER])
        U_lb = cs.repmat(U_lb_row, T, 1)
        U_ub = cs.repmat(U_ub_row, T, 1)

        # Stacking decision variables and contraints
        vec = lambda M: cs.reshape(M, -1, 1)
        w = cs.vertcat(vec(X), vec(U))
        lbw = cs.vertcat(X_lb, U_lb)
        ubw = cs.vertcat(X_ub, U_ub)
        lbw = cs.vertcat(X_lb, U_lb)
        lbw = cs.vertcat(X_lb, U_lb)

        # Cost Matrices
        Q   = cs.DM(np.diag([20, 20, 10, 1e-3]))   # state error cost
        Qf  = cs.DM(np.diag([30, 30, 30, 1e-3]))   # state  final error cost
        R   = cs.DM(np.diag([10, 10, 10]))      # input cost
        R_  = cs.DM(np.diag([10, 10, 10]))      # input rate of change cost

        x_ref, _ = get_reference_trajectory(x_bar[:, 0], track, REF_VEL, 0.05)
        x_ref[3, :] = np.unwrap(x_ref[3, :])
        
        for k in range(T):
            # Initialize action for current_timestep
            e = X[:, k] - x_ref[:, k]
            J += e.T @ Q @ e
            J += U[:, k].T @ R @ U[:, k]

            # Handling constraints to smooth controls
            if k < (T - 1):
                e_u = U[:, k+1] - U[:, k]
                J += e_u.T @ R_ @ e_u
                
                d_u = e_u / DT
                g += [d_u]
                lbg += [-MAX_D_ACC, -MAX_D_STEER, -MAX_D_STEER]
                ubg += [MAX_D_ACC, MAX_D_STEER, MAX_D_STEER]

            x_kp1, A, B, C = cs_kbm.integrate(x_bar[:, k], u_bar[:, k])
            
            g += [A @ X[:, k] + B @ U[:, k] + C - X[:, k + 1]]
            lbg += [0] * N_STATES
            ubg += [0] * N_STATES

        e_xn = X[:, T] - x_ref[:, T]
        J += e_xn.T @ Qf @ e_xn

        g += [X[:, 0] - x_bar[:, 0]]
        lbg += [0] * N_STATES
        ubg += [0] * N_STATES

        qp_opts = {'printLevel': 'none'}
        prob    = {'x': w, 'f': J, 'g': cs.vertcat(*g)}
        solver  = cs.qpsol('solver', 'qpoases', prob, qp_opts)
        sol     = solver(lbx=lbw, ubx=ubw, lbg=cs.vertcat(*lbg), ubg=cs.vertcat(*ubg))

        X_mpc = cs.reshape(sol['x'][:N_STATES * (T+1)], 4, T+1).full()
        U_mpc = cs.reshape(sol['x'][N_STATES * (T+1):], 3, T).full()
        
        a_mpc   = np.array(U_mpc[0, :]).flatten()
        d_f_mpc = np.array(U_mpc[1, :]).flatten()
        d_r_mpc = np.array(U_mpc[2, :]).flatten()
        u_bar_new = np.vstack((a_mpc, d_f_mpc, d_r_mpc))

        delta_u = np.sum(np.sum(np.abs(u_bar_new - u_bar), axis=0), axis=0)
        if delta_u < 0.05:
            break
            
        u_bar = u_bar_new
    
    current_state = X_mpc[:, 0]
    l_state.append(current_state)
    x_mpc       = np.array(X_mpc[0, :]).flatten()
    y_mpc       = np.array(X_mpc[1, :]).flatten()
    v_mpc       = np.array(X_mpc[2, :]).flatten()
    theta_mpc   = np.array(X_mpc[3, :]).flatten()

    a_mpc   = np.array(U_mpc[0, :]).flatten()
    df_mpc  = np.array(U_mpc[1, :]).flatten()
    dr_mpc  = np.array(U_mpc[2, :]).flatten()

    l_a.append(a_mpc[0])
    l_df.append(df_mpc[0])
    l_dr.append(dr_mpc[0])

    u_bar = np.vstack((a_mpc, df_mpc, dr_mpc))
    
    # Take first action
    u_sim[:, sim_time] = u_bar[:, 0]
    
    # Measure elpased time to get results from cvxpy
    opt_time.append(time.time() - iter_start)

    # move simulation to t+1
    # tspan = [0, DT]
    x_sim[:, sim_time + 1] = cs_kbm.forward_one_step(x_sim[:, sim_time], u_sim[:, sim_time])
    # x_sim[:, sim_time + 1] = kbm.forward_one_step(x_sim[:, sim_time], u_sim[:, sim_time])

print("TOTAL TIME: {}".format(np.sum(opt_time)))
# 12.10 seconds
# 18.933314323425293 (previous implementation)

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
