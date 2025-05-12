import numpy as np
from KBM import KinematicBicycleModel
import cvxpy as cp
from scripts import *
import matplotlib.pyplot as plt

# This code does MPC for one time step to check that everything is working.

N_STATES = 4
N_ACTIONS = 2
L = 0.3
T = 10
DT = 0.2

MAX_SPEED = 1.5
MAX_STEER = np.radians(30)
MAX_ACC = 1.0
REF_VEL = 1.0

kbm = KinematicBicycleModel(N_STATES, N_ACTIONS, L, T, DT)

# Step 1: Create a sample track
track = generate_path_from_wp([0, 3, 6], [0, 0, 2], 0.05)

# Step 2: Create starting conditions x0
x0 = np.zeros(N_STATES)
x0[0] = 0
x0[1] = -0.25
x0[2] = 0.0
x0[3] = np.radians(0)

# Step 3: Generate starting guess for u_bar (does not have to be too accurate I suppose.)
u_bar = np.zeros((N_ACTIONS, T))
u_bar[0, :] = MAX_ACC / 2
u_bar[1, :] = 0.0

# Step 4: x_bar
x_bar = np.zeros((N_STATES, T + 1))
x_bar[:, 0] = x0

# Step 5: Rollout using dynamics to get x_bar
for t in range(1, T + 1):
    x_t = x_bar[:, t-1]
    u_t = u_bar[:, t-1]
    A, B, C = kbm.linearize_model(x_t, u_t)
    x_bar[:, t] = A @ x_t + B @ u_t + C.flatten()

x = cp.Variable((N_STATES, T + 1))
u = cp.Variable((N_ACTIONS, T))
cost = 0
constr = []

# Cost Matrices
Q = np.diag([10, 10, 10, 10])  # state error cost
Qf = np.diag([10, 10, 10, 10])  # state  final error cost
R = np.diag([10, 10])  # input cost
R_ = np.diag([10, 10])  # input rate of change cost

x_ref, _ = get_reference_trajectory(x_bar[:, 0], track, REF_VEL, 0.05)

for t in range(T):
    cost += cp.quad_form(x[:, t] - x_ref[:, t], Q)
    cost += cp.quad_form(u[:, t], R)
    if t < (T - 1):
        cost += cp.quad_form(u[:, t+1] - u[:, t], R_)
    
    A, B, C = kbm.linearize_model(x_bar[:, t], u_bar[:, t])
    constr += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C.flatten()]

cost += cp.quad_form(x[:, T] - x_ref[:, T], Qf)

constr += [x[:, 0] == x_bar[:, 0]]
constr += [x[2, :] <= MAX_SPEED]
constr += [x[2, :] >= 0.0]
constr += [cp.abs(u[0, :]) <= MAX_ACC]
constr += [cp.abs(u[1, :]) <= MAX_STEER]

prob = cp.Problem(cp.Minimize(cost), constr)
solution = prob.solve(solver=cp.OSQP, verbose=True)
breakpoint()

x_mpc = np.array(x.value[0, :]).flatten()
y_mpc = np.array(x.value[1, :]).flatten()
v_mpc = np.array(x.value[2, :]).flatten()
theta_mpc = np.array(x.value[3, :]).flatten()

a_mpc = np.array(u.value[0, :]).flatten()
delta_mpc = np.array(u.value[1, :]).flatten()

x_traj = kbm.forward(x0, np.vstack((a_mpc, delta_mpc)))

# Visualization
plt.subplot(2, 2, 1)
plt.plot(track[0, :], track[1, :], "b")
plt.plot(x_ref[0, :], x_ref[1, :], "g+")
plt.plot(x_traj[0, :], x_traj[1, :])
plt.axis("equal")
plt.ylabel("y")
plt.xlabel("x")
plt.show()

breakpoint()