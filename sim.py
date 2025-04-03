import numpy as np
from KBM import KinematicBicycleModel
from CGKBM import CGKinematicBicycleModel
import cvxpy as cp
from scripts import *
import matplotlib.pyplot as plt
import time

N_STATES = 4
N_ACTIONS = 2
L = 0.3
l_f = 0.1
l_r = 0.2
T = 10
DT = 0.2

MAX_SPEED = 1.5
MAX_STEER = np.radians(30)
MAX_D_ACC = 1.0
MAX_D_STEER = np.radians(30)  # rad/s
MAX_ACC = 1.0
REF_VEL = 1.0

kbm = KinematicBicycleModel(N_STATES, N_ACTIONS, L, T, DT)
cg_kbm = CGKinematicBicycleModel(N_STATES, N_ACTIONS, L, l_f, l_r, T, DT)

# Step 1: Create a sample track
track = generate_path_from_wp(
    [0, 3, 4, 6, 10, 12, 14, 6, 1, 0], [0, 0, 2, 4, 3, 3, -2, -6, -2, -2], 0.05
)

# To be converted
cg_a = np.load("cg_a.npy")
cg_state = np.load("cg_state.npy")

# Nothing to be done here.
d = np.load("cg_d.npy")

l_a = []

# Goal: convert accelerations from the CG to the rear wheel.

# We know for each time step, angular acceleration (d theta dot/ d t) = (d theta_dot / d delta) * a

for i in range(cg_a.shape[0]):
    # x_i, y_i, v_i, theta_i = cg_state[i]
    # # Calculate theta_dot (or omega)
    # beta = cg_kbm.calculate_sideslip(d[i])
    # omega = v_i * np.tan(d[i]) * np.cos(beta) / cg_kbm.L
    # l_a.append(cg_a[i] + omega**2 * l_r)
    l_a.append(cg_a[i])

u_sim = np.zeros((2, len(d)))
x_sim = np.zeros((4, len(d)))
for sim_time in range(len(d) - 1):
    # Take first action
    u_sim[:, sim_time] = np.array([l_a[sim_time],d[sim_time]])

    # move simulation to t+1
    tspan = [0, DT]
    x_sim[:, sim_time + 1] = kbm.forward_one_step(x_sim[:, sim_time], u_sim[:, sim_time])

# Visualization
fig = plt.figure(figsize=(10, 6))
gs = fig.add_gridspec(2, 2)

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
x = np.arange(len(d))
ax3.plot(x, d)
ax3.set_ylabel("steering command")
ax3.set_xlabel("time")

plt.show()
