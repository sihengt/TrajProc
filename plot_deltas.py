import numpy as np
import matplotlib.pyplot as plt
from CGKBM import CGKinematicBicycleModel

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

cg_kbm = CGKinematicBicycleModel(N_STATES, N_ACTIONS, L, l_f, l_r, T, DT)

# cg_d = np.load("cg_d.npy")
# d = np.load("d.npy")

# x = np.arange(cg_d.shape[0])

# plt.plot(x, cg_d, color='red', label="CG KBM")
# plt.plot(x, d, color='green', label="KBM")
# plt.xlabel("Time")
# plt.ylabel("delta (rads)")
# plt.legend()
# plt.show()

cg_a = np.load("cg_a.npy")
a = np.load("a.npy")
cg_state = np.load("cg_state.npy")
d = np.load("cg_d.npy")

l_a = []
for i in range(cg_a.shape[0]):
    x_i, y_i, v_i, theta_i = cg_state[i]
    # Calculate theta_dot (or omega)
    beta = cg_kbm.calculate_sideslip(d[i])
    omega = v_i * np.tan(d[i]) * np.cos(beta) / cg_kbm.L
    l_a.append(cg_a[i] + omega**2 * l_r)

x = np.arange(cg_a.shape[0])

plt.plot(x, cg_a, color='red', label="CG KBM")
plt.plot(x, a, color='green', label="KBM")
plt.plot(x, l_a, color='plum', label="CG KBM Transformed")
plt.xlabel("Time")
plt.ylabel("accel (m/s^2)")
plt.legend()
plt.show()
