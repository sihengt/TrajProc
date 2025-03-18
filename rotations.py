import numpy as np
import matplotlib.pyplot as plt

# This rotation example is equivalent to saying that you want
# the car to have a displacement of [1, 0] locally.

# And you want to see it in global coordinates.

# Conversely, if you want the car to have a displacement of [x, y] globally.
# But get it in local coordinates,
# -yaw.

state = np.array([2.0, 2.0, np.pi/4])

fig, ax = plt.subplots(figsize=(6, 6))

def rotation(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])

distance = np.array([1, 0])
r_dist = rotation(-state[2]) @ distance
ax.quiver(state[0], state[1], 1, 0, color='r', angles='xy', scale_units='xy', scale=1, width=0.008, label="State vector")
ax.quiver(state[0], state[1], r_dist[0], r_dist[1], color='b', angles='xy', scale_units='xy', scale=1, width=0.008, label="Rotated state vector")
ax.set_xlim(0, 8)
ax.set_ylim(0, 8)
ax.legend()
ax.grid(True)
plt.show()
# ax1.quiver(state[0], state[1], )
