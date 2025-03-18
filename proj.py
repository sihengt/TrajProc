import numpy as np
import matplotlib.pyplot as plt

state = np.array([2, 3])
state_2 = np.array([3.5, 4])

path = np.array([[1, 3, 5, 7], [1, 3, 5, 7]])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

def calc_and_plot_vectors(ax, path, state):
    dist = np.linalg.norm(np.expand_dims(state, 1) - path, axis=0)
    nn_idx = np.argmin(dist)

    if nn_idx < path.shape[1] - 1:
        v = np.array([path[0, nn_idx + 1] - path[0, nn_idx],
                    path[1, nn_idx + 1] - path[1, nn_idx]])
        v = v / np.linalg.norm(v)

    d = path[:, nn_idx] - state

    proj_length = np.dot(d, v)
    proj_d_on_v = proj_length * v

    ax.plot(path[0], path[1], 'bo-', label="path")

    ax.plot(state[0], state[1], 'ro', markersize=8, label="state")

    ax.plot(path[0, nn_idx], path[1, nn_idx], 'go', markersize=8, label="NN")

    ax.quiver(path[0, nn_idx], path[1, nn_idx], v[0], v[1], color='b', angles='xy', scale_units='xy', scale=1, width=0.008, label="Direction Vector v")
    ax.quiver(state[0], state[1], d[0], d[1], color='r', angles='xy', scale_units='xy', scale=1, width=0.008, label="Displacement Vector d")
    ax.quiver(state[0], state[1], proj_d_on_v[0], proj_d_on_v[1], color='purple', angles='xy', scale_units='xy', scale=1, width=0.008, label="Projection of d onto v")

    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.legend()
    ax.grid(True)

calc_and_plot_vectors(ax1, path, state)
calc_and_plot_vectors(ax2, path, state_2)

plt.show()