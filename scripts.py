import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def generate_path_from_wp(wp_xs, wp_ys, step=0.1):
    """
    Params:
        wp_xs: list of starting x points to form trajectory with
        wp_ys: list of starting y points to form trajectory with
        step: defines the desired distance we need between subsequent points.
    
    Returns:
        Waypoints with step between them in the following format:
        [final_x;
         final_y]
    """
    path_xs = []
    path_ys = []

    N_points = len(wp_xs)
    # Calculate the difference between each point and its subsequent point.
    for i in range(N_points - 1):
        section_length = np.linalg.norm([wp_xs[i+1] - wp_xs[i], wp_ys[i+1] - wp_ys[i]])
        
        # Given section length, calculates n_points we need to interpolate between subsequent coordinates.
        # Special case: only include the endpoint for the last point to prevent duplicates
        if i == N_points - 2:
            interp_range = np.linspace(0, 1, np.floor(section_length / step).astype(int))
        else:
            interp_range = np.linspace(0, 1, np.floor(section_length / step).astype(int), endpoint=False)
        
        # Linear interpolation
        fx = interp1d([0.0, 1.0], wp_xs[i:i+2], kind=1)
        fy = interp1d([0.0, 1.0], wp_ys[i:i+2], kind=1)

        # Final points
        path_xs = np.append(path_xs, fx(interp_range))
        path_ys = np.append(path_ys, fy(interp_range))
    
    # To get the angle we 
    dx = np.append(0, np.diff(path_xs))
    dy = np.append(0, np.diff(path_ys))
    theta = np.arctan2(dy, dx)
    
    return np.vstack((path_xs, path_ys, theta))

def get_nn_idx(state, path):
    """
    Params:
        state: (x, y, yaw)
        path: (2, N)
    """
    # cartesian state
    c_state = state[:2]
    c_path = path[:2]

    # Calculate the distance between the current state and sample points along the path    
    dist = np.linalg.norm(np.expand_dims(c_state, 1) - c_path, axis=0)
    nn_idx = np.argmin(dist)

    # If the nn_idx corresponds to the last point in the path, return it.
    if nn_idx == c_path.shape[1] - 1:
        return nn_idx
    
    # Else we check which index is the correct point to return.
    
    # 1. Form the unit vector from current index point to next
    v = [
        path[0, nn_idx + 1] - path[0, nn_idx],
        path[1, nn_idx + 1] - path[1, nn_idx]
    ]
    v /= np.linalg.norm(v)

    # 2. Form vector from nn_idx to current state point
    d = c_path[:, nn_idx] - c_state
    
    # 3. A positive projection implies that the current state has not surpassed the nearest neighbor point.
    if np.dot(d, v) > 0:
        return nn_idx
    else:
        return nn_idx + 1
    
def rotation(theta):
    """ Creates rotation matrix with angle theta. """
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])

def f(x, coeff):
    """
    Helper function to compute value of function given j-nomial coefficients.

    Params:
        x: x-value to get value from
        coeff: j-nomial coefficients
    """
    y = 0
    j = len(coeff)
    for k in range(j):
        y += coeff[k] * x ** (j - k - 1)
    return y

def get_trajectory_coeffs_body(state, track, n_lookahead):
        """
        5-nomial interpolation for all points n_lookahead ahead of nearest neighbor index.
        Rotates all lookahead points to vehicular frame.

        Returns coefficients for 5-nomial.
        """
        # Get lookahead waypoints and rotate them to an offset in vehicular frame
        lk_wp = track[:, nn_idx : nn_idx + LOOKAHEAD]
        offset_s = lk_wp - np.expand_dims(state[0:2], 1)
        offset_b = rotation(-state[-1]) @ offset_s
        coeff = np.polyfit(
            offset_b[0, :],
            offset_b[1, :],
            5,
            rcond=None,
            full=False,
            w=None,
            cov=False,)
        return coeff, offset_b

def get_reference_trajectory(state, path, target_v, track_step, T, dt):
    """ 
    Given a target velocity, get a reference trajectory based on number of indices
    traversed along the precomputed path.

    Params:
        state: (n_states, 1): In our specific case (x, y, velocity, yaw)
        path: (3, N) full path that contains (x, y, theta).
        target_v: target velocity to generate reference trajectory with.
    Returns:
        xref: (n_states, T+1): T = horizon
        dref: all 0's because it's to be optimized for.
    """

    xref = np.zeros((state.shape[0], T + 1))
    dref = np.zeros((1, T + 1))

    path_length = path.shape[1]

    nn_idx = get_nn_idx(state, path)

    xref[0, 0] = path[0, nn_idx]
    xref[1, 0] = path[1, nn_idx]
    xref[2, 0] = target_v
    xref[2, 0] = path[2, nn_idx]
    
    # The track is formed with a step parameter dictating distance between each waypoint.
    dl = track_step
    travel = 0.0

    for i in range(T + 1):
        travel += abs(target_v) * dt
        n_indices = int(round(travel / dl))

        if (nn_idx + n_indices) < path_length:
            xref[0, i] = path[0, nn_idx + n_indices]
            xref[1, i] = path[1, nn_idx + n_indices]
            xref[2, i] = target_v
            xref[2, i] = path[2, nn_idx + n_indices]
        else:
            xref[0, i] = path[0, path_length - 1]
            xref[1, i] = path[1, path_length - 1]
            xref[2, i] = 0.0
            xref[2, i] = path[2, path_length - 1]

    return xref, dref

def get_reference_trajectory_no_accel(state, path, target_v, track_step, T, dt):
    """ 
    Given a target velocity, get a reference trajectory based on number of indices
    traversed along the precomputed path.

    Params:
        state: (n_states, 1): In our specific case (x, y, velocity, yaw) OR (x, y, yaw)
        path: (3, N) full path that contains (x, y, theta).
        target_v: target velocity to generate reference trajectory with.
        track_step: distance between subsequent points on the reference track.
        T: time horizon of MPC (also number of points we need to collect from the track for reference)
        dt: timestep used in MPC
    Returns:
        xref: (n_states, T+1): T = horizon
        dref: all 0's because it's to be optimized for.
    """
    xref = np.zeros((3, T + 1))
    dref = np.zeros((1, T + 1))

    path_length = path.shape[1]

    nn_idx = get_nn_idx(state, path)

    xref[0, 0] = path[0, nn_idx]
    xref[1, 0] = path[1, nn_idx]
    xref[2, 0] = path[2, nn_idx]
    
    # The track is formed with a step parameter dictating distance between each waypoint.
    dl = track_step
    travel = 0.0

    for i in range(T + 1):
        travel += abs(target_v) * dt
        n_indices = int(round(travel / dl))

        if (nn_idx + n_indices) < path_length:
            xref[0, i] = path[0, nn_idx + n_indices]
            xref[1, i] = path[1, nn_idx + n_indices]
            xref[2, i] = path[2, nn_idx + n_indices]
        else:
            xref[0, i] = path[0, path_length - 1]
            xref[1, i] = path[1, path_length - 1]
            xref[2, i] = path[2, path_length - 1]

    return xref, dref

if __name__ == "__main__":
    # Horizon
    LOOKAHEAD = 6

    # Sample waypoints to generate a track from.
    waypoints = np.array([[0, 0], [5, 0], [6, 2], [10, 2], [11, 0], [15, 4]]).T
    track = generate_path_from_wp(waypoints[0, :], waypoints[1, :], step=0.5)
    
    # Sample vehicle state
    state = [3.5, 0.5, np.radians(30)]

    # Find closest index.
    nn_idx = (
        get_nn_idx(state, track[:2, :]) - 1
    )  # index ox closest wp, take the previous to have a straighter line
    
    coeff, offset_b = get_trajectory_coeffs_body(state, track[:2, :], LOOKAHEAD)
    
    x = np.arange(-1, 2, 0.001)  # interp range of curve

    # Visualizing trajectory coefficients in body frame.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.set_title("Lookahead points with trajectory fitted in body frame")
    ax1.scatter(0, 0)
    ax1.scatter(offset_b[0, :], offset_b[1, :])
    ax1.plot(x, [f(xs, coeff) for xs in x])
    ax1.axis("equal")

    # Visualizing all waypoints in global frame.
    ax2.set_title("Waypoints in global frame")
    ax2.scatter(state[0], state[1], c='r')
    ax2.scatter(track[0, :], track[1, :])
    ax2.scatter(track[0, nn_idx:nn_idx+LOOKAHEAD], track[1, nn_idx:nn_idx+LOOKAHEAD])
    ax2.axis("equal")

    # plt.show()

