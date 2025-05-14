import numpy as np
from scipy.interpolate import interp1d

class TrajProc:
    """
    Generates trajectories from trajectories, finds nearest neighbors.
    """
    def __init__(self):
        self.lastIndex = -1

    def generate_path_from_wp(self, wp_xs, wp_ys, step=0.1):
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

    def get_nn_idx(self, state, path):
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
        if np.dot(d, v) <= 0:
            nn_idx += 1
    
        return nn_idx

    def get_reference_trajectory(self, state, path, target_v, track_step, T, dt):
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

        nn_idx = self.get_nn_idx(state, path)

        xref[0, 0] = path[0, nn_idx]
        xref[1, 0] = path[1, nn_idx]
        xref[2, 0] = target_v
        xref[3, 0] = path[2, nn_idx]
        
        # The track is formed with a step parameter dictating distance between each waypoint.
        dl = track_step
        travel = 0.0

        # Populate reference trajectory. Since reference trajectory
        for i in range(T + 1):
            travel += abs(target_v) * dt
            n_indices = int(round(travel / dl))

            if (nn_idx + n_indices) < path_length:
                xref[0, i] = path[0, nn_idx + n_indices]
                xref[1, i] = path[1, nn_idx + n_indices]
                xref[2, i] = target_v
                xref[3, i] = path[2, nn_idx + n_indices]
            else:
                xref[0, i] = path[0, path_length - 1]
                xref[1, i] = path[1, path_length - 1]
                xref[2, i] = 0.0
                xref[3, i] = path[2, path_length - 1]

        xref[3, :] = np.unwrap(xref[3, :])

        return xref, dref

    def get_reference_trajectory_no_accel(self, state, path, target_v, track_step, T, dt):
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

        nn_idx = self.get_nn_idx(state, path)

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
