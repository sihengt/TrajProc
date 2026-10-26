import numpy as np
from scipy.interpolate import interp1d

class TrajProc:
    """
    Generates trajectories from trajectories, finds nearest neighbors.
    """
    def __init__(self):
        self.lastIndex = 0

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

    def get_nn_idx_with_window(self, state, path, last_idx, window=10):
        """
        Index of the path point the car is at or just before, moving forward at most `window` indices
        past last_idx per call.

        Params:
            state: (x, y, ...)
            path: (2+, N)
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
        
        if nn_idx > last_idx + window:
            return last_idx + window
        
        return nn_idx

    @staticmethod
    def is_closed(path, path_step):
        """True if the path ends where it starts (a loop), within one path step."""
        return bool(np.linalg.norm(path[:2, 0] - path[:2, -1]) < path_step)

    def get_reference_trajectory(self, state, path, target_v, path_step, T, dt, closed=None):
        """ 
        Given a target velocity, get a reference trajectory based on number of indices
        traversed along the precomputed path.

        Point i of the reference (i = 0..T) is (i + 1) * target_v * dt ahead of the nearest path point.
        Near the end of the path:
            closed path: the reference continues into the next lap at target_v.
            open path:   the reference stops at the last point with velocity 0.

        Params:
            state: (n_states, 1): In our specific case (x, y, velocity, yaw)
            path: (3, N) full path that contains (x, y, theta).
            target_v: target velocity to generate reference trajectory with.
            path_step: spacing between each index in the path
            closed: whether the path is a loop. Default: detected with is_closed().
        Returns:
            xref: (n_states, T+1): T = horizon
            nn_idx: index of nearest neighbor on path
            ref_traj_idx: [nn_idx] + the path index of each reference point
        """
        if closed is None:
            closed = self.is_closed(path, path_step)

        xref = np.zeros((state.shape[0], T + 1))
        path_length = path.shape[1]

        nn_idx = self.get_nn_idx_with_window(state, path, self.lastIndex)
        ref_traj_idx = [nn_idx]

        # Updates TrajProc memory so we do not accidentally backtrack or skip forward too much.
        self.lastIndex = nn_idx

        travel = 0.0
        for i in range(T + 1):
            travel += abs(target_v) * dt
            idx = nn_idx + int(round(travel / path_step))
            v = target_v
            if idx >= path_length:
                if closed:
                    idx %= path_length
                else:
                    idx = path_length - 1
                    v = 0.0

            xref[0, i] = path[0, idx]
            xref[1, i] = path[1, idx]
            xref[2, i] = v
            xref[3, i] = path[2, idx]
            ref_traj_idx.append(idx)

        xref[3, :] = np.unwrap(xref[3, :])

        return xref, nn_idx, ref_traj_idx
