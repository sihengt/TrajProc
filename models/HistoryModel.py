import casadi as cs


class HistoryModel:
    """
    Discrete model whose state also holds the last n_past steps of selected robot states and all actions.

    acados needs dynamics of the form z_{k+1} = F(z_k, u_k). A model that reads a window of past steps (for
    example a network on a history window), or a cost on the change of action, does not have that form in the
    robot state x alone. Carrying the window in the state gives it that form:

        z = [x, h],  h = [e_{k-n_past}, ..., e_{k-1}],  e_j = [x_j[state_idx], u_j]

    Each step drops the oldest entry and adds the current one, e_k. If a residual is given, it reads the
    window [e_{k-n_past}, ..., e_k] (one row per step, oldest first) and its output is added to the robot's
    x_dot, as for the CRN.
    """

    def __init__(self, robot, n_past, state_idx=(), residual=None):
        """
        Params:
            robot: provides N_STATES, N_ACTIONS and next_state(x, u, c): the symbolic next state with c added
                to x_dot
            n_past: number of past steps kept in the state
            state_idx: indices of the robot states kept in the window (all actions are always kept)
            residual: optional CasADi function from the (n_past + 1, n_features) window to an (N_STATES,)
                correction of x_dot
        """
        self.nx = robot.N_STATES
        self.nu = robot.N_ACTIONS
        self.n_past = n_past
        self.state_idx = list(state_idx)
        self.n_features = len(self.state_idx) + self.nu
        self.nz = self.nx + n_past * self.n_features

        self.z = cs.SX.sym('z', self.nz)
        self.u = cs.SX.sym('u', self.nu)
        x, past = self.unpack(self.z)
        window = cs.vertcat(past, cs.horzcat(x[self.state_idx].T, self.u.T))
        c = residual(window) if residual is not None else cs.SX.zeros(self.nx)
        self.z_next = self.pack(robot.next_state(x, self.u, c), window[1:, :])
        self.f = cs.Function('F', [self.z, self.u], [self.z_next], ['z', 'u'], ['z_next'])

    def pack(self, x, past):
        """State z from the robot state x and the (n_past, n_features) past entries, oldest first."""
        return cs.vertcat(x, cs.vec(past.T))

    def unpack(self, z):
        """Robot state x and the (n_past, n_features) past entries, oldest first, from the state z."""
        return z[:self.nx], cs.reshape(z[self.nx:], self.n_features, self.n_past).T
