import casadi as cs


class csDSKBM:
    """
    Double Steer Kinematic Bicycle Model (DSKBM) in CasADi.

    State:  X = [x, y, v, yaw]       position and speed of the centre of gravity (CG), heading
    Action: U = [a, delta_f, delta_r] acceleration, front and rear steering angle

    Assumes:
        1. Velocity of each wheel points in the direction the wheel is facing (i.e. steering angle)
        2. omega = velocity / turning radius
        3. Tracks position and velocity of CG.

    integrate() linearizes the model at (X_bar, U_bar), integrates the linear model for one dt, and returns the
    next state together with the discrete affine model X_{k+1} = A_d X_k + B_d U_k + C_d that the MPC uses as a
    constraint. A_d and B_d are the exact Jacobians of that next state w.r.t. (X_bar, U_bar).
    """
    N_STATES = 4
    N_ACTIONS = 3

    def __init__(self, mpc_params):
        """
        Params:
            mpc_params: [dict] MPC parameters. Uses mpc_params['model'] (nStates, nActions, l_f, l_r),
                mpc_params['T'] (horizon) and mpc_params['dt'] (time step).
        """
        model_params = mpc_params['model']
        assert model_params['nStates'] == self.N_STATES
        assert model_params['nActions'] == self.N_ACTIONS
        self.l_f = model_params['l_f']  # CG to front axle
        self.l_r = model_params['l_r']  # CG to rear axle
        self.T = mpc_params['T']        # Time horizon
        self.dt = mpc_params['dt']      # dt

        self.X = cs.MX.sym('X', self.N_STATES)
        self.U = cs.MX.sym('U', self.N_ACTIONS)
        X_dot = self.x_dot_expression(self.X, self.U)

        self.f_x_dot = cs.Function('X_dot', [self.X, self.U], [X_dot], ['X', 'U'], ['X_dot'])
        self.get_A = cs.Function('get_A', [self.X, self.U], [cs.jacobian(X_dot, self.X)])
        self.get_B = cs.Function('get_B', [self.X, self.U], [cs.jacobian(X_dot, self.U)])
        self.RK4 = self.init_rk4()
        self.f_step = self.init_step()

        self.dae = {'x': self.X, 'p': self.U, 'ode': X_dot}
        self._integrator = None

    def sideslip(self, delta_f, delta_r):
        return cs.arctan((self.l_r * cs.tan(delta_f) + self.l_f * cs.tan(delta_r)) / (self.l_r + self.l_f))

    def x_dot_expression(self, X, U):
        v, theta = X[2], X[3]
        a, delta_f, delta_r = U[0], U[1], U[2]
        beta = self.sideslip(delta_f, delta_r)

        x_dot = v * cs.cos(beta + theta)
        y_dot = v * cs.sin(beta + theta)
        v_dot = a
        theta_dot = (v * cs.cos(beta)) / (self.l_f + self.l_r) * (cs.tan(delta_f) - cs.tan(delta_r))
        return cs.vertcat(x_dot, y_dot, v_dot, theta_dot)

    def linearize(self, X_bar, U_bar):
        """
        Returns A, B, C of the continuous affine model X_dot = A X + B U + C around (X_bar, U_bar).
        """
        A = self.get_A(X_bar, U_bar)
        B = self.get_B(X_bar, U_bar)
        f_bar = self.f_x_dot(X=X_bar, U=U_bar)['X_dot']
        C = f_bar - A @ X_bar - B @ U_bar
        return A, B, C

    def init_step(self):
        """
        Next state of one dt from X with the model linearized at (X, U) and a constant c added to X_dot, and its
        Jacobians w.r.t. X and U (c held fixed). The Jacobians include how A, B and C change with (X, U).
        """
        c = cs.MX.sym('c', self.N_STATES)
        A, B, C = self.linearize(self.X, self.U)
        x_next = self.RK4(self.X, self.U, A, B, C + c)
        return cs.Function('step', [self.X, self.U, c],
                           [x_next, cs.jacobian(x_next, self.X), cs.jacobian(x_next, self.U)],
                           ['X', 'U', 'c'], ['x_next', 'A_d', 'B_d'])

    def step(self, X_bar, U_bar, c=0):
        """
        Integrates the model linearized at (X_bar, U_bar), with c added to X_dot, from X_bar for one dt (RK4).

        Returns:
            x_kp1, A_d, B_d, C_d: next state and the affine model A_d X + B_d U + C_d tangent to it at (X_bar, U_bar)
        """
        x_kp1, A_d, B_d = self.f_step(X_bar, U_bar, c)
        return x_kp1, A_d, B_d, x_kp1 - A_d @ X_bar - B_d @ U_bar

    def integrate(self, X_bar, U_bar):
        return self.step(X_bar, U_bar)

    def init_rk4(self):
        x0 = cs.MX.sym("x0_int", self.N_STATES)
        u = cs.MX.sym("u_int", self.N_ACTIONS)
        A = cs.MX.sym("A", self.N_STATES, self.N_STATES)
        B = cs.MX.sym("B", self.N_STATES, self.N_ACTIONS)
        C = cs.MX.sym("C", self.N_STATES)

        k1 = A @ x0 + B @ u + C
        k2 = A @ (x0 + self.dt / 2 * k1) + B @ u + C
        k3 = A @ (x0 + self.dt / 2 * k2) + B @ u + C
        k4 = A @ (x0 + self.dt * k3) + B @ u + C
        x_next = x0 + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        return cs.Function('F', [x0, u, A, B, C], [x_next], ['x0', 'u', 'A', 'B', 'C'], ['xf'])

    def forward_one_step(self, x0, u):
        """
        Integrates the nonlinear model by one dt with IDAS (not used by the MPC; useful as a reference).

        Params:
            x0: state to integrate for
            u: control input
        """
        if self._integrator is None:
            self._integrator = cs.integrator('F', 'idas', self.dae, {'tf': self.dt})
        r = self._integrator(x0=x0, p=u)
        return r['xf'].full().flatten()
