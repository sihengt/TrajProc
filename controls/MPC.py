import casadi as cs
import numpy as np
from scripts import *
from DKBM_casadi import csDSKBM

class MPC:
    def __init__(self, params, model):
        """
        Params:
            model: contains functions for integrate
            params: dictionary containing the following attributes:
                1. n_x           (x in R^{n_x})
                2. m_u           (u in R^{m_u})
                3. T            (Time horizon of MPC)
                4. DT           (T control interval)
                5. X_lb[list]   (lower bounds of x)
                6. X_ub[list]   (upper bounds of x)
                7. u_lb[list]   (lower bounds of control)
                8. u_ub[list]   (upper bounds of control)
        """
        self.model = model # model.integrate / model.forward_one_step
        self.params = params
        self.nX = params['n_x']
        self.mU = params['m_u']
        self.T = params['T']
        self.dt = params['dt']
        
        # Pack the decision variables in the following order:
        # [x_1; u_1; x_2; u_2; ... x_{T+1}]
        # Note the absence of the last control.
        # self.X = cs.MX.sym('W', self.nX * (self.T+1) + self.mU * self.T, 1)
        
        # Initializes and flattens lower and upper bounds for state for each state variable in w.
        self.X = cs.MX.sym('X', self.nX, self.T + 1)
        X_lb_row = params['X_lb']
        X_ub_row = params['X_ub']
        X_lb = cs.repmat(X_lb_row, self.T + 1, 1)
        X_ub = cs.repmat(X_ub_row, self.T + 1, 1)

        self.U = cs.MX.sym('U', self.mU, self.T)
        U_lb_row = params['U_lb']
        U_ub_row = params['U_ub']
        U_lb = cs.repmat(U_lb_row, self.T, 1)
        U_ub = cs.repmat(U_ub_row, self.T, 1)

        # Stacking decision variables and contraints
        vec = lambda M: cs.reshape(M, -1, 1)
        self.w = cs.vertcat(vec(self.X), vec(self.U))
        self.lbw = cs.vertcat(X_lb, U_lb)
        self.ubw = cs.vertcat(X_ub, U_ub)
        self.lbw = cs.vertcat(X_lb, U_lb)
        self.lbw = cs.vertcat(X_lb, U_lb)

        # When initialized, we do not have to re-define the problem at each iteration.
        self.S = None

    def rollout(self, x_bar, u_bar):
        """
        Rollout using linearized dynamics from model to populate x_bar.
        """
        for k in range (0, self.T):
            x_k = x_bar[:, k]
            u_k = u_bar[:, k]
            x_kp1, A_d, B_d, C_d = self.model.integrate(x_k, u_k)
            x_bar[:, k+1] = x_kp1.full().flatten()
        
        return x_bar

    def predict(self, x_bar_0, u_bar, track):
        """
        Solves MPC problem over a reference track.

        Params:
            x_bar_0
            u_bar: warm-start for actions
            track: numpy array representing track (x, y, theta)
        """
        
        # Initialization for solving MPC problem.
        g = []      # Constraint
        lbg = []    # Lower bound for constraints
        ubg = []    # Upper bound for constraints
        J = 0       # Cost

        # Initial guess of state. It doesn't have to be accurate but it helps to be dynamically accurate.
        x_bar = np.zeros((self.nX, self.T+1))
        x_bar[:, 0] = x_bar_0
        
        # Gets x_bar based on u_bar (warm-start for actions).
        self.rollout(x_bar, u_bar)

        # TODO: PARAMETERIZE!
        REF_VEL = 1.0

        x_ref, _ = get_reference_trajectory(x_bar[:, 0], track, REF_VEL, 0.05)
        x_ref[3, :] = np.unwrap(x_ref[3, :])
        
        for k in range(self.T):
            # Initialize action for current_timestep
            e = self.X[:, k] - x_ref[:, k]
            J += e.T @ self.params['Q'] @ e
            J += self.U[:, k].T @ self.params['R'] @ self.U[:, k]

            # Handling constraints to smooth controls
            if k < (self.T - 1):
                e_u = self.U[:, k+1] - self.U[:, k]
                J += e_u.T @ self.params['R_'] @ e_u

                d_u = e_u / DT
                g += [d_u]
                lbg.append(-1 * self.params['dU_b'])
                ubg.append(self.params['dU_b'])

            x_kp1, A, B, C = self.model.integrate(x_bar[:, k], u_bar[:, k])

            g += [A @ self.X[:, k] + B @ self.U[:, k] + C - self.X[:, k + 1]]
            lbg += [0] * self.nX
            ubg += [0] * self.nX

        # Terminal cost
        e_xn = self.X[:, self.T] - x_ref[:, self.T]
        J += e_xn.T @ self.params['Qf'] @ e_xn

        # Setting a constraint for initial state.
        g += [self.X[:, 0] - x_bar[:, 0]]
        lbg += [0] * self.nX
        ubg += [0] * self.nX

        qp_opts = {}
        prob    = {'x': self.w, 'f': J, 'g': cs.vertcat(*g)}
        solver  = cs.qpsol('solver', 'qpoases', prob, qp_opts)
        sol     = solver(lbx=self.lbw, ubx=self.ubw, lbg=cs.vertcat(*lbg), ubg=cs.vertcat(*ubg))

        X_sol = sol['x']
        # Take the evenly shaped portions first.
        # Reshape so every column corresponds to a time step
        X_mpc = cs.reshape(sol['x'][:self.nX * (self.T+1)], 4, self.T+1).full()
        U_mpc = cs.reshape(sol['x'][self.nX * (self.T+1):], 3, self.T).full()
        
        return X_mpc, U_mpc

    def build_g(self, x_ref):
        g = []
        for k in range(self.T):
            g_block = cs.vertcat(-2*self.params['Q'] @ cs.DM(x_ref[:,k]),
                                cs.DM.zeros(self.mU,1))
            g.append(g_block)
        g.append(cs.vertcat(-2*self.params['Qf'] @ cs.DM(x_ref[:,self.T])))
        return cs.vertcat(*g)

if __name__ == "__main__":
    N_STATES = 4
    N_ACTIONS = 3
    L = 0.3
    l_f = 0.1
    l_r = 0.2
    T = 10          # MPC horizon
    N = 50          # Control Interval
    DT = 0.2        # dt = T/N

    MAX_SPEED = 1.5
    MAX_STEER = np.radians(30)
    MAX_D_ACC = 1.0
    MAX_D_STEER = np.radians(30)  # rad/s
    MAX_ACC = 1.0
    REF_VEL = 1.0

    MPC_PARAMS = {
        "n_x": N_STATES,
        "m_u": N_ACTIONS,
        "T": T,
        "dt": 0.2,
        "X_lb": cs.DM([-cs.inf, -cs.inf, 0, -cs.inf],),
        "X_ub": cs.DM([cs.inf, cs.inf, MAX_SPEED, cs.inf],),
        "U_lb": cs.DM([-MAX_ACC, -MAX_STEER, -MAX_STEER]), 
        "U_ub": cs.DM([MAX_ACC, MAX_STEER, MAX_STEER]),
        "dU_b": cs.DM([MAX_D_ACC, MAX_D_STEER, MAX_D_STEER]),
        "Q": cs.DM(np.diag([20, 20, 10, 0])),
        "Qf": cs.DM(np.diag([30, 30, 30, 0])),
        "R": cs.DM(np.diag([10, 10, 10])),
        "R_": cs.DM(np.diag([10, 10, 10]))
    }

    track = generate_path_from_wp(
    [0, 3, 4, 6, 10, 12, 14, 6, 1, 0], [0, 0, 2, 4, 3, 3, -2, -6, -2, -2], 0.05
    )

    cs_kbm = csDSKBM(N_STATES, N_ACTIONS, L, l_f, l_r, T, N)
    mpc = MPC(MPC_PARAMS, cs_kbm)
    mpc.predict(np.array([1, 2, 1.0, 4]).T, np.random.rand(3, T), track)