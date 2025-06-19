import casadi as cs
import numpy as np
from collections import namedtuple

LinearizedDynamics = namedtuple('LinearizedDynamics', ['A', 'B', 'C'])

class MPC:
    def __init__(self, mpc_params, model):
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
        self.model  = model # model.integrate / model.forward_one_step
        self.hasLSTM = getattr(self.model, "hasLSTM", False)
        self.params = mpc_params
        self.nX     = mpc_params['model']['nStates']
        self.mU     = mpc_params['model']['nActions']
        self.T      = mpc_params['T']
        self.dt     = mpc_params['dt']
        
        # Initializes and flattens lower and upper bounds for state for each state variable in w.
        self.X = cs.MX.sym('X', self.nX, self.T + 1)
        X_lb_row = mpc_params['X_lb']
        X_ub_row = mpc_params['X_ub']
        X_lb = cs.repmat(X_lb_row, self.T + 1, 1)
        X_ub = cs.repmat(X_ub_row, self.T + 1, 1)

        self.U = cs.MX.sym('U', self.mU, self.T)
        U_lb_row = mpc_params['U_lb']
        U_ub_row = mpc_params['U_ub']
        U_lb = cs.repmat(U_lb_row, self.T, 1)
        U_ub = cs.repmat(U_ub_row, self.T, 1)

        # Stacking decision variables and contraints
        self.vec = lambda M: cs.reshape(M, -1, 1)
        self.w = cs.vertcat(self.vec(self.X), self.vec(self.U))
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
        l_linearized_dynamics = []
        
        # xd_km1 = np.zeros(4)
        
        for k in range (0, self.T):
            x_k = x_bar[:, k]
            u_k = u_bar[:, k]
            
            x_kp1, A_d, B_d, C_d = self.model.integrate(x_k, u_k)
            
            # x_kp1, A_d, B_d, C_d = self.model.integrate(x_k, u_k, xd_km1)
            # xd_km1 = self.model.f_x_dot(x_k, u_k).full().flatten()
            
            l_linearized_dynamics.append(LinearizedDynamics(A_d, B_d, C_d))            
            assert(np.linalg.norm((x_kp1 - (A_d @ x_k + B_d @ u_k + C_d)).full()) < 1e-3)
            x_bar[:, k+1] = x_kp1.full().flatten()
        
        return x_bar, l_linearized_dynamics

    def predict(self, x_bar_0, x_ref, u_bar, n_iters=1):
        """
        Solves MPC problem over a reference track.

        Params:
            x_bar_0: current position of robot = first position of next robot
            x_ref: reference x positions for MPC
            u_bar: warm-start for actions
            
            

        Returns:
            X_mpc: optimized states
            U_mpc: optimized actions
            x_ref: reference trajectory used to get optimized states / actions.
        """
        
        # Make a copy of self.model here.
        # Reinitialize to copy of self.model everytime.
        import copy
        current_model = copy.deepcopy(self.model)

        for _ in range(n_iters):
            # Initial guess of state. It doesn't have to be accurate but it helps to be dynamically accurate.
            x_bar = np.zeros((self.nX, self.T+1))
            x_bar[:, 0] = x_bar_0
            
            # Initialization for solving MPC problem.
            g = []      # Constraint
            lbg = []    # Lower bound for constraints
            ubg = []    # Upper bound for constraints
            J = 0       # Cost

            self.model = copy.deepcopy(current_model)

            # Gets x_bar based on u_bar (warm-start for actions).
            _, l_linearized_dynamics = self.rollout(x_bar, u_bar)
            assert(len(l_linearized_dynamics) == self.T)

            for k in range(self.T):
                # Initialize action for current_timestep
                e = self.X[:, k] - x_ref[:, k]
                J += e.T @ self.params['Q'] @ e
                J += self.U[:, k].T @ self.params['R'] @ self.U[:, k]

                # Handling constraints to smooth controls
                if k < (self.T - 1):
                    e_u = self.U[:, k + 1] - self.U[:, k]
                    J += e_u.T @ self.params['R_'] @ e_u

                    d_u = e_u / self.dt
                    g += [d_u]
                    lbg.append(-1 * self.params['dU_b'])
                    ubg.append(self.params['dU_b'])

                # x_kp1, A, B, C = self.model.integrate(x_bar[:, k], u_bar[:, k])
                A, B, C = l_linearized_dynamics[k].A, l_linearized_dynamics[k].B, l_linearized_dynamics[k].C

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

            # g += [self.U[:, 0] - u_bar[:, 0]]
            # lbg.append(-1 * self.params['dU_b'])
            # ubg.append(self.params['dU_b'])

            qp_opts = {"error_on_fail": True, "print_problem": False, "verbose":False, "printLevel": "none"}
            assert cs.vertcat(*g).shape == cs.vertcat(*lbg).shape
            assert cs.vertcat(*g).shape == cs.vertcat(*ubg).shape
            prob    = {'x': self.w, 'f': J, 'g': cs.vertcat(*g)}
            solver  = cs.qpsol('solver', 'qpoases', prob, qp_opts)
            sol     = solver(lbx=self.lbw, ubx=self.ubw, lbg=cs.vertcat(*lbg), ubg=cs.vertcat(*ubg),
                             x0=cs.vertcat(self.vec(cs.DM(x_bar)), self.vec(cs.DM(u_bar)))
                            )

            X_sol = sol['x']
            # Take the evenly shaped portions first.
            # Reshape so every column corresponds to a time step
            X_mpc = cs.reshape(sol['x'][:self.nX * (self.T+1)], self.nX, self.T+1).full()
            U_mpc = cs.reshape(sol['x'][self.nX * (self.T+1):], self.mU, self.T).full()

            u_bar = U_mpc
        
        return X_mpc, U_mpc, x_ref

if __name__ == "__main__":
    from ..models.DSKBM import csDSKBM
    from ..TrajProc import TrajProc
    
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

    tp = TrajProc()

    track = tp.generate_path_from_wp(
    [0, 3, 4, 6, 10, 12, 14, 6, 1, 0], [0, 0, 2, 4, 3, 3, -2, -6, -2, -2], 0.05
    )

    cs_kbm = csDSKBM(L, l_f, l_r, T, N)
    mpc = MPC(MPC_PARAMS, cs_kbm, tp)
    mpc.predict(np.array([1, 2, 0.8, np.radians(10)]).T, np.random.rand(3, T), REF_VEL, track)