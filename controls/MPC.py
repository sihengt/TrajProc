import copy
from collections import namedtuple

import casadi as cs
import numpy as np

LinearizedDynamics = namedtuple('LinearizedDynamics', ['A', 'B', 'C'])

class MPC:
    def __init__(self, mpc_params, model):
        """
        Params:
            mpc_params: dict from MPCConfigLoader.construct_casadi_params(). Uses:
                model.nStates, model.nActions, T (horizon), dt,
                X_lb, X_ub, U_lb, U_ub (box bounds), dU_b (input rate bound),
                Q, Qf, R, R_ (cost weights on state error, terminal error, input, input change)
            model: provides integrate(x_bar, u_bar) -> (x_next, A_d, B_d, C_d)
        """
        self.model  = model
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
                        
            l_linearized_dynamics.append(LinearizedDynamics(A_d, B_d, C_d))            
            assert(np.linalg.norm((x_kp1 - (A_d @ x_k + B_d @ u_k + C_d)).full()) < 1e-3)
            x_bar[:, k+1] = x_kp1.full().flatten()
        
        return x_bar, l_linearized_dynamics

    def predict(self, x_bar_0, x_ref, u_bar, n_iters=1):
        """
        Solves the MPC problem by successive linearization: n_iters times, roll out u_bar through the
        model, build a QP with the linearized dynamics, and use its actions as the next u_bar.

        The model is restored to its state at the start of the call before each rollout, so models with
        history (ResDSKBM) see the same history in every iteration. After the call, self.model holds the
        history of the last rollout; callers that keep a separate copy should restore it.

        Params:
            x_bar_0: current state of the robot
            x_ref: (nX, T+1) reference trajectory
            u_bar: (mU, T) warm-start for actions
            n_iters: number of linearize-and-solve iterations

        Returns:
            X_mpc: optimized states
            U_mpc: optimized actions
            x_ref: reference trajectory used to get optimized states / actions.
        """
        
        current_model = copy.deepcopy(self.model)

        for i_mpc_iter in range(n_iters):
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

            # Simple convergence check
            # tol = 1e-3
            # delta_u = U_mpc - u_bar
            # if np.linalg.norm(delta_u, np.inf) < tol:
            #     print("Early convergence in {} iters.".format(i_mpc_iter))
            #     break

            u_bar = U_mpc
        
        return X_mpc, U_mpc, x_ref
