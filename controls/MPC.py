import copy
from collections import namedtuple

import casadi as cs
import numpy as np

LinearizedDynamics = namedtuple('LinearizedDynamics', ['A', 'B', 'C'])

ARMIJO_C = 1e-4     # fraction of the decrease predicted along the step that the cost must achieve
MIN_STEP = 1e-4     # smallest step length the line search tries

class MPC:
    def __init__(self, mpc_params, model):
        """
        Params:
            mpc_params: dict from MPCConfigLoader.construct_casadi_params(). Uses:
                model.nStates, model.nActions, T (horizon), dt,
                X_lb, X_ub, U_lb, U_ub (box bounds), dU_b (input rate bound),
                Q, Qf, R, R_ (cost weights on state error, terminal error, input, input change)
            model: provides integrate(x_bar, u_bar) -> (x_next, A_d, B_d, C_d), where A_d and B_d are the
                Jacobians of x_next w.r.t. x_bar and u_bar
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

        # Cost of a plan and its gradients, for the line search.
        X_ref = cs.MX.sym('X_ref', self.nX, self.T + 1)
        J = self.tracking_cost(self.X, self.U, X_ref)
        self.cost = cs.Function('cost', [self.X, self.U, X_ref], [J, cs.gradient(J, self.X), cs.gradient(J, self.U)])

    def tracking_cost(self, X, U, X_ref):
        """
        Quadratic cost on state error, terminal error, input and input change.
        """
        J = 0
        for k in range(self.T):
            e = X[:, k] - X_ref[:, k]
            J += e.T @ self.params['Q'] @ e
            J += U[:, k].T @ self.params['R'] @ U[:, k]
            if k < (self.T - 1):
                e_u = U[:, k + 1] - U[:, k]
                J += e_u.T @ self.params['R_'] @ e_u

        e_xn = X[:, self.T] - X_ref[:, self.T]
        J += e_xn.T @ self.params['Qf'] @ e_xn
        return J

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

    def rollout_from(self, start_model, x_bar_0, u_bar):
        """
        Rolls out u_bar from x_bar_0 with a fresh copy of start_model, so models with history (ResDSKBM) see
        the same history in every rollout. Leaves self.model with the history of this rollout.
        """
        self.model = copy.deepcopy(start_model)
        x_bar = np.zeros((self.nX, self.T + 1))
        x_bar[:, 0] = x_bar_0
        return self.rollout(x_bar, u_bar)

    def solve_qp(self, x_bar, u_bar, x_ref, l_linearized_dynamics):
        """
        Solves the QP with the dynamics linearized along (x_bar, u_bar).

        Returns:
            X_qp: the QP's states (the linear prediction of the rollout of U_qp)
            U_qp: the QP's actions
        """
        g = []      # Constraint
        lbg = []    # Lower bound for constraints
        ubg = []    # Upper bound for constraints
        J = self.tracking_cost(self.X, self.U, x_ref)

        for k in range(self.T):
            # Handling constraints to smooth controls
            if k < (self.T - 1):
                d_u = (self.U[:, k + 1] - self.U[:, k]) / self.dt
                g += [d_u]
                lbg.append(-1 * self.params['dU_b'])
                ubg.append(self.params['dU_b'])

            A, B, C = l_linearized_dynamics[k].A, l_linearized_dynamics[k].B, l_linearized_dynamics[k].C

            g += [A @ self.X[:, k] + B @ self.U[:, k] + C - self.X[:, k + 1]]
            lbg += [0] * self.nX
            ubg += [0] * self.nX

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

        # Reshape so every column corresponds to a time step
        X_qp = cs.reshape(sol['x'][:self.nX * (self.T+1)], self.nX, self.T+1).full()
        U_qp = cs.reshape(sol['x'][self.nX * (self.T+1):], self.mU, self.T).full()
        return X_qp, U_qp

    def predict(self, x_bar_0, x_ref, u_bar, n_iters=1, tol=1e-4):
        """
        Solves the MPC problem by sequential quadratic programming (SQP) over the actions: roll out u_bar
        through the model, build a QP with the dynamics linearized along the rollout, and move u_bar towards
        the QP's actions with a backtracking line search on the cost of the rolled-out plan. Every iterate is
        a rollout, so the dynamics always hold; the line search keeps the bounds and rate limits because u_bar
        and the QP's actions both satisfy them. Stops when the QP step is at most tol in every action.

        After the call, self.stats holds:
            status: 'converged' (QP step <= tol), 'max_iterations', or 'no_decrease' (no step length lowered
                the cost enough, e.g. because the model's Jacobians are not exact)
            iterations: number of QPs solved
            cost: cost of the rolled-out plan for u_bar and after each accepted step

        The model is restored to its state at the start of the call before each rollout, so models with
        history (ResDSKBM) see the same history in every rollout. After the call, self.model holds the
        history of the rollout of the returned actions; callers that keep a separate copy should restore it.

        Params:
            x_bar_0: current state of the robot
            x_ref: (nX, T+1) reference trajectory
            u_bar: (mU, T) warm-start for actions; must satisfy the input bounds and rate limits
            n_iters: maximum number of SQP iterations
            tol: largest QP step (in any action) at which the plan counts as converged

        Returns:
            X_mpc: states of the rollout of U_mpc
            U_mpc: optimized actions
            x_ref: reference trajectory used to get optimized states / actions.
        """
        
        current_model = copy.deepcopy(self.model)
        u_bar = np.array(u_bar, dtype=float)
        x_bar, l_linearized_dynamics = self.rollout_from(current_model, x_bar_0, u_bar)
        rollout_model = self.model
        J, dJ_dX, dJ_dU = (np.asarray(v.full()) for v in self.cost(x_bar, u_bar, x_ref))
        self.stats = {'status': 'max_iterations', 'iterations': 0, 'cost': [float(J)]}

        for i_mpc_iter in range(n_iters):
            X_qp, U_qp = self.solve_qp(x_bar, u_bar, x_ref, l_linearized_dynamics)
            self.stats['iterations'] += 1
            d = U_qp - u_bar
            if np.max(np.abs(d)) <= tol:
                self.stats['status'] = 'converged'
                break
            
            # Slope of the rolled-out cost along d. X_qp - x_bar is the linear prediction of how the states change.
            slope = np.sum(dJ_dX * (X_qp - x_bar)) + np.sum(dJ_dU * d)

            alpha = 1.0
            while alpha >= MIN_STEP:
                u_try = u_bar + alpha * d
                x_try, lin_try = self.rollout_from(current_model, x_bar_0, u_try)
                J_try, dJ_dX_try, dJ_dU_try = (np.asarray(v.full()) for v in self.cost(x_try, u_try, x_ref))
                if J_try <= J + ARMIJO_C * alpha * slope:
                    break
                alpha /= 2
            else:
                self.stats['status'] = 'no_decrease'
                break

            u_bar, x_bar, l_linearized_dynamics = u_try, x_try, lin_try
            J, dJ_dX, dJ_dU = J_try, dJ_dX_try, dJ_dU_try
            rollout_model = self.model
            self.stats['cost'].append(float(J))

        self.model = rollout_model
        return x_bar, u_bar, x_ref
