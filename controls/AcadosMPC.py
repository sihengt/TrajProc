import os
import time

import casadi as cs
import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from scipy.linalg import block_diag

MAX_ITERATIONS = 100    # largest n_iters; acados sizes its memory for this when the solver is built
STATUS = {0: 'converged', 2: 'max_iterations', 3: 'min_step'}


class AcadosMPC:
    def __init__(self, mpc_params, model, build_dir, tol=1e-6, name='mpc'):
        """
        MPC solved by acados (SQP, Gauss-Newton Hessian, line search on a merit function): the same problem as
        MPC (tracking cost, input bounds, input rate cost and limit) with the dynamics of a HistoryModel.

        The rate cost and limit read the previous action from the model's window, so the model keeps at least one
        past step. As in MPC, the first action of the plan has no rate cost or limit.

        Params:
            mpc_params: dict with T, dt, Q, Qf, R, R_, U_lb, U_ub, dU_b (as from MPCConfigLoader)
            model: HistoryModel with n_past >= 1
            build_dir: directory for the generated C code of the solver (built here, takes seconds)
            tol: tolerance on the KKT conditions at which the plan counts as converged
            name: name of the generated solver (one per model in the same build_dir)
        """
        assert model.n_past >= 1, "the rate cost and limit need the previous action in the window"
        self.params = mpc_params
        self.model = model
        self.T = mpc_params['T']
        nx, nu = model.nx, model.nu
        x, past = model.unpack(model.z)
        du = model.u - past[-1, model.n_features - nu:].T

        ocp = AcadosOcp()
        ocp.model = AcadosModel()
        ocp.model.name = name
        ocp.model.x = model.z
        ocp.model.u = model.u
        ocp.model.disc_dyn_expr = model.z_next

        # acados minimizes 0.5 ||y - y_ref||^2_W at each stage, hence the factor 2.
        Q, Qf, R, R_ = (np.asarray(cs.DM(mpc_params[k])) for k in ('Q', 'Qf', 'R', 'R_'))
        ocp.cost.cost_type_0 = 'NONLINEAR_LS'
        ocp.model.cost_y_expr_0 = cs.vertcat(x, model.u)
        ocp.cost.W_0 = 2 * block_diag(Q, R)
        ocp.cost.yref_0 = np.zeros(nx + nu)
        ocp.cost.cost_type = 'NONLINEAR_LS'
        ocp.model.cost_y_expr = cs.vertcat(x, model.u, du)
        ocp.cost.W = 2 * block_diag(Q, R, R_)
        ocp.cost.yref = np.zeros(nx + 2 * nu)
        ocp.cost.cost_type_e = 'NONLINEAR_LS'
        ocp.model.cost_y_expr_e = x
        ocp.cost.W_e = 2 * Qf
        ocp.cost.yref_e = np.zeros(nx)

        ocp.constraints.x0 = np.zeros(model.nz)
        ocp.constraints.idxbu = np.arange(nu)
        ocp.constraints.lbu = np.asarray(cs.DM(mpc_params['U_lb'])).ravel()
        ocp.constraints.ubu = np.asarray(cs.DM(mpc_params['U_ub'])).ravel()
        ocp.model.con_h_expr = du / mpc_params['dt']    # stages 1..T-1; stage 0 has its own (empty) set
        ocp.constraints.lh = -np.asarray(cs.DM(mpc_params['dU_b'])).ravel()
        ocp.constraints.uh = np.asarray(cs.DM(mpc_params['dU_b'])).ravel()

        opts = ocp.solver_options
        opts.N_horizon = self.T
        opts.tf = self.T * mpc_params['dt']
        opts.integrator_type = 'DISCRETE'
        opts.cost_scaling = np.ones(self.T + 1)     # the default scales stage costs (not the terminal one) by dt
        opts.hessian_approx = 'GAUSS_NEWTON'
        opts.nlp_solver_type = 'SQP'
        opts.globalization = 'MERIT_BACKTRACKING'
        opts.nlp_solver_max_iter = MAX_ITERATIONS
        opts.tol = tol
        opts.print_level = 0
        ocp.code_export_directory = os.path.join(build_dir, 'c_generated_code')
        self.solver = AcadosOcpSolver(ocp, json_file=os.path.join(build_dir, f'{name}.json'), verbose=False)

    def rollout(self, z0, U):
        Z = [z0]
        for u in U.T:
            Z.append(np.asarray(self.model.f(Z[-1], u)).ravel())
        return np.column_stack(Z)

    def predict(self, x_bar_0, x_ref, u_bar, n_iters=10, past=None):
        """
        Solves the MPC problem from the warm start u_bar.

        After the call, self.stats holds:
            status: 'converged', 'max_iterations', 'min_step' (the line search found no acceptable step),
                or 'failed (<acados status>)'
            iterations: number of SQP iterations
            time: wall time of the call [s]

        Params:
            x_bar_0: current state of the robot
            x_ref: (nx, T+1) reference trajectory
            u_bar: (nu, T) warm start for actions
            n_iters: maximum number of SQP iterations (at most MAX_ITERATIONS)
            past: (n_past, n_features) window entries before the current step, oldest first (default: zeros)

        Returns:
            X_mpc: robot states of the rollout of U_mpc
            U_mpc: optimized actions
            x_ref: reference trajectory used
        """
        t_start = time.perf_counter()
        model, nu = self.model, self.model.nu
        if past is None:
            past = np.zeros((model.n_past, model.n_features))
        z0 = np.asarray(model.pack(np.asarray(x_bar_0, dtype=float), np.asarray(past, dtype=float))).ravel()
        u_bar = np.asarray(u_bar, dtype=float).reshape(nu, self.T)
        Z_bar = self.rollout(z0, u_bar)

        self.solver.reset()
        self.solver.constraints_set(0, 'lbx', z0)
        self.solver.constraints_set(0, 'ubx', z0)
        for k in range(self.T):
            self.solver.set(k, 'x', Z_bar[:, k])
            self.solver.set(k, 'u', u_bar[:, k])
            n_zeros = nu if k == 0 else 2 * nu
            self.solver.set(k, 'yref', np.concatenate([x_ref[:, k], np.zeros(n_zeros)]))
        self.solver.set(self.T, 'x', Z_bar[:, self.T])
        self.solver.set(self.T, 'yref', np.asarray(x_ref[:, self.T], dtype=float))
        self.solver.options_set('max_iter', n_iters)

        status = self.solver.solve()
        U = np.column_stack([self.solver.get(k, 'u') for k in range(self.T)])
        self.stats = {'status': STATUS.get(status, f'failed ({status})'),
                      'iterations': int(self.solver.get_stats('sqp_iter'))}
        X = self.rollout(z0, U)[:model.nx]
        self.stats['time'] = time.perf_counter() - t_start
        return X, U, x_ref
