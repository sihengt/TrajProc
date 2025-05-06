from DKBM import DSKinematicBicycleModel
from DKBM_casadi import csDSKBM
from scripts import *

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import time
import casadi as cs
 
N_STATES = 4
N_ACTIONS = 3
L = 0.3
l_f = 0.1
l_r = 0.2
T = 2          # MPC horizon
N = 50          # Control Interval
DT = 0.2        # dt = T/N

MAX_SPEED = 1.5
MAX_STEER = np.radians(30)
MAX_D_ACC = 1.0
MAX_D_STEER = np.radians(30)  # rad/s
MAX_ACC = 1.0
REF_VEL = 1.0

cs_kbm = csDSKBM(N_STATES, N_ACTIONS, L, l_f, l_r, T, N)
kbm = DSKinematicBicycleModel(N_STATES, N_ACTIONS, L, l_f, l_r, T, DT)

# Step 1: Create a sample track
track = generate_path_from_wp(
    [0, 3, 4, 6, 10, 12, 14, 6, 1, 0], [0, 0, 2, 4, 3, 3, -2, -6, -2, -2], 0.05
)

sim_duration = 200  # time steps
opt_time = []

# VARIABLES FOR TRACKING
x_sim = np.zeros((N_STATES, sim_duration))
u_sim = np.zeros((N_ACTIONS, sim_duration - 1))

# Step 2: Create starting conditions x0
x_sim[:, 0] = np.array([0.0, -0.25, 0.0, np.radians(0)]).T

# Step 3: Generate starting guess for u_bar (does not have to be too accurate I suppose.)
u_bar_start = np.zeros((N_ACTIONS, T))
u_bar_start[0, :] = MAX_ACC / 2
u_bar_start[1, :] = 0.0
u_bar_start[2, :] = 0.0

l_a = []
l_df = []
l_dr = []
l_state = []

# In my case 
# 1/2 x^T H x 
# x would be the decision variables

MPC_PARAMS = {
    "n_x": N_STATES,
    "m_u": N_ACTIONS,
    "T": T,
    "dt": 0.2,
    "X_lb": cs.DM([-cs.inf, -cs.inf, 0, -cs.inf],),
    "X_ub": cs.DM([cs.inf, cs.inf, MAX_SPEED, cs.inf],),
    "U_lb": cs.DM([-MAX_ACC, -MAX_STEER, -MAX_STEER]), 
    "U_ub": cs.DM([MAX_ACC, MAX_STEER, MAX_STEER]),
    "Q": cs.DM(np.diag([20, 20, 10, 0])),
    "Qf": cs.DM(np.diag([30, 30, 30, 0])),
    "R": cs.DM(np.diag([10, 10, 10])),
    "R_": cs.DM(np.diag([10, 10, 10]))
}

class MPC:
    def __init__(self, params, model):
        """
        Params:
            model: contains functions for integrate
            params: dictionary containing the following attributes:
                1. N_X          (X \in R^{N_X})
                2. M_U          (U \in R^{M_U})
                3. T            (Time horizon of MPC)
                4. DT           (T_HORIZON / control interval)
                5. X_lb[list]   ()
                6. X_ub[list]   ()
                7. U_lb[list]   ()
                8. U_ub[list]   ()
        """
        self.model = model # model.integrate / model.forward_one_step
        self.params = params
        self.nX = params['n_x']
        self.mU = params['m_u']
        self.T = params['T']
        self.dt = params['dt']

        # Helper function to flatten casadi MX
        self.vec = lambda M: cs.reshape(M, -1, 1)
        
        # TODO: with the high level interface we did not really have to keep track of
        # variable order so we can just stack things. Now we do.

        # Pack the decision variables in the following order:
        # [x_1; u_1; x_2; u_2; ... x_{T+1}]
        # Note the absence of the last control.
        self.X = cs.MX.sym('W', self.nX * (T+1) + self.mU * T, 1)

        # self.X = cs.MX.sym('X', self.nX, T + 1)
        # self.U = cs.MX.sym('U', self.mU, T)
        # self.w = cs.vertcat(self.vec(self.X), self.vec(self.U))
        
        
        # Flattens and stacks all decision variables
        # self.W = cs.vertcat(self.vec(self.X), self.vec(self.U))

        # Initializes and flattens lower and upper bounds for state for each state variable in w.
        # X_lb = cs.repmat(params['X_lb'], T + 1, 1)
        # X_ub = cs.repmat(params['X_ub'], T + 1, 1)
        # U_lb = cs.repmat(params['U_lb'], T, 1)
        # U_ub = cs.repmat(params['U_ub'], T, 1)

        # Constraints        
        # self.lbw = cs.vertcat(X_lb, U_lb)
        # self.ubw = cs.vertcat(X_ub, U_ub)

        # Hessian
        self.H = self.init_H()

        self.qp = None
        self.S = None

    def init_H(self):
        """
        Initializes Hessian of the MPC problem. This will be constant throughout the problem
        """

        J = 0
        Q, Qf, R, R_ = self.params['Q'], self.params['Qf'], self.params['R'], self.params['R_']
        block_size = self.nX + self.mU
        hessian = cs.SX.zeros(self.X.size1(), self.X.size1())
        for k in range(self.T):
            block = cs.diagcat(self.params['Q'], self.params['R'])
            hessian[block_size*k:block_size*(k+1), block_size*k:block_size*(k+1)] = block
        hessian[block_size*(self.T):, block_size*(self.T):] = self.params['Qf']
        
        return hessian

    def rollout(self, x_bar, u_bar):
        """
        Rollout using linearized dynamics from model to populate x_bar.
        """
        for k in range (0, T):
            x_k = x_bar[:, k]
            u_k = u_bar[:, k]
            x_kp1, A_d, B_d, C_d = self.model.integrate(x_k, u_k)
            x_bar[:, k+1] = x_kp1.full().flatten()
        
        return x_bar

    def predict(self, x_bar_0, u_bar, track):
        # Initial guess of state. It doesn't have to be accurate but it helps to be dynamically accurate.
        x_bar = np.zeros((N_STATES, T+1))
        x_bar[:, 0] = x_bar_0
        
        # Gets x_bar based on u_bar, the warm-start for the actions.
        self.rollout(x_bar, u_bar)

        x_ref, _ = get_reference_trajectory(x_bar[:, 0], track, REF_VEL, 0.05)
        x_ref[3, :] = np.unwrap(x_ref[3, :])

        # Construct g (linear portion of QP)
        # Construct A (constraints of QP)
        # Construct a_lb, a_ub (bounds of the constraints)
        g = 0
        A = []
        lba = []
        uba = []
        # TODO: I don't want to construct this whole mess again and again every single time.
        # After we ensure that it works, can we make a function so we only sub in the values that change (i.e. x_bar / x_ref)?

        # TODO: we need to rethink how we construct lba / uba.
        l_u_ba_states = [0] * (N_STATES)
        lba_action = [-MAX_D_ACC, -MAX_D_STEER, -MAX_D_STEER]
        uba_action = [MAX_D_ACC, MAX_D_STEER, MAX_D_STEER]
        
        # We can initialize the entire A matrix corresponding to constraints.
        # It is a sparse matrix with dimensions (4 x (N-1) TIMES (Nx7))
        # (Dimensions) cols: we have to add self.nX for the final state (X_N)
        A_dynamics = cs.SX.zeros(self.nX * self.T, self.T*(self.nX + self.mU) + self.nX)
        a_dynamics_bounds = []
        
        A_controls = cs.SX.zeros()

        for k in range(T):
            g += -2 * cs.DM(x_ref[:, k].T @ self.params['Q']).T @ self.X[:, k]
            
            if k < (T - 1):
                d_u = (self.U[:, k+1] - self.U[:, k]) / self.dt
                # A += [d_u]

            x_kp1, A_d, B_d, C_d = cs_kbm.integrate(x_bar[:, k], u_bar[:, k])

            # (Indexing) row: each row has nX constraints corresponding to each state element.
            # col: each col has 2 blocks of (self.nX + self.mU).
            #   The first block is A B (Ax + Bu).
            #   The second block is [-I 0] (x_kp1, 0u)

            # TODO: bug with how we deal with constraint shapes because currently we're "skipping" timesteps in 
            # our construction of the row for A

            A_row = cs.horzcat(A_d, B_d, cs.SX.eye(4), cs.SX.zeros(4, 3))
            A_row_i_start   = self.nX * k
            A_row_i_end     = self.nX * (k + 1)
            A_col_i_start   = (self.nX + self.mU) * k
            A_col_i_end     = 2 * (self.nX + self.mU) * (k + 1)
            A_dynamics[A_row_i_start:A_row_i_end, A_col_i_start:A_col_i_end] = A_row
            a_dynamics_bounds.append(C_d)

        # A += [self.X[:, 0] - x_bar[:, 0]]
        A_start = cs.horzcat(cs.SX.eye(4), cs.SX.zeros(4, (self.nX + self.mU) * T))
        A_start_bounds = x_bar[:, 0]

        # QP has not been initialized yet
        if not self.S:
            opts = {'printLevel': 'tabular', 'print_in':True, 'print_out':True, 'print_problem':True, 'verbose':True}
            qp = {'h': self.H.sparsity(), 'a': A.sparsity()}
            self.S = cs.conic('S', 'qpoases', qp, opts)
        
        lbx = self.lbw
        ubx = self.ubw
        H = self.H
        g = cs.gradient(g, self.W)
        lba = cs.vertcat(*lba)
        uba = cs.vertcat(*uba)

        r = self.S(
            lbx=self.lbw,
            ubx=self.ubw,
            h=self.H,
            g=g,
            a=A,
            lba=lba,
            uba=uba,
        )

mpc = MPC(MPC_PARAMS, cs_kbm)
mpc.predict(np.array([1, 2, 3, 4]).T, np.random.rand(3, T), track)
exit()

################################
# BUILDING THE CONSTRAINTS (g) #
################################
# Dynamic constraints = N_STATES * (T+1), dAction constraint = N_ACTIONS * (T-1), terminal constraint = N_STATES
N_C = (N_STATES * (T+1)) + N_ACTIONS * (T-1) + N_STATES
g_sym = cs.MX.sym('g', N_STATES * (T+1) + N_ACTIONS * T)
A_sym = cs.MX.sym('A', N_C, N_STATES * (T+1) + N_ACTIONS * T)

# Initializing constraints
qp_opts = {'printLevel': 'none'}
prob    = {'h': H, 'a': A_sym.sparsity()}
solver  = cs.conic('solver', 'qpoases', prob, qp_opts)

    # # Handling constraints to smooth controls
    # if k < (T - 1):
    #     e_u = U[:, k+1] - U[:, k]
    #     J += e_u.T @ R_ @ e_u
        
    #     d_u = e_u / DT
    #     g += [d_u]
    #     lbg += [-MAX_D_ACC, -MAX_D_STEER, -MAX_D_STEER]
    #     ubg += [MAX_D_ACC, MAX_D_STEER, MAX_D_STEER]

#     x_kp1, A, B, C = cs_kbm.integrate(x_bar[:, k], u_bar[:, k])
    
#     g += [A @ X[:, k] + B @ U[:, k] + C - X[:, k + 1]]
#     lbg += [0] * N_STATES
#     ubg += [0] * N_STATES

# # Terminal cost
# e_xn = X[:, T] - x_ref[:, T]
# J += e_xn.T @ Qf @ e_xn

# # Setting a constraint for initial state.
# g += [X[:, 0] - x_bar[:, 0]]
# lbg += [0] * N_STATES
# ubg += [0] * N_STATES


for sim_time in range(sim_duration - 1):
    iter_start = time.time()

    # The ideal would be just a single call to MPC here
    # The function call will return X_mpc and U_mpc, which contains all the states and
    # actions across the time horizon.

    for i_iter in range(5):
        if i_iter == 0:
            u_bar = u_bar_start
        
        # Step 4: x_bar
        # Setting the very first x_bar to be same as the simulator
        x_bar = np.zeros((N_STATES, T + 1))        
        x_bar[:, 0] = x_sim[:, sim_time]

        # Step 5: Rollout using dynamics to get rest of x_bar
        for t in range(1, T + 1):
            # New implementation
            x_bar_t = x_bar[:, t-1]
            u_bar_t = u_bar[:, t-1]
            x_kp1, A_d, B_d, C_d = cs_kbm.integrate(x_bar_t, u_bar_t)
            x_bar[:, t] = x_kp1.full().flatten()

            # Comparing with previous implementation
            # A, B, C = kbm.linearize_model(x_bar_t, u_bar_t)
            # x_bar_compare[:, t] = A @ x_bar_t + B @ u_bar_t + C.flatten()
        
        g = []      # Constraint
        lbg = []    # Lower bound for constraints
        ubg = []    # Upper bound for constraints
        J = 0       # Cost

        # Initializing decision variables (X, U)
        X = cs.MX.sym('X', N_STATES, T + 1)

        # Initializing lower and upper bounds for state
        X_lb_row = cs.DM([-cs.inf, -cs.inf, 0, -cs.inf],)
        X_ub_row = cs.DM([cs.inf, cs.inf, MAX_SPEED, cs.inf],)
        X_lb = cs.repmat(X_lb_row, T + 1, 1)
        X_ub = cs.repmat(X_ub_row, T + 1, 1)
        
        U = cs.MX.sym('U', N_ACTIONS, T)
        U_lb_row = cs.DM([-MAX_ACC, -MAX_STEER, -MAX_STEER])
        U_ub_row = cs.DM([MAX_ACC, MAX_STEER, MAX_STEER])
        U_lb = cs.repmat(U_lb_row, T, 1)
        U_ub = cs.repmat(U_ub_row, T, 1)

        # Stacking decision variables and contraints
        vec = lambda M: cs.reshape(M, -1, 1)
        w = cs.vertcat(vec(X), vec(U))
        lbw = cs.vertcat(X_lb, U_lb)
        ubw = cs.vertcat(X_ub, U_ub)
        lbw = cs.vertcat(X_lb, U_lb)
        lbw = cs.vertcat(X_lb, U_lb)

        # Cost Matrices
        Q   = cs.DM(np.diag([20, 20, 10, 1e-3]))   # state error cost
        Qf  = cs.DM(np.diag([30, 30, 30, 1e-3]))   # state  final error cost
        R   = cs.DM(np.diag([10, 10, 10]))      # input cost
        R_  = cs.DM(np.diag([10, 10, 10]))      # input rate of change cost

        x_ref, _ = get_reference_trajectory(x_bar[:, 0], track, REF_VEL, 0.05)
        x_ref[3, :] = np.unwrap(x_ref[3, :])
        
        for k in range(T):
            # Initialize action for current_timestep
            e = X[:, k] - x_ref[:, k]
            J += e.T @ Q @ e
            J += U[:, k].T @ R @ U[:, k]

            # Handling constraints to smooth controls
            if k < (T - 1):
                e_u = U[:, k+1] - U[:, k]
                J += e_u.T @ R_ @ e_u
                
                d_u = e_u / DT
                g += [d_u]
                lbg += [-MAX_D_ACC, -MAX_D_STEER, -MAX_D_STEER]
                ubg += [MAX_D_ACC, MAX_D_STEER, MAX_D_STEER]

            x_kp1, A, B, C = cs_kbm.integrate(x_bar[:, k], u_bar[:, k])
            
            g += [A @ X[:, k] + B @ U[:, k] + C - X[:, k + 1]]
            lbg += [0] * N_STATES
            ubg += [0] * N_STATES

        # Terminal cost
        e_xn = X[:, T] - x_ref[:, T]
        J += e_xn.T @ Qf @ e_xn

        # Setting a constraint for initial state.
        g += [X[:, 0] - x_bar[:, 0]]
        lbg += [0] * N_STATES
        ubg += [0] * N_STATES

        qp_opts = {'printLevel': 'none'}
        prob    = {'x': w, 'f': J, 'g': cs.vertcat(*g)}
        solver  = cs.qpsol('solver', 'qpoases', prob, qp_opts)
        sol     = solver(lbx=lbw, ubx=ubw, lbg=cs.vertcat(*lbg), ubg=cs.vertcat(*ubg))

        X_mpc = cs.reshape(sol['x'][:N_STATES * (T+1)], 4, T+1).full()
        U_mpc = cs.reshape(sol['x'][N_STATES * (T+1):], 3, T).full()
        
        a_mpc   = np.array(U_mpc[0, :]).flatten()
        d_f_mpc = np.array(U_mpc[1, :]).flatten()
        d_r_mpc = np.array(U_mpc[2, :]).flatten()
        u_bar_new = np.vstack((a_mpc, d_f_mpc, d_r_mpc))

        delta_u = np.sum(np.sum(np.abs(u_bar_new - u_bar), axis=0), axis=0)
        if delta_u < 0.05:
            break
            
        u_bar = u_bar_new
    
    current_state = X_mpc[:, 0]
    l_state.append(current_state)
    x_mpc       = np.array(X_mpc[0, :]).flatten()
    y_mpc       = np.array(X_mpc[1, :]).flatten()
    v_mpc       = np.array(X_mpc[2, :]).flatten()
    theta_mpc   = np.array(X_mpc[3, :]).flatten()

    a_mpc   = np.array(U_mpc[0, :]).flatten()
    df_mpc  = np.array(U_mpc[1, :]).flatten()
    dr_mpc  = np.array(U_mpc[2, :]).flatten()

    l_a.append(a_mpc[0])
    l_df.append(df_mpc[0])
    l_dr.append(dr_mpc[0])

    u_bar = np.vstack((a_mpc, df_mpc, dr_mpc))
    
    # Take first action
    u_sim[:, sim_time] = u_bar[:, 0]
    
    # Measure elpased time to get results from cvxpy
    opt_time.append(time.time() - iter_start)

    # move simulation to t+1
    # tspan = [0, DT]
    x_sim[:, sim_time + 1] = cs_kbm.forward_one_step(x_sim[:, sim_time], u_sim[:, sim_time])
    # x_sim[:, sim_time + 1] = kbm.forward_one_step(x_sim[:, sim_time], u_sim[:, sim_time])

print("TOTAL TIME: {}".format(np.sum(opt_time)))
# 12.10 seconds
# 18.933314323425293 (previous implementation)

# Visualization
fig = plt.figure(figsize=(10, 6))
gs = fig.add_gridspec(2, 3)

ax1 = fig.add_subplot(gs[0, :])
ax1.plot(track[0, :], track[1, :], "b")
ax1.scatter(x_sim[0, :], x_sim[1, :], s=0.5, color='red', zorder=1)
# for i in range(x_sim.shape[1]):
#     ax1.text(
#         x_sim[0, i], x_sim[1, i], str(i),
#         fontsize=4, color='black', ha='center', va='center',
#         zorder=2  # Make sure it's above the scatter dots
#     )

ax1.plot(x_sim[0, :], x_sim[1, :], color='green', zorder=-1)
# plt.plot(x_traj[0, :], x_traj[1, :])
ax1.axis("equal")
ax1.set_ylabel("y")
ax1.set_xlabel("x")

ax2 = fig.add_subplot(gs[1, 0]) 
x = np.arange(len(l_a))
ax2.plot(x, l_a)
ax2.set_ylabel("acceleration command")
ax2.set_xlabel("time")

ax3 = fig.add_subplot(gs[1, 1])
x = np.arange(len(l_df))
ax3.plot(x, l_df)
ax3.set_ylabel("Front steering")
ax3.set_xlabel("time")

ax4 = fig.add_subplot(gs[1, 2])
x = np.arange(len(l_dr))
ax4.plot(x, l_dr)
ax4.set_ylabel("Rear steering")
ax4.set_xlabel("time")
plt.show()

# Test
# cs_kbm.integrate(np.random.rand(4, 1), np.random.rand(3, 1))
# cs_kbm.RK4(np.random.rand(4, 1), np.random.rand(3, 1), np.random.rand(4, 4), np.random.rand(4, 3))
