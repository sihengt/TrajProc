import casadi as cs
import numpy as np
from scripts import *
from DKBM_casadi import csDSKBM

## THIS CLASS IS NOT WORKING YET. REQUIRES MORE WORK TO DEBUG. I'VE SWITCHED BACK TO THE HIGH LEVEL SOLVER FOR NOW.

class MPCLowLevel:
    def __init__(self, params, model):
        """
        Params:
            model: contains functions for integrate
            params: dictionary containing the following attributes:
                1. n_x           (x in R^{n_x})
                2. m_u           (u in R^{m_u})
                3. T            (Time horizon of MPC)
                4. DT           (T control interval)
                5. x_lb[list]   (lower bounds of x)
                6. x_ub[list]   (upper bounds of x)
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
        self.X = cs.MX.sym('W', self.nX * (self.T+1) + self.mU * self.T, 1)
        
        # Initializes and flattens lower and upper bounds for state for each state variable in w.
        X_lb = (params['X_lb'], params['U_lb']) * self.T + (params['X_lb'],)
        X_ub = (params['X_ub'], params['U_ub']) * self.T + (params['X_ub'],)
        self.X_ub = cs.vertcat(*X_ub)
        self.X_lb = cs.vertcat(*X_lb)

        # Hessian
        self.H = self.init_H()

        # When initialized, we do not have to re-define the problem at each iteration.
        self.S = None

    def init_H(self):
        """
        Initializes Hessian of the MPC problem. This will be constant throughout the problem
        The structure of the Hessian are alternating diagonal stacks of Q and R up to the MPC horizon, followed by a
        a diagonal matrix of Qf as the terminal cost.
        """

        Q, Qf, R, R_ = self.params['Q'], self.params['Qf'], self.params['R'], self.params['R_']
        block_size = self.nX + self.mU
        hessian = cs.SX.zeros(self.X.size1(), self.X.size1())
        # Penalizes both state and controls
        for k in range(self.T):
            hessian[block_size * k : block_size * (k + 1), block_size * k : block_size * (k + 1)] = cs.diagcat(2*Q, 2*R)
        
        # TODO: something is wrong here. 
        # Adds penalty for smooth controls
        for k in range(self.T - 1):
            i_u     = block_size * k + self.nX
            i_up1   = block_size * (k+1) + self.nX
            
            hessian[i_u:i_u+self.mU,
                    i_u:i_u+self.mU] += 2*R_
            hessian[i_up1:i_up1+self.mU,
                    i_up1:i_up1+self.mU] += 2*R_

            hessian[i_u:i_u + self.mU,
                    i_up1:i_up1 + self.mU] -= 2*R_
            hessian[i_up1:i_up1 + self.mU,
                    i_u:i_u + self.mU] -= 2*R_
            
        hessian[block_size * (self.T):, block_size * (self.T):] = 2*Qf
        
        return hessian

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
        # Initial guess of state. It doesn't have to be accurate but it helps to be dynamically accurate.
        x_bar = np.zeros((N_STATES, self.T+1))
        x_bar[:, 0] = x_bar_0
        
        # Gets x_bar based on u_bar (warm-start for actions).
        self.rollout(x_bar, u_bar)

        # TODO: PARAMETERIZE!
        REF_VEL = 1.0

        x_ref, _ = get_reference_trajectory(x_bar[:, 0], track, REF_VEL, 0.05)
        x_ref[3, :] = np.unwrap(x_ref[3, :])
        
        ###############
        # CONSTRAINTS #
        ###############        
        
        # Initialize all three different types of constraints used in this MPC problem for path tracking:
        # 1. Dynamics constraint (A x_k + B u_k = x_kp1)
        # 2. Change in controls constraint ( dU <= (u_kp1 - u_k) / dt <= dU)
        # 3. Starting state constraint (x_k = x_bar_0)
        
        # Initializing A matrix for dynamics constraints
        A_dynamics = cs.SX.zeros(self.nX * self.T, self.X.size1())
        A_dynamics_bounds = []

        # Initializing A matrix for change in controls constraint
        A_controls = cs.SX.zeros(self.mU * (self.T-1) , self.X.size1())
        A_controls_ub = []
        A_controls_lb = []

        # Linear component of MPC problem
        g = []
        for k in range(self.T):
            g_block = cs.vertcat(-2 * self.params['Q'] @ cs.DM(x_ref[:, k]), cs.DM.zeros(self.mU, 1))
            g.append(g_block)

            if k < (self.T - 1):
                A_controls_row = cs.horzcat(cs.SX.zeros(3, 4), -cs.SX.eye(3)/self.dt, cs.SX.zeros(3, 4), cs.SX.eye(3)/self.dt)
                A_row_i_start   = self.mU * k
                A_row_i_end     = self.mU * (k + 1)
                A_col_i_start   = (self.nX + self.mU) * k
                A_col_i_end     = A_col_i_start + (self.nX + self.mU) * 2

                A_controls[A_row_i_start:A_row_i_end, A_col_i_start:A_col_i_end] = A_controls_row
                A_controls_lb.append(-self.params["dU_b"])
                A_controls_ub.append(self.params["dU_b"])

            x_kp1, A_d, B_d, C_d = self.model.integrate(x_bar[:, k], u_bar[:, k])

            # (Indexing) row: each row has nX constraints corresponding to each state element.
            # col: each col has 2 blocks of (self.nX + self.mU).
            #   The first block is A B (Ax + Bu).
            #   The second block is [-I 0] (x_kp1, 0u)
            
            # Terminal condition: there's actually mU less for the last col offset.
            if k == self.T - 1:
                A_row = cs.horzcat(A_d, B_d, -cs.SX.eye(self.nX))
                col_offset = self.nX * 2 + self.mU
            else:
                A_row = cs.horzcat(A_d, B_d, -cs.SX.eye(self.nX), cs.SX.zeros(self.nX, self.mU))
                col_offset = (self.nX + self.mU) * 2

            A_row_i_start   = self.nX * k
            A_row_i_end     = self.nX * (k + 1)
            A_col_i_start   = (self.nX + self.mU) * k
            A_col_i_end     = A_col_i_start + col_offset

            A_dynamics[A_row_i_start:A_row_i_end, A_col_i_start:A_col_i_end] = A_row
            A_dynamics_bounds.append(-C_d)

        # Terminal state linear component
        g.append(cs.vertcat(-2 * self.params['Qf'] @ cs.DM(x_ref[:, self.T])))

        # Adding constraint for starting state, which has to match.
        A_start = cs.horzcat(cs.DM.eye(4), cs.DM.zeros(4, (self.nX + self.mU) * self.T))
        A_start_bounds = cs.DM(x_bar[:, 0])

        # Concatenating all the constraints
        A = cs.vertcat(A_start, A_controls, A_dynamics)
        A_lb = cs.vertcat(A_start_bounds, *A_controls_lb, *A_dynamics_bounds)
        A_ub = cs.vertcat(A_start_bounds, *A_controls_ub, *A_dynamics_bounds)
        breakpoint()
        # QP has not been initialized yet. Skips this part if QP has been initialized with appropriate sparsity 
        # patterns already.
        if not self.S:
            opts = {
                'verbose':True
            }
            qp = {'h': self.H.sparsity(), 'a': A.sparsity()}
            self.S = cs.conic('S', 'qpoases', qp, opts)

        g = cs.vertcat(*g)

        r = self.S(
            lbx=self.X_lb,
            ubx=self.X_ub,
            h=cs.DM(self.H),
            g=g,
            a=cs.DM(A),
            lba=A_lb,
            uba=A_ub,
        )
        X_sol = r['x']
        # Take the evenly shaped portions first.
        # Reshape so every column corresponds to a time step
        X_reshaped = X_sol[: (self.nX + self.mU)*self.T].reshape(((self.nX + self.mU), self.T))
        states = X_reshaped[:self.nX, :]
        states = cs.horzcat(states, X_sol[-self.nX:])
        actions = X_reshaped[self.nX:, :]
        
        return states.full(), actions.full()

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
    T = 2          # MPC horizon
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