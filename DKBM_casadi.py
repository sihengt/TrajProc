import numpy as np
from scipy.integrate import odeint
import sympy as sp
import casadi as cs

# Double Steer Kinematic Bicycle Model
class csDSKBM:
    """
    Double Steer Kinematic Bicycle Model assumes:
        1. Velocity of each wheel points at the direction the wheel is facing (i.e. steering angle)
        2. omega = velocity / turning radius
        3. Tracks position and velocity of CG.
    """

    """
    Constructor

    Params:
        n_states: 
        n_actions: 
        L: 
        l_f: 
        l_r: 
        T: 
        N: 
        M: 
    """
    def __init__(self, n_states, n_actions, L, l_f, l_r, T, N):
        self.x = np.zeros((n_states, 1))
        self.u = np.zeros((n_actions, T))
        self.L = L          # Wheelbase
        self.l_f = l_f      # Front wheel
        self.l_r = l_r      # Rear wheel
        self.T = T          # Time horizon
        self.N = N          # Control intervals
        self.DT = T/N     # dt

        # ---- CasADi expressions --- #
        # State
        self.x = cs.MX.sym('x')
        self.y = cs.MX.sym('y')
        self.v = cs.MX.sym('v')
        self.theta = cs.MX.sym('theta')
        self.X = cs.vertcat(self.x, self.y, self.v, self.theta)
        
        # Reference state
        self.x_ref = cs.MX.sym('x_ref')
        self.y_ref = cs.MX.sym('y_ref')
        self.v_ref = cs.MX.sym('v_ref')
        self.theta_ref = cs.MX.sym('theta_ref')
        self.X_ref = cs.vertcat(self.x_ref, self.y_ref, self.v_ref, self.theta_ref)
        
        # Controls
        self.a = cs.MX.sym('a')
        self.delta_f = cs.MX.sym('df')
        self.delta_r = cs.MX.sym('dr')
        self.U = cs.vertcat(self.a, self.delta_f, self.delta_r)

        self.sideslip = self.init_sideslip_function()

        # Expression for x_dot
        self.X_dot = self.init_X_dot()
        self.f_x_dot = cs.Function('X_dot', [self.X, self.U], [self.X_dot], ['X', 'U'], ['X_dot'])

        self.get_A = self.init_A()
        self.get_B = self.init_B()
        self.RK4 = self.init_rk4()
        
        self.dae = self.initialize_dae()

    def init_sideslip_function(self):
        return cs.Function(
            'get_sideslip',
            [self.delta_f, self.delta_r],
            [cs.arctan( (self.l_r * cs.tan(self.delta_f) + self.l_f * cs.tan(self.delta_r)) / (self.l_r + self.l_f))],
            ['delta_f', 'delta_r'],
            ['beta']
        )
        
    def init_X_dot(self):
        beta = self.sideslip(delta_f=self.delta_f, delta_r=self.delta_r)
        x_dot = self.v * cs.cos(beta['beta'] + self.theta)
        y_dot = self.v * cs.sin(beta['beta'] + self.theta)
        v_dot = self.a
        theta_dot = (self.v * cs.cos(beta['beta'])) / (self.l_f + self.l_r) * \
            (cs.tan(self.delta_f) - cs.tan(self.delta_r))
        return cs.vertcat(x_dot, y_dot, v_dot, theta_dot)

    def init_A(self):
        return cs.Function('get_A', [self.X, self.U], [cs.jacobian(self.X_dot, self.X)])

    def init_B(self):
        return cs.Function('get_B', [self.X, self.U], [cs.jacobian(self.X_dot, self.U)])

    def integrate(self, X_bar, U_bar):
        A = self.get_A(X_bar, U_bar)
        B = self.get_B(X_bar, U_bar)
        
        # To obtain C, we need to integrate f(\bar{x}, \bar{u}) using RK4
        
        f_bar = self.f_x_dot(X=X_bar, U=U_bar)['X_dot']
        g = f_bar - A @ X_bar - B @ U_bar
        
        x_kp1 = self.RK4(X_bar, U_bar, A, B, g)
        
        A_exp_2 = A @ A
        A_exp_3 = A_exp_2 @ A
        A_exp_4 = A_exp_3 @ A

        A_d = cs.DM(np.eye(4)) + (self.DT * A) + (self.DT**2 / 2 * A_exp_2) + (self.DT**3/6 * A_exp_3) + (self.DT**4/24 * A_exp_4)
        B_d = self.DT * B + (self.DT**2/2 * A @ B) + (self.DT**3/6 * A_exp_2 @ B) + (self.DT**4/24 * A_exp_3 @ B)
        C_d = self.DT * g + self.DT**2/2 * A @ g + self.DT**3/6 * A_exp_2 @ g + self.DT**4/24 * A_exp_3 @ g     
        
        # x_kp1_compare = A_d @ X_bar + B_d @ U_bar + C_d
        # assert(x_kp1 == x_kp1_compare)

        return x_kp1, A_d, B_d, C_d
        
    def init_rk4(self):
        x0_int = cs.MX.sym("x0_int", 4)
        u_int = cs.MX.sym("u_int", 3)
        
        A = cs.MX.sym("A", 4, 4)
        B = cs.MX.sym("B", 4, 3)
        C = cs.MX.sym("C", 4)
        
        x_accumulated = x0_int

        # for j in range(self.M):
        k1 = A @ x_accumulated + B @ u_int + C
        k2 = A @ (x_accumulated + self.DT/2 * k1) + B @ u_int + C
        k3 = A @ (x_accumulated + self.DT/2 * k2) + B @ u_int + C
        k4 = A @ (x_accumulated + self.DT * k3) + B @ u_int + C
        x_accumulated += self.DT/6 * (k1 + 2*k2 + 2*k3 + k4)

        return cs.Function('F', [x0_int, u_int, A, B, C], [x_accumulated], ['x0', 'u', 'A', 'B', 'C'], ['xf'])

    def linearize_model(self, X_bar, U_bar):
        x_bar       = X_bar[0]
        y_bar       = X_bar[1]
        v_bar       = X_bar[2]
        theta_bar   = X_bar[3]
        
        a_bar       = U_bar[0]
        d_f_bar     = U_bar[1]
        d_r_bar     = U_bar[2] 

        beta_bar    = self.calculate_sideslip(d_f_bar, d_r_bar)

        A = np.zeros((4, 4))
        A[0, 2] = np.cos(theta_bar + beta_bar)
        A[0, 3] = -v_bar * np.sin(theta_bar + beta_bar)
        A[1, 2] = np.sin(theta_bar + beta_bar)
        A[1, 3] = v_bar * np.cos(theta_bar + beta_bar)
        A[3, 2] = np.cos(beta_bar) / (self.l_f + self.l_r) * (np.tan(d_f_bar) - np.tan(d_r_bar))

        A_lin = np.eye(4) + self.DT * A
        
        B = np.zeros((4, 3))
        B[2, 0] = 1
        B[3, 1] = self.d_f_d_df(v_bar, self.l_f, self.l_r, beta_bar, d_f_bar, d_r_bar)
        B[3, 2] = self.d_f_d_dr(v_bar, self.l_f, self.l_r, beta_bar, d_f_bar, d_r_bar)
        B_lin = self.DT * B

        f_x_bar_u_bar = np.array([
            [v_bar * np.cos(theta_bar)],
            [v_bar * np.sin(theta_bar)],
            [a_bar],
            [(v_bar * np.cos(beta_bar)) / (self.l_f + self.l_r) * (np.tan(d_f_bar) - np.tan(d_r_bar))]
        ])
        
        C_lin = self.DT * (f_x_bar_u_bar - A @ X_bar.reshape(4, 1) - B @ U_bar.reshape(3, 1))

        return A_lin, B_lin, C_lin

    def initialize_dae(self):
        v       = self.X[2]
        theta   = self.X[3]
        a       = self.U[0]
        d_f     = self.U[1]
        d_r     = self.U[2]
        
        beta    = self.sideslip(delta_f=d_f, delta_r=d_r)['beta']

        x_dot   = v * np.cos(beta + theta)
        y_dot   = v * np.sin(beta + theta)
        v_dot   = a
        theta_dot = (v * np.cos(beta)) / (self.l_f + self.l_r) * (np.tan(d_f) - np.tan(d_r))
        F = cs.vertcat(x_dot, y_dot, v_dot, theta_dot)
        
        dae = {'x':self.X, 'p':self.U, 'ode':F}

        return dae
    
    # TODO: re-think design for sure.
    def forward(self, x0, u):
        x_ = np.zeros((self.n_states, self.T + 1))
        x_[:, 0] = x0

        for t in range(1, self.T + 1):
            tspan = [0, self.DT]
            x_next = odeint(self.model, x0, tspan, args=(u[:, t-1],))
            x0 = x_next[1]
            x_[:, t] = x_next[1]
        
        return x_
    
    def forward_one_step(self, x0, u):
        F = cs.integrator('F', 'idas', self.dae, {'tf':self.DT})
        r = F(x0=x0, p=u)
        return r['xf'].full().flatten()
        # tspan = [0, self.DT]
        # x_next = odeint(self.model, x0, tspan, args=(u,))
        # return x_next[1]
