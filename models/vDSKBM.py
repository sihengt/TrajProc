import numpy as np
from scipy.integrate import odeint
import sympy as sp
import casadi as cs

class vDSKBM:
    """
    Velocity Double Steer Kinematic Bicycle Model assumes:
        1. Velocity of each wheel points at the direction the wheel is facing (i.e. steering angle)
        2. omega = velocity / turning radius
        3. Tracks position and velocity of CG.
    """
    def __init__(self, L, l_f, l_r, T, dt):
        """
        Params:
            n_states: 
            n_actions: 
            L:  wheelbase
            l_f: distance from CG to front wheel 
            l_r: distance from CG to rear wheel
            T: time horizon of MPC problem
            N: control intervals
            M: 
        """
        self.nX = 3
        self.mU = 3
        
        self.L = L          # Wheelbase
        self.l_f = l_f      # Front wheel
        self.l_r = l_r      # Rear wheel
        self.T = T          # Time horizon
        self.dt = dt        # dt

        # ---- CasADi expressions --- #
        # State
        self.x = cs.MX.sym('x')
        self.y = cs.MX.sym('y')
        self.theta = cs.MX.sym('theta')
        self.X = cs.vertcat(self.x, self.y, self.theta)
        
        # Reference state
        self.x_ref = cs.MX.sym('x_ref')
        self.y_ref = cs.MX.sym('y_ref')
        self.theta_ref = cs.MX.sym('theta_ref')
        self.X_ref = cs.vertcat(self.x_ref, self.y_ref, self.theta_ref)
        
        # Controls
        self.v = cs.MX.sym('v')
        self.delta_f = cs.MX.sym('df')
        self.delta_r = cs.MX.sym('dr')
        self.U = cs.vertcat(self.v, self.delta_f, self.delta_r)

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
        theta_dot = (self.v * cs.cos(beta['beta'])) / (self.l_f + self.l_r) * \
            (cs.tan(self.delta_f) - cs.tan(self.delta_r))
        return cs.vertcat(x_dot, y_dot, theta_dot)

    def init_A(self):
        return cs.Function('get_A', [self.X, self.U], [cs.jacobian(self.X_dot, self.X)])

    def init_B(self):
        return cs.Function('get_B', [self.X, self.U], [cs.jacobian(self.X_dot, self.U)])

    def integrate(self, X_bar, U_bar):
        """
        Integration with RK4
        """
        A = self.get_A(X_bar, U_bar)
        B = self.get_B(X_bar, U_bar)
        
        # To obtain C, we need to integrate f(\bar{x}, \bar{u}) using RK4
        f_bar = self.f_x_dot(X=X_bar, U=U_bar)['X_dot']
        g = f_bar - A @ X_bar - B @ U_bar
        
        x_kp1 = self.RK4(X_bar, U_bar, A, B, g)
        
        A_exp_2 = A @ A
        A_exp_3 = A_exp_2 @ A
        A_exp_4 = A_exp_3 @ A

        A_d = cs.DM(np.eye(self.nX)) + (self.dt * A) + (self.dt**2 / 2 * A_exp_2) + (self.dt**3/6 * A_exp_3) + (self.dt**4/24 * A_exp_4)
        B_d = self.dt * B + (self.dt**2/2 * A @ B) + (self.dt**3/6 * A_exp_2 @ B) + (self.dt**4/24 * A_exp_3 @ B)
        C_d = self.dt * g + self.dt**2/2 * A @ g + self.dt**3/6 * A_exp_2 @ g + self.dt**4/24 * A_exp_3 @ g     
        
        return x_kp1, A_d, B_d, C_d
        
    def init_rk4(self):
        x0_int = cs.MX.sym("x0_int", self.nX)
        u_int = cs.MX.sym("u_int", self.mU)
        
        A = cs.MX.sym("A", self.nX, self.nX)
        B = cs.MX.sym("B", self.nX, self.mU)
        C = cs.MX.sym("C", self.nX)
        
        x_accumulated = x0_int

        k1 = A @ x_accumulated + B @ u_int + C
        k2 = A @ (x_accumulated + self.dt/2 * k1) + B @ u_int + C
        k3 = A @ (x_accumulated + self.dt/2 * k2) + B @ u_int + C
        k4 = A @ (x_accumulated + self.dt * k3) + B @ u_int + C
        x_accumulated += self.dt/6 * (k1 + 2*k2 + 2*k3 + k4)

        return cs.Function('F', [x0_int, u_int, A, B, C], [x_accumulated], ['x0', 'u', 'A', 'B', 'C'], ['xf'])

    def initialize_dae(self):
        theta   = self.X[2]
        v       = self.U[0]
        d_f     = self.U[1]
        d_r     = self.U[2]
        
        beta    = self.sideslip(delta_f=d_f, delta_r=d_r)['beta']

        x_dot   = v * np.cos(beta + theta)
        y_dot   = v * np.sin(beta + theta)
        theta_dot = (v * np.cos(beta)) / (self.l_f + self.l_r) * (np.tan(d_f) - np.tan(d_r))
        F = cs.vertcat(x_dot, y_dot, theta_dot)
        
        dae = {'x':self.X, 'p':self.U, 'ode':F}

        return dae
    
    def forward_one_step(self, x0, u):
        F = cs.integrator('F', 'idas', self.dae, {'tf':self.dt})
        r = F(x0=x0, p=u)
        return r['xf'].full().flatten()
