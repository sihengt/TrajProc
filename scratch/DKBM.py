import numpy as np
from scipy.integrate import odeint
import sympy as sp

# Double Steer Kinematic Bicycle Model
class DSKinematicBicycleModel:
    """
    Double Steer Kinematic Bicycle Model assumes:
        1. Velocity of each wheel points at the direction the wheel is facing (i.e. steering angle)
        2. omega = velocity / turning radius
        3. Tracks position and velocity of CG.
    """
    def __init__(self, n_states, n_actions, L, l_f, l_r, T, DT):
        self.x = np.zeros((n_states, 1))
        self.u = np.zeros((n_actions, T))
        self.L = L      # Wheelbase
        self.l_f = l_f  # Front wheel
        self.l_r = l_r  # Rear wheel
        self.T = T      # Time horizon
        self.DT = DT    # dt

        v, l_f, l_r, beta, d_f, d_r = sp.symbols('v l_f l_r beta d_f d_r')

        # Expression for psi_dot
        theta_dot = (v * sp.cos(beta))/(l_f + l_r) * (sp.tan(d_f) - sp.tan(d_r))
        d_theta_dot_d_df = sp.diff(theta_dot, d_f)
        d_theta_dot_d_dr = sp.diff(theta_dot, d_r)
        self.d_f_d_df = sp.lambdify((v, l_f, l_r, beta, d_f, d_r), d_theta_dot_d_df)
        self.d_f_d_dr = sp.lambdify((v, l_f, l_r, beta, d_f, d_r), d_theta_dot_d_dr)

        # TODO: maybe use self.M instead of self.n_states
        self.n_states = n_states

    def calculate_sideslip(self, d_f, d_r):
        return np.arctan( (self.l_r * np.tan(d_f) + self.l_f * np.tan(d_r)) / (self.l_r + self.l_f))
    
    def linearize_model(self, X_bar, U_bar):
        """
        Requires an operating point to linearize about.

        Params:
            x_bar: assumed to \in R^4 [x, y, v, yaw]
            u_bar: assumed to \in R^2 [a, delta_f, delta_r]
        """
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

    def model(self, x, t, u):
        v       = x[2]
        theta   = x[3]
        a       = u[0]
        d_f     = u[1]
        d_r     = u[2]
        
        beta    = self.calculate_sideslip(d_f, d_r)

        x_dot   = v * np.cos(beta + theta)
        y_dot   = v * np.sin(beta + theta)
        v_dot   = a
        theta_dot = (v * np.cos(beta)) / (self.l_f + self.l_r) * (np.tan(d_f) - np.tan(d_r))

        dqdt = [x_dot, y_dot, v_dot, theta_dot]

        return dqdt
    
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
        tspan = [0, self.DT]
        x_next = odeint(self.model, x0, tspan, args=(u,))
        return x_next[1]
