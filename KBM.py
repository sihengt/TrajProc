import numpy as np
from scipy.integrate import odeint

# TODO: maybe for MPC we'll have a base model like this and inherit from it.

class KinematicBicycleModel:
    def __init__(self, n_states, n_actions, L, T, DT):
        self.x = np.zeros((n_states, 1))
        self.u = np.zeros((n_actions, T))
        self.L = L      # Wheelbase
        self.T = T      # Time horizon
        self.DT = DT    # dt
        
        self.n_states = n_states
    
    def linearize_model(self, X_bar, U_bar):
        """
        Requires an operating point to linearize about.

        Params:
            x_bar: assumed to \in R^4 [x, y, v, yaw]
            u_bar: assumed to \in R^2 [a, delta]
        """
        x_bar       = X_bar[0]
        y_bar       = X_bar[1]
        v_bar       = X_bar[2]
        theta_bar   = X_bar[3]
        
        a_bar       = U_bar[0]
        delta_bar   = U_bar[1]

        A = np.zeros((4, 4))
        A[0, 2] = np.cos(theta_bar)
        A[0, 3] = -v_bar * np.sin(theta_bar)
        A[1, 2] = np.sin(theta_bar)
        A[1, 3] = v_bar * np.cos(theta_bar)
        A[3, 2] = np.tan(delta_bar) / self.L
        A_lin = np.eye(4) + self.DT * A
        
        B = np.zeros((4, 2))
        B[2, 0] = 1
        B[3, 1] = v_bar / self.L * np.cos(delta_bar)**2
        B_lin = self.DT * B

        f_x_bar_u_bar = np.array([
            [v_bar * np.cos(theta_bar)],
            [v_bar * np.sin(theta_bar)],
            [a_bar],
            [(v_bar/self.L) * np.tan(delta_bar)]
        ])
        
        C_lin = self.DT * (f_x_bar_u_bar - A @ X_bar.reshape(4, 1) - B @ U_bar.reshape(2, 1))

        return A_lin, B_lin, C_lin

    def model(self, x, t, u):
        x_dot = x[2] * np.cos(x[3])
        y_dot = x[2] * np.sin(x[3])
        v_dot = u[0]
        theta_dot = x[2] * np.tan(u[1]) / self.L

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
