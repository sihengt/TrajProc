import numpy as np
import casadi as cs
import torch
import l4casadi as l4c
import os
from learning.architecture.dynamicsModel import AdaptiveDynamicsModel
from collections import deque

import torch.nn as nn

torch.backends.cudnn.enabled = False

class FullResDSKBM:
    """
    Double Steer Kinematic Bicycle Model assumes:
        1. Velocity of each wheel points at the direction the wheel is facing (i.e. steering angle)
        2. omega = velocity / turning radius
        3. Tracks position and velocity of CG.
    """

    def __init__(self, params):
        """
        Constructor

        Params:
            params: [dict] contains all the parameters required for MPC to run.
        """        
        mpc_params = params['mpc']

        self.nX = 4
        self.mU = 3
        self.windowSize = params['train']['trainPredSeqLen']
        self.hasLSTM = True

        model_params = mpc_params['model']
        assert model_params['nStates']  == self.nX
        assert model_params['nActions'] == self.mU
        self.L      = model_params['L']     # Wheelbase
        self.l_f    = model_params['l_f']   # Front wheel
        self.l_r    = model_params['l_r']   # Rear wheel
        self.T      = mpc_params['T']       # Time horizon
        self.dt     = mpc_params['dt']      # dt

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

        ## NEW CODE
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.lstm = AdaptiveDynamicsModel(params['network'], params['controls'], mpc_params).to(self.device)
        self.lstm.load_state_dict(torch.load(os.path.join(params['dataDir'],'adm.pt'), map_location=torch.device(self.device)))
        self.hidden = None

        self.x_q = deque([torch.zeros(self.nX, dtype=torch.float32, device=self.device)] * self.windowSize, self.windowSize)
        self.xd_q = deque([torch.zeros(self.nX, dtype=torch.float32, device=self.device)] * self.windowSize, self.windowSize)
        self.u_q = deque([torch.zeros(self.mU, dtype=torch.float32, device=self.device)] * self.windowSize, self.windowSize)

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

    def window_from_queue(self, queue):
        """
        Converts deque into tensor
        """
        return torch.stack(list(queue))

    def update_queue(self, X_new, Xd_new, U_new):
        self.x_q.append(torch.tensor(X_new, dtype=torch.float32, device=self.device))
        self.xd_q.append(torch.tensor(Xd_new, dtype=torch.float32, device=self.device))
        self.u_q.append(torch.tensor(U_new, dtype=torch.float32, device=self.device))

    def query_lstm(self, useHidden=False, updateHidden=False):
        with torch.no_grad():
            if useHidden:                
                mean, _, hidden = self.lstm(
                    self.window_from_queue(self.x_q).unsqueeze(0),
                    self.window_from_queue(self.xd_q).unsqueeze(0),
                    self.window_from_queue(self.u_q).unsqueeze(0), 
                    hidden=self.hidden,
                    returnHidden=True) # (1, 4)
            else:
                mean, _, hidden = self.lstm(
                    self.window_from_queue(self.x_q).unsqueeze(0),
                    self.window_from_queue(self.xd_q).unsqueeze(0),
                    self.window_from_queue(self.u_q).unsqueeze(0), 
                    hidden=self.hidden,
                    returnHidden=True) # (1, 4)
            
            if updateHidden:
                self.hidden = hidden

        return mean

    # TODO: include LSTM stuff
    def integrate(self, X_bar, U_bar, Xd_prev):
        """
        Params:
            X_bar: 
            U_bar: 
        """
        ## Part 1: Getting A, B, g from KBM. This one is definitely correct.
        A = self.get_A(X_bar, U_bar)
        B = self.get_B(X_bar, U_bar)

        f_bar = self.f_x_dot(X=X_bar, U=U_bar)['X_dot']
        g = f_bar - A @ X_bar - B @ U_bar

        ## Part 2: Getting A_res, B_res, C_res from LSTM.
        # Update both x_q and u_q for querying during the next timestep
        self.update_queue(X_bar, Xd_prev, U_bar)

        x_window = self.window_from_queue(self.x_q)
        xd_window = self.window_from_queue(self.xd_q)
        u_window = self.window_from_queue(self.u_q)

        x_window = x_window.unsqueeze(0)
        xd_window = xd_window.unsqueeze(0)
        u_window = u_window.unsqueeze(0)

        # A_res, B_res, C_res for the LSTM
        prev_x = x_window[:, :-1, :].detach() # (1, seqLen-1, 4)
        prev_u = u_window[:, :-1, :].detach() # (1, seqLen-1, 3)
        prev_xd = xd_window.detach()
        curr_x = x_window[:, -1:, :].clone().detach().requires_grad_(True) # (1, 1, 4)
        curr_u = u_window[:, -1:, :].clone().detach().requires_grad_(True) # (1, 1, 3)        

        def f(curr_x, curr_u):
            x_w = torch.cat([prev_x, curr_x.unsqueeze(1)], dim=1)
            u_w = torch.cat([prev_u, curr_u.unsqueeze(1)], dim=1)
            mean, _ = self.lstm(x_w, prev_xd, u_w)
            return mean.squeeze(0)
        
        Jx, Ju = torch.autograd.functional.jacobian(
            f, 
            (curr_x.squeeze(0), curr_u.squeeze(0)),
            create_graph=False,
            vectorize=True
        )
        A_res = cs.DM(Jx.squeeze(1).cpu().numpy())
        B_res = cs.DM(Ju.squeeze(1).cpu().numpy())

        # To obtain C, we need to integrate f(\bar{x}, \bar{u}) using RK4
        mu = self.query_lstm(useHidden=False, updateHidden=False)
        # with torch.no_grad():
        #     if not self.hidden is None:
        #         mu, sigma, self.hidden = self.lstm(
        #             x_window,
        #             xd_window,
        #             u_window,
        #             hidden=self.hidden,
        #             returnHidden=True)
        #         mu = mu.cpu()
        #     else:
        #         mu, sigma, self.hidden = self.lstm(
        #             x_window,
        #             xd_window,
        #             u_window,
        #             returnHidden=True)
        #         mu = mu.cpu()
        
        C_res = cs.DM(mu.numpy().T) - A_res @ X_bar - B_res @ U_bar

        # Linear so just add them together.
        A_eff = A - A_res
        B_eff = B - B_res
        C_eff = g + C_res
        
        x_kp1 = self.RK4(X_bar, U_bar, A_eff, B_eff, C_eff)
        
        A_exp_2 = A_eff @ A_eff
        A_exp_3 = A_exp_2 @ A_eff
        A_exp_4 = A_exp_3 @ A_eff

        A_d = cs.DM(np.eye(4)) + (self.dt * A_eff) + (self.dt**2 / 2 * A_exp_2) + (self.dt**3/6 * A_exp_3) + (self.dt**4/24 * A_exp_4)
        B_d = self.dt * B_eff + (self.dt**2/2 * A_eff @ B_eff) + (self.dt**3/6 * A_exp_2 @ B_eff) + (self.dt**4/24 * A_exp_3 @ B_eff)
        C_d = self.dt * C_eff + self.dt**2/2 * A_eff @ C_eff + self.dt**3/6 * A_exp_2 @ C_eff + self.dt**4/24 * A_exp_3 @ C_eff
        
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
        k2 = A @ (x_accumulated + self.dt/2 * k1) + B @ u_int + C
        k3 = A @ (x_accumulated + self.dt/2 * k2) + B @ u_int + C
        k4 = A @ (x_accumulated + self.dt * k3) + B @ u_int + C
        x_accumulated += self.dt/6 * (k1 + 2*k2 + 2*k3 + k4)

        return cs.Function('F', [x0_int, u_int, A, B, C], [x_accumulated], ['x0', 'u', 'A', 'B', 'C'], ['xf'])

    def initialize_dae(self):
        v       = self.X[2]
        theta   = self.X[3]
        a       = self.U[0]
        d_f     = self.U[1]
        d_r     = self.U[2]
        
        beta    = self.sideslip(delta_f=d_f, delta_r=d_r)['beta']

        x_dot   = v * cs.cos(beta + theta)
        y_dot   = v * cs.sin(beta + theta)
        v_dot   = a
        theta_dot = (v * cs.cos(beta)) / (self.l_f + self.l_r) * (cs.tan(d_f) - cs.tan(d_r))
        F = cs.vertcat(x_dot, y_dot, v_dot, theta_dot)
        
        dae = {'x':self.X, 'p':self.U, 'ode':F}

        return dae
    
    def forward_one_step(self, x0, u):
        """
        Integrates the DAE by one timestep specified by self.dt.
        
        Params:
            x0: state to integrate for
            u: control input
        """
        F = cs.integrator('F', 'idas', self.dae, {'tf':self.dt})
        r = F(x0=x0, p=u)
        return r['xf'].full().flatten()


