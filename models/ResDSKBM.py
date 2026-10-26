import os
from collections import deque

import casadi as cs
import torch

from learning.architecture.dynamicsModel import AdaptiveDynamicsModel
from .DSKBM import csDSKBM

torch.backends.cudnn.enabled = False


class ResDSKBM(csDSKBM):
    """
    DSKBM with a learned LSTM term added to the dynamics.

    The LSTM reads the last `trainPredSeqLen` pairs of (v, yaw) and action, and predicts mu. With
    use_lstm=True the model used by the MPC is X_dot = f_kbm(X, U) + mu. mu is held constant over the
    step: its Jacobians are not used, so it only enters the affine term C.
    """
    LSTM_STATE_SLICE = slice(2, 4)  # the LSTM sees (v, yaw), not position

    def __init__(self, params, use_lstm=True):
        """
        Params:
            params: [dict] contains all the parameters required for MPC to run. Uses 'mpc', 'network',
                'controls', 'train' and 'dataDir' (the model dir that holds adm.pt).
            use_lstm: [bool] if True, the LSTM term is added to the KBM in integrate() (CRN/PDN).
                If False, integrate() uses the nominal KBM only (DSKBM baseline). The LSTM is loaded
                either way so query_lstm() still works for diagnostics.
        """
        super().__init__(params['mpc'])
        self.use_lstm = use_lstm
        self.window_size = params['train']['trainPredSeqLen']

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.lstm = AdaptiveDynamicsModel(params['network'], params['controls'], params['mpc']).to(self.device)
        weights = torch.load(os.path.join(params['dataDir'], 'adm.pt'), map_location=torch.device(self.device))
        self.lstm.load_state_dict(weights)
        self.lstm.eval()
        self.hidden = None

        n_lstm_states = len(range(self.N_STATES)[self.LSTM_STATE_SLICE])
        self.x_q = deque([torch.zeros(n_lstm_states, dtype=torch.float32, device=self.device)] * self.window_size,
                         self.window_size)
        self.u_q = deque([torch.zeros(self.N_ACTIONS, dtype=torch.float32, device=self.device)] * self.window_size,
                         self.window_size)

    def window_from_queue(self, queue):
        """
        Converts deque into tensor
        """
        return torch.stack(list(queue))

    def update_queue(self, X_new, U_new):
        self.x_q.append(torch.tensor(X_new[self.LSTM_STATE_SLICE], dtype=torch.float32, device=self.device))
        self.u_q.append(torch.tensor(U_new, dtype=torch.float32, device=self.device))

    def query_lstm(self, useHidden=False, updateHidden=False):
        """
        Runs the LSTM on the current history window and returns its mean output, shape (1, 4).
        """
        hidden = self.hidden if useHidden else None
        with torch.no_grad():
            mean, _, hidden = self.lstm(
                self.window_from_queue(self.x_q).unsqueeze(0),
                self.window_from_queue(self.u_q).unsqueeze(0),
                hidden=hidden,
                returnHidden=True)
        if updateHidden:
            self.hidden = hidden
        return mean

    def integrate(self, X_bar, U_bar):
        """
        Linearizes the (KBM + LSTM) model at (X_bar, U_bar), then adds (X_bar, U_bar) to the history.

        Returns:
            x_kp1, A_d, B_d, C_d (see csDSKBM.step)
        """
        A, B, C = self.linearize(X_bar, U_bar)
        if self.use_lstm:
            mu = self.query_lstm().to("cpu")
            C = C + cs.DM(mu.numpy().T)

        # Update both x_q and u_q for querying during the next timestep
        self.update_queue(X_bar, U_bar)

        return self.step(X_bar, U_bar, A, B, C)
