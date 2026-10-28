import os
from collections import deque

import casadi as cs
import torch

from learning.architecture.dynamicsModel import AdaptiveDynamicsModel
from .DSKBM import csDSKBM

torch.backends.cudnn.enabled = False


class ResDSKBM(csDSKBM):
    """
    DSKBM with a learned LSTM term in the dynamics.

    The LSTM reads the last `trainPredSeqLen` pairs of (v, yaw) and action, and predicts mu. With
    use_lstm=True the model used by the MPC at the linearization point (X_bar, U_bar) is:
        target='residual' (CRN): X_dot = f_kbm(X_bar, U_bar) + mu
        target='full'     (PDN): X_dot = mu
    The network's Jacobians are not used. For 'residual', A_d and B_d are the exact Jacobians of the step with mu
    held fixed. For 'full', A_d and B_d are the ones of the KBM step.
    """
    LSTM_STATE_SLICE = slice(2, 4)  # the LSTM sees (v, yaw), not position
    TARGETS = ('residual', 'full')

    def __init__(self, params, use_lstm=True, target='residual'):
        """
        Params:
            params: [dict] contains all the parameters required for MPC to run. Uses 'mpc', 'network',
                'controls', 'train' and 'dataDir' (the model dir that holds adm.pt).
            use_lstm: [bool] if True, integrate() uses the LSTM (CRN/PDN). If False, integrate() uses
                the nominal KBM only (DSKBM baseline). The LSTM is loaded either way so query_lstm()
                still works for diagnostics.
            target: what the network was trained to predict: 'residual' (x_dot - f_kbm) or 'full' (x_dot).
        """
        super().__init__(params['mpc'])
        if target not in self.TARGETS:
            raise ValueError(f"target must be one of {self.TARGETS}, got {target!r}")
        self.use_lstm = use_lstm
        self.target = target
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
        Adds (X_bar, U_bar) to the history, then linearizes the (KBM + LSTM) model at (X_bar, U_bar).

        The LSTM window ends with (X_bar, U_bar), the step being predicted, as in training.

        Returns:
            x_kp1, A_d, B_d, C_d (see csDSKBM.step)
        """
        self.update_queue(X_bar, U_bar)

        if not self.use_lstm:
            return self.step(X_bar, U_bar)

        mu = cs.DM(self.query_lstm().to("cpu").numpy().T)
        if self.target == 'residual':
            return self.step(X_bar, U_bar, mu)

        # Replace f_kbm(X_bar, U_bar) by mu, so X_dot = mu at the linearization point. The network has no
        # Jacobians here, so the QP gets the ones of the KBM step.
        x_kp1, _, _, _ = self.step(X_bar, U_bar, mu - self.f_x_dot(X_bar, U_bar))
        _, A_d, B_d, _ = self.step(X_bar, U_bar)
        return x_kp1, A_d, B_d, x_kp1 - A_d @ X_bar - B_d @ U_bar
