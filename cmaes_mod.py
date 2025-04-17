import numpy as np
from collections import deque
from typing import Optional

from cmaes import CMA


class CMA_Mod(CMA):
    """
    Args:

        history:
            A history window width for C computation (optional; default=3).
    """
    def __init__(self, history: Optional[int] = 3, *args, **kwargs) -> None:
        assert history > 0, "history must be longer than 0"
        super(CMA_Mod, self).__init__(*args, **kwargs)

        # history
        self._history = history
        self._hist_C1 = deque(maxlen=history)
        self._hist_Cmu = deque(maxlen=history)

    def update_C(self, w_io, y_k, delta_h_sigma=0):
        # (eq.47)
        rank_one = np.outer(self._pc, self._pc)
        rank_mu = np.sum(
            np.array([w * np.outer(y, y) for w, y in zip(w_io, y_k)]), axis=0
        )
        self._hist_C1.append(rank_one)
        self._hist_Cmu.append(rank_mu)

        # c_cov = self._c1 + self._cmu
        # self._C = sum([(1 - c_cov) ** (self._history - tau) * (self._c1 * self._hist_C1[tau] + self._cmu * self._hist_Cmu[tau]) for tau in range(len(self._hist_C1))]) + (1 - c_cov) ** self._history * np.identity(self._n_dim)
        alpha_hist = 1 / (len(self._hist_C1) + 1)
        self._C = sum(
            [alpha_hist * (self._c1 * self._hist_C1[tau] + self._cmu * self._hist_Cmu[tau])
             for tau in range(len(self._hist_C1))]
        ) + (1 - len(self._hist_C1) * alpha_hist) * np.identity(self._n_dim)
