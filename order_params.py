import numpy as np

from swarm.swarmalator import Swarmalator


def S(swarm, sol):
    S_plus = np.abs(1.0 / swarm.N * sum([np.exp(1j * (np.arctan2(sol.y[j, :], sol.y[swarm.N + j, :]) + sol.y[2 * swarm.N + j, :])) for j in range(swarm.N)]))
    S_minus = np.abs(1.0 / swarm.N * sum([np.exp(1j * (np.arctan2(sol.y[j, :], sol.y[swarm.N + j, :]) - sol.y[2 * swarm.N + j, :])) for j in range(swarm.N)]))
    return S_plus, S_minus


def gamma(swarm, sol, t_start):
    start_idx = np.where(sol.t > t_start)[0][0]
    dtheta = np.zeros((swarm.N, sol.t[start_idx:].shape[0]))

    for idx in range(sol.t[start_idx:].shape[0]):
        x1 = sol.y[:swarm.N, start_idx + idx]
        x2 = sol.y[swarm.N:2 * swarm.N, start_idx + idx]
        theta = sol.y[2 * swarm.N:, start_idx + idx]
        
        x1d = x1[:, np.newaxis] - x1 
        x2d = x2[:, np.newaxis] - x2

        thetad = theta[:, np.newaxis] - theta
        norm_sq = x1d ** 2 + x2d ** 2
        norm = np.sqrt(norm_sq)

        theta_rhs = -swarm.K / swarm.N * np.sum((1 - np.eye(theta.shape[0])) * (swarm.G(norm)  * swarm.H(thetad)), axis=1)
        dtheta[:, idx] = theta_rhs

    return dtheta @ np.diff(sol.t[start_idx:])