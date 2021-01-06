import numpy as np 
from scipy.interpolate import interp1d, interp2d
from ripser import ripser

from swarm.swarmalator import Swarmalator


class TimeDependentHomology(object):

    def __init__(self, swarm, sol, eps=None, t_start=0, t_end=None, dt=0.1, maxdim=None, **kwargs):
        if maxdim is None:
            maxdim = swarm.n + 2

        self.maxdim = maxdim

        if eps is None:
            eps = np.arange(0, 2.5, 0.1)
        self.eps = eps
        self.rips = list()
        
        if t_end is None:
            t_end = sol.t[-1]

        self.t_uniform = np.arange(t_start, t_end, dt)
        self.y_uniform = interp1d(sol.t, sol.y, kind="linear")(self.t_uniform)

        for idx in range(self.t_uniform.shape[0]):
            z = self.y_uniform[:, idx]
            
            if swarm.n == 2:
                pos = np.hstack((z[:swarm.N, np.newaxis], z[swarm.N:2 * swarm.N, np.newaxis]))
                phase = z[2 * swarm.N:, np.newaxis]
            elif swarm.n == 3:
                pos = np.hstack((z[:swarm.N, np.newaxis], z[swarm.N:2 * swarm.N, np.newaxis], z[2 * swarm.N:3 * swarm.N, np.newaxis]))
                phase = z[3 * swarm.N:, np.newaxis]

            data = np.hstack((pos, np.cos(phase), np.sin(phase)))
            self.rips.append(ripser(data, maxdim=maxdim, **kwargs))

        #self.grids = [None for i in range(maxdim)]

    def betti(self, eps, i, t_idx, downsample=1):
        return len(np.where((eps >= self.rips[::downsample][t_idx]["dgms"][i][:, 0]) * (eps < self.rips[::downsample][t_idx]["dgms"][i][:, 1]))[0])
    
    def grid(self, i, downsample=1):
        #if self.grids[i] is None:
        c_grid = np.zeros((self.eps.shape[0], self.t_uniform[::downsample].shape[0]))

        for j in range(c_grid.shape[0]):
            for k in range(c_grid.shape[1]):
                c_grid[j, k] = self.betti(self.eps[j], i, k, downsample)

        return c_grid








