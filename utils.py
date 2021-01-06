import numpy as np 
from numpy.random import default_rng
from scipy.interpolate import interp1d
from persim import bottleneck

rng = default_rng()

def sup_bottleneck(hom1, hom2, k, downsample=1):
    return max([bottleneck(rips1["dgms"][k], rips2["dgms"][k]) for rips1, rips2 in zip(hom1.rips[::downsample], hom2.rips[::downsample])])

def wiener(t_start=0.0, t_end=1.0, dt=0.1, mean=None, std_dev=1.0, n=1):
    t = np.arange(t_start, t_end, dt)

    if mean is None:
        mean = np.zeros((n, 1))

    increments = np.concatenate((mean + std_dev * rng.standard_normal(size=(n, 1)), np.sqrt(dt) * rng.standard_normal(size=(n, t.shape[0] - 1))), axis=1)
    w_seq = np.cumsum(increments, axis=1)
    wfunc = interp1d(t, w_seq, kind="linear")

    return wfunc

def wiener_s1(t_start=0.0, t_end=1.0, dt=0.1, mean=None, std_dev=1.0, n=1.0):
    """
    Simulates a Wiener process on S1 by taking the inverse stereographic projection of an R-Wiener process
    """

    wfunc = wiener(t_start, t_end, dt, mean, std_dev, n)

    def sfunc(t):
        w = wfunc(t)
        return np.arctan2((w ** 2 - 1) / (w ** 2 + 1), 2 * w / (w ** 2 + 1)) + np.pi / 2

    return sfunc