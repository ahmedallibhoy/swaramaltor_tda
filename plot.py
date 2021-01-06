import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation

from scipy.interpolate import interp1d


def animate_swarm(swarms, sols, num_frames, t_start=None, t_end=None, size=100, fig=None, axs=None, figsize=(10, 8), lim=None, cmap="hsv", interp=True, **kwargs):
    if not isinstance(swarms, list):
        swarms = [swarms]
        sols = [sols]
        axs = [axs]

    if t_start is None:
        t_start = 0.0

    if t_end is None:
        t_end = sols[0].t[-1]

    if interp or len(sols) > 1:
        t_range = np.linspace(t_start, t_end, num_frames)
        y_ranges = [interp1d(sol.t, sol.y, kind="linear")(t_range) for sol in sols]
    else:
        t_range = sols[0].t
        y_ranges = [sol.y for sol in sols]

    cmap = plt.get_cmap("hsv")

    if fig is None:
        fig = plt.figure(figsize=figsize)  

    if t_start is not None and not interp:
        start_idx = np.where(t_range > t_start)[0][0]
    else:
        start_idx = 0

    if t_end is not None and not interp:
        nsteps = np.where(t_range < t_end)[0][-1] - start_idx
    else:
        nsteps = t_range.shape[0] - start_idx

    if isinstance(lim, int) or isinstance(lim, float):
        lim = [-float(lim), float(lim)] * swarms[0].n

    if swarms[0].n == 2:
        if axs is None:
            axs = [plt.gca()]
        
        scats = [None for _ in sols]
        tts = [None for _ in sols]

        for idx, (swarm, sol, ax, y_range) in enumerate(zip(swarms, sols, axs, y_ranges)):
            scats[idx] = ax.scatter(y_range[:swarm.N, start_idx], 
                                    y_range[swarm.N:2 * swarm.N, start_idx], s=size,
                                    c=cmap(np.mod(y_range[2 * swarm.N:, start_idx], 2 * np.pi) / (2 * np.pi)))
            scats[idx].set_cmap(cmap)
            tts[idx] = ax.text(.01, .01, s="t=%1.2f" % t_range[start_idx], transform=ax.transAxes)

            if lim is not None:
                ax.set_xlim(lim[:2])
                ax.set_ylim(lim[2:])
        
    elif swarm.n == 3:
        if axs is None:
            axs = [fig.add_subplot(111, projection='3d')]

        scats = [None for _ in sols]
        tts = [None for _ in sols]

        for idx, (swarm, sol, ax, y_range) in enumerate(zip(swarms, sols, axs, y_ranges)):
            scats[idx] = ax.scatter(y_range[:swarm.N, start_idx], 
                               y_range[swarm.N:2 * swarm.N, start_idx], 
                               y_range[2 * swarm.N:3 * swarm.N, start_idx], cmap=cmap,
                               c=cmap(np.mod(y_range[3 * swarm.N:, start_idx], 2 * np.pi) / (2 * np.pi)))
                               
            scats[idx].set_cmap(cmap)
            tts[idx] = ax.text(.01, .01, 0.1, s="t=%1.2f" % t_range[start_idx], transform=ax.transAxes)

            if lim is not None:
                ax.set_xlim(lim[:2])
                ax.set_ylim(lim[2:4])
                ax.set_ylim(lim[4:])

    def update(frame_idx):
        curr_idx = int(start_idx + (float(frame_idx) / num_frames) * nsteps)

        if swarms[0].n == 2:
            for scat, swarm, y_range in zip(scats, swarms, y_ranges):
                y = np.vstack((y_range[:swarm.N, curr_idx], 
                               y_range[swarm.N:2 * swarm.N, curr_idx])).T
                scat.set_offsets(y)
                #scat.set_cmap
                scat.set_color(cmap(np.mod(y_range[2 * swarm.N:, curr_idx], 2 * np.pi) / (2 * np.pi)))
                scat._facecolor3d = scat.get_facecolor()
                scat._edgecolor3d = scat.get_edgecolor()

        elif swarms[0].n == 3:
            for scat, swarm, y_range in zip(scats, swarms, y_ranges):
                scat._offsets3d = (y_range[:swarm.N, curr_idx], 
                                   y_range[swarm.N:2 * swarm.N, curr_idx], 
                                   y_range[2 * swarm.N:3 * swarm.N, curr_idx])
                scat.set_color(cmap(np.mod(y_range[3 * swarm.N:, curr_idx], 2 * np.pi) / (2 * np.pi)))
                scat._facecolor3d = scat.get_facecolor()
                scat._edgecolor3d = scat.get_edgecolor()

        for tt in tts:
            tt.set_text("t=%1.2f, %d %d" % (t_range[curr_idx], frame_idx, curr_idx))

        return scats + tts

    anim = FuncAnimation(fig, update, frames=num_frames, **kwargs)
    return anim


def crocker_plot(homology, i, ax=None, downsample=1, **kwargs):
    if ax is None:
        ax = plt.gca()

    im = ax.imshow(homology.grid(i, downsample), 
                   origin="lower", 
                   extent=(homology.t_uniform[0], homology.t_uniform[-1], homology.eps[0], homology.eps[-1]), 
                   aspect="auto", **kwargs)
    return im