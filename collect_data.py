import numpy as np
import os.path
import pathos.multiprocessing as mp
import dill as pickle
import inspect

from itertools import product

from swarm.swarmalator import Swarmalator
from swarm.homology import TimeDependentHomology
from swarm.batch_params import batch_params, fnames, num_sims
from swarm.utils import sup_bottleneck


def do_sim(swarm_params, solve_params, homology_params, fname="sim", redo=False, do_homology=False, dry_run=False):
    if not os.path.isfile(fname) or redo:
        swarm = Swarmalator(**swarm_params)
        sol = swarm.simulate(**solve_params)

        if do_homology:
            homology = TimeDependentHomology(swarm, sol, **homology_params)
        else:
            homology = None

        if not dry_run:
            with open(fname, "wb+") as f:
                pickle.dump({
                    "swarm": swarm,
                    "sol":   sol,
                    "homology":  homology,
                }, f, pickle.HIGHEST_PROTOCOL)
            
        return True 
    else:
        return False


def do_batch_sims(batch_idx, redo=False, do_homology=True, num_processes=6, block=True, dry_run=False):
    def batch(swarm_indices, solve_indices, homology_indices):
        swarm_params = {key: val[s_idx] for (key, val), s_idx in zip(batch_params[batch_idx]["swarm_params"].items(), swarm_indices) }
        solve_params = {key: val[v_idx] for (key, val), v_idx in zip(batch_params[batch_idx]["solve_params"].items(), solve_indices) }
        homology_params = {key: val[h_idx] for (key, val), h_idx in zip(batch_params[batch_idx]["homology_params"].items(), homology_indices) }

        fname = "../data/batch%s_swarm%s_solve%s_homology%s.pickle" % (batch_idx, 
                                                                       "".join([str(i) + "_" for i in swarm_indices]), 
                                                                       "".join([str(i) + "_" for i in solve_indices]),
                                                                       "".join([str(i) + "_" for i in homology_indices]))

        return do_sim(swarm_params, solve_params, homology_params, fname, redo, do_homology, dry_run) 

    iterator = product(product(*[range(len(val)) for key, val in batch_params[batch_idx]["swarm_params"].items()]), 
                       product(*[range(len(val)) for key, val in batch_params[batch_idx]["solve_params"].items()]), 
                       product(*[range(len(val)) for key, val in batch_params[batch_idx]["homology_params"].items()]))

    pool = mp.Pool(num_processes)
    pool.starmap(batch, iterator)

    if block:
        pool.close()
        pool.join()


def get_nearest_idx(nparray, x, tol=1e-3):
    idx = np.abs(np.array(nparray) - x).argmin()
    x_closest = nparray[idx]
    if np.abs(x - x_closest) < tol:
        return idx
    return -1


def get_batch_indices(params, bparams, sig, tol=1e-3):
    indices = list()
    invalid = False

    for key, val in params.items():
        if key not in bparams:
            if ((sig.parameters[key].default is not None and isinstance(val, float) and abs(val - sig.parameters[key].default) > tol)) \
                or val != sig.parameters[key].default:
                    invalid = True 
                    break

    if invalid:
        return None, invalid

    for key, val in bparams.items():
        idx = -1

        if key not in params:
            pval = sig.parameters[key].default
        else:
            pval = params[key]

        if isinstance(val, np.ndarray):
            idx = get_nearest_idx(val, pval, tol)
        elif key == "tspan":
            for i, tspan in enumerate(val):
                if tspan[1] >= pval[1]:
                    idx = i
                    break
        else:
            try:
                idx = val.index(pval)
            except ValueError:
                idx = -1

        if idx < 0:
            invalid = True 
            break
        else:
            indices.append(idx)

    return indices, invalid


def retrieve_sim(swarm_params, solve_params, homology_params, do_homology=False, do_solve=False, force_solve=False, return_fname=False, tol=1e-3):
    swarm_sig = inspect.signature(Swarmalator.__init__)
    solve_sig = inspect.signature(Swarmalator.simulate)
    homology_sig = inspect.signature(TimeDependentHomology.__init__)

    sol_exists = False 
    fname = None

    if not force_solve:
        for batch_idx, param_dict in enumerate(batch_params):
            swarm_indices, swarm_invalid = get_batch_indices(swarm_params, param_dict["swarm_params"], swarm_sig, tol)
            if swarm_invalid:
                continue
                
            solve_indices, solve_invalid = get_batch_indices(solve_params, param_dict["solve_params"], solve_sig, tol)
            if solve_invalid:
                continue

            if do_homology:
                hom_indices, hom_invalid = get_batch_indices(homology_params, param_dict["homology_params"], homology_sig, tol)
                if hom_invalid:
                    continue
            else:
                hom_indices = [0] * len(param_dict["homology_params"].keys())

            sol_exists = True
            fname = "../data/batch%s_swarm%s_solve%s_homology%s.pickle" % (batch_idx, 
                "".join([str(i) + "_" for i in swarm_indices]), 
                "".join([str(i) + "_" for i in solve_indices]),
                "".join([str(i) + "_" for i in hom_indices]))
            break

    if sol_exists and not force_solve:
        try:
            with open(fname, "rb") as f:
                data = pickle.load(f)
                swarm, sol, homology = data["swarm"], data["sol"], data["homology"]
        except (OSError, IOError):
            print("Cannot find file: %s" % fname)
            swarm, sol, homology = None, None, None

    elif (do_solve and not sol_exists) or force_solve:
        swarm = Swarmalator(**swarm_params)
        sol = swarm.simulate(**solve_params)
        if not do_homology:
            homology = None
        else:
            homology = TimeDependentHomology(swarm, sol, **homology_params)

    else:
        swarm, sol, homology = None, None, None
    

    if return_fname:
        return swarm, sol, homology, fname
    else:
        return swarm, sol, homology


def compute_distances(num_processes=8, k=0, downsample=5, batch_idx=0, redo=False, dry_run=False):
    homologies = list()

    for fname in fnames[batch_idx]:
        with open(fname, "rb") as fd:
            data = pickle.load(fd)
            homologies.append(data["homology"])

    def do_dmat(i):
        newfname = fnames[batch_idx][i].replace("pickle", "H%d_distance.pickle" % k)
        if os.path.isfile(newfname) and not redo:
            try:
                with open(newfname, "rb") as fd:
                    _ = pickle.load(fd)
                return False
            except EOFError:
                pass

        if not dry_run:
            distances = dict()
            with open(newfname, "wb+") as fd:
                #TODO: Fix this
                #for j in range(i):
                #    distances[fnames[batch_idx][j]] = sup_bottleneck(homologies[j], homologies[i], k, downsample)
                #pickle.dump(distances, fd, pickle.HIGHEST_PROTOCOL)  
                pass

        return True

    pool = mp.Pool(num_processes)
    result = pool.map(do_dmat, range(num_sims[batch_idx]))
    pool.close()
    pool.join()

    return result

"""
def do_pool(num_processes=8, k=0, downsample=5):
    def do_dmat(i):
        newfname = fnames[i].replace("pickle", "H%d_distance.pickle" % k)
        distances = dict()
        
        if os.path.isfile(newfname):
            try:
                with open(newfname, "rb") as fd:
                    data = pickle.load(fd)
                return True
            except EOFError:
                pass
            
        with open(newfname, "wb+") as fd:
            for j in range(i):
                distances[fnames[j]] = sm.sup_bottleneck(homologies[j], homologies[i], k, downsample)
            pickle.dump(distances, fd, pickle.HIGHEST_PROTOCOL)  
            
        return False
            
    pool = mp.Pool(num_processes)
    result = pool.map(do_dmat, range(num_sims))
    pool.close()
    pool.join()
    
    return result
"""

def retrieve_dmat(k, batch_idx=0):
    dmat = np.zeros((num_sims[batch_idx], num_sims[batch_idx],))
    
    for i, fname in enumerate(fnames[batch_idx]):
        newfname = fname.replace("pickle", "H%d_distance.pickle" % k)
        
        with open(newfname, "rb") as fd:
            data = pickle.load(fd)
            for nfname, dist in data.items():
                j = fnames[batch_idx].index(nfname)
                dmat[i, j] = dist
                
    return dmat + dmat.T