import numpy as np
from itertools import product

batch_params = [
    {
        "swarm_params": {
            "J1": np.arange(0, 1.51, 0.1),
            "K": np.arange(-1.0, 1.01, 0.1),
            "N": [100], 
            "n": [2],
            "option": ["vanilla"],
        }, 
        "solve_params": {
            "noise": [False], 
            "tspan": [[0, 500]],
            "v": [0], 
            "omega": [0]
        }, 
        "homology_params":{
            "t_start": [450], 
            "dt": [1.0],
            "n_perm": [50]
        }
    }, 

    {
        "swarm_params": {
            "J1": np.arange(0, 1.51, 0.1),
            "K": np.arange(-1.0, 1.01, 0.1),
            "N": [100], 
            "n": [2],
            "option": ["vanilla"],
        }, 
        "solve_params": {
            "noise": [False], 
            "tspan": [[0, 500]],
            "thetavar": [2 * np.pi],
            "v": [0], 
            "omega": [0]
        }, 
        "homology_params":{
            "t_start": [450], 
            "dt": [1.0],
            "n_perm": [50]
        }
    }, 
    
    {
        "swarm_params": {
            "J1": np.arange(0, 1.51, 0.1),
            "K": np.arange(-1.0, 1.01, 0.1),
            "N": [100], 
            "n": [2],
            "option": ["vanilla"],
        }, 
        "solve_params": {
            "noise": [False], 
            "tspan": [[0, 1000]],
            "thetavar": [2 * np.pi],
            "v": [0], 
            "omega": [0]
        }, 
        "homology_params":{
            "t_start": [950], 
            "dt": [1.0],
            "n_perm": [50]
        }
    }, 

    {
        "swarm_params": {
            "J1": np.arange(0, 1.51, 0.1),
            "K": np.arange(-1.0, 1.01, 0.1),
            "N": [100], 
            "n": [2],
            "option": ["vanilla"],
        }, 
        "solve_params": {
            "noise": [True], 
            "tspan": [[0, 500]], 
            "sigma": np.arange(0.01, 1.01, 0.1),
            "v": [0], 
            "omega": [0]
        }, 
        "homology_params":{
            "t_start": [450], 
            "dt": [0.5],
            "n_perm": [50]
        }
    }, 

    {
        "swarm_params": {
            "J1": np.arange(0, 1.51, 0.1),
            "K": np.arange(-1.0, 1.01, 0.1),
            "N": [100], 
            "n": [3],
            "option": ["vanilla"],
        }, 
        "solve_params": {
            "noise": [True], 
            "tspan": [[0, 500]], 
            "sigma": np.arange(0.01, 1.01, 0.1),
            "v": [0], 
            "omega": [0]
        }, 
        "homology_params":{
            "t_start": [450], 
            "dt": [0.5],
            "n_perm": [50]
        }
    }, 
]

fnames = list()
num_sims = list()

for batch_idx, param_dict in enumerate(batch_params):
    iterator = product(product(*[range(len(val)) for key, val in param_dict["swarm_params"].items()]), 
                       product(*[range(len(val)) for key, val in param_dict["solve_params"].items()]), 
                       product(*[range(len(val)) for key, val in param_dict["homology_params"].items()]))

    batch_fnames = list()

    for it1, it2, it3 in iterator:
        swarm_str = "".join(["%d_" % idx for idx in it1])
        sim_str = "".join(["%d_" % idx for idx in it2])
        hom_str = "".join(["%d_" % idx for idx in it3])
        batch_fnames.append("../data/batch%s_swarm%s_solve%s_homology%s.pickle" % (batch_idx, swarm_str, sim_str, hom_str))

    fnames.append(batch_fnames)
    num_sims.append(len(batch_fnames))