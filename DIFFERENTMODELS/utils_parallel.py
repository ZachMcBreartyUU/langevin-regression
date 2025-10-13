### Utility functions for the various models, modifified from LR's original utils
from time import time
import os

import numpy as np
import multiprocessing as mp

NUM_CPUS = int(os.environ.get("SLURM_NTASKS_PER_NODE", default=1))
print(f"{NUM_CPUS=}")


def SSR_loop_parallel(opt_fun, params):
    """
    Stepwise sparse regression: general function for a given optimization problem
       opt_fun should take the parameters and return coefficients and cost

    Requires a list of drift and diffusion expressions,
        (although these are just passed to the opt_fun)
    """
    A_len = len(params["A_expr"])
    n_terms = A_len + len(params["C_expr"])
    minimum_Xis = np.zeros((n_terms - 1, n_terms), dtype=params["Xi0"].dtype)
    minimum_Vs = np.full((n_terms - 1), np.inf)

    minimum_Xis[0], minimum_Vs[0] = opt_fun(params)

    active = np.array([i for i in range(n_terms)])
    active_history = [active]
    for k in range(1, n_terms - 1):
        start = time()
        params_list = []
        for j in range(len(active)):
            tmp_active = active.copy()
            tmp_active = np.delete(tmp_active, j)

            reduced_params = params.copy()
            f_active = tmp_active[tmp_active < A_len]
            a_active = tmp_active[tmp_active >= A_len] - A_len
            reduced_params["A_expr"] = params["A_expr"][f_active]
            reduced_params["C_expr"] = params["C_expr"][a_active]
            reduced_params["lib_A"] = params["lib_A"][f_active]
            reduced_params["lib_C"] = params["lib_C"][a_active]
            reduced_params["Xi0"] = params["Xi0"][tmp_active]
            params_list.append(reduced_params)
        with mp.Pool(NUM_CPUS) as p:
            results = p.map(opt_fun, params_list)
        Vs = []
        Xis = []
        for Xi, V in results:
            Xis.append(Xi)
            Vs.append(V)
        min_idx = np.argmin(Vs)
        min_V = Vs[min_idx]
        min_Xi = Xis[min_idx]
        active = np.delete(active, min_idx)
        minimum_Vs[k] = min_V
        minimum_Xis[k, active] = min_Xi
        active_history.append(active)
        end = time()
        print(f"Regression {k} took {end-start} seconds")
        print(f"Cost = {min_V} from Xi = {min_Xi}")

        f_active = active[active < A_len]
        a_active = active[active >= A_len] - A_len
        print(f"Active f: {params["A_expr"][f_active]}")
        print(f"Active a: {params["C_expr"][a_active]}", flush=True)

        params["Xi0"][active] = min_Xi

    return minimum_Xis, minimum_Vs, active_history
