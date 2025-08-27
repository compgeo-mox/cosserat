import os, sys
import numpy as np

sys.path.append("./src")

from functions import *
from solver import SolverBDM1_L1


if __name__ == "__main__":
    np.set_printoptions(precision=2, linewidth=9999)

    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    if not os.path.exists(folder):
        os.makedirs(folder)

    solver_class = SolverBDM1_L1
    dims = [3]
    alphas_betas = [(0, 1), (1, 0)]

    run = {2: run_2d, 3: run_3d}
    for dim in dims:
        for alpha, beta in alphas_betas:
            d_setup = setup(dim, alpha, beta)

            name = (
                "bdm1_l1_" + str(dim) + "d_alpha_" + str(alpha) + "_beta_" + str(beta)
            )

            # Run the lumped case
            file_name = name + "_lump.tex"
            print("solve " + file_name)
            # run[dim](solve_lumped, folder, file_name, d_setup, solver_class)

            # Run the non-lumped case
            file_name = name + ".tex"
            print("solve " + file_name)
            run[dim](solve_not_lumped, folder, file_name, d_setup, solver_class)
