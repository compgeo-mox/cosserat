import os, sys
import numpy as np

import porepy as pp

sys.path.append("./src")

from functions import make_summary
from analytical_solutions import cosserat_exact_3d
from solver import SolverBDM1_P0


def main(mesh_size, folder):
    key = "cosserat"

    # Get the exact solution
    mu_s, mu_sc, lambda_s = 0.5, 0.25, 1
    mu_w, mu_wc, lambda_w = 0.5, 0.25, 1
    data_pb = cosserat_exact_3d(mu_s, mu_sc, lambda_s, mu_w, mu_wc, lambda_w)
    data = {pp.PARAMETERS: {key: {"mu": mu_s, "lambda": lambda_s, "mu_c": mu_sc}}}

    dim = 3
    solver = SolverBDM1_P0(dim, key)
    solver.create_grid(mesh_size, folder)
    solver.create_family()

    return solver.solve_problem(data, data_pb)


if __name__ == "__main__":
    np.set_printoptions(precision=2, linewidth=9999)

    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    if not os.path.exists(folder):
        os.makedirs(folder)

    mesh_size = [0.4, 0.3]  # , 0.2, 0.1, 0.05, 0.025]
    errs = np.vstack([main(h, folder) for h in mesh_size])
    errs_latex = make_summary(errs)

    # Write to a file
    with open(folder + "/case4_not_lump.tex", "w") as file:
        file.write(errs_latex)
