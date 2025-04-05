import os, sys
import numpy as np

import porepy as pp

sys.path.append("./src")

from functions import make_summary
from analytical_solutions import cosserat_exact_2d
from solver import SolverBDM1_P0


def setup():
    key = "cosserat"

    # return the exact solution and related rhs
    mu_s, mu_sc, lambda_s = 0.5, 0.25, 1
    mu_w = 0.5
    data_pb = cosserat_exact_2d(mu_s, mu_sc, lambda_s, mu_w)
    data = {pp.PARAMETERS: {key: {"mu": mu_s, "lambda": lambda_s, "mu_c": mu_sc}}}

    return data, data_pb, key


def main_lumped(mesh_size, folder):
    data, data_pb, key = setup()

    dim = 2
    solver = SolverBDM1_P0(dim, key)
    solver.create_grid(mesh_size, folder)
    solver.create_family()

    return solver.solve_problem_lumped(data, data_pb)


def main(mesh_size, folder):
    data, data_pb, key = setup()

    dim = 2
    solver = SolverBDM1_P0(dim, key)
    solver.create_grid(mesh_size, folder)
    solver.create_family()

    return solver.solve_problem(data, data_pb)


def run(func, folder, file_name):
    mesh_size = np.power(2.0, -np.arange(3, 3 + 5))
    errs = np.vstack([func(h, folder) for h in mesh_size])
    errs_latex = make_summary(errs)

    # Write to a file
    with open(folder + "/" + file_name, "w") as file:
        file.write(errs_latex)


if __name__ == "__main__":
    np.set_printoptions(precision=2, linewidth=9999)

    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Run the lumped case
    run(main_lumped, folder, "case2_lump.tex")
    # Run the non-lumped case
    run(main, folder, "case2_not_lump.tex")
