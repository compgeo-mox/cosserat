import os, sys
import numpy as np

sys.path.append("./src")

from functions import *
from solver import SolverRT1_P1


if __name__ == "__main__":
    np.set_printoptions(precision=2, linewidth=9999)

    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    if not os.path.exists(folder):
        os.makedirs(folder)

    solver_class = SolverRT1_P1
    dims = [2, 3]

    run = {2: run_2d, 3: run_3d}
    for dim in dims:
        d_setup = setup(dim)

        # Run the lumped case
        file_name = "rt1_p1_" + str(dim) + "d_lump.tex"
        print("solve " + file_name)
        run[dim](solve_lumped, folder, file_name, d_setup, solver_class)

        # Run the non-lumped case
        file_name = "rt1_p1_" + str(dim) + "d.tex"
        print("solve " + file_name)
        run[dim](solve_not_lumped, folder, file_name, d_setup, solver_class)
