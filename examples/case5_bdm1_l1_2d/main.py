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

    dim = 2
    d_setup = setup(dim)

    solver_class = SolverBDM1_L1
    # Run the lumped case
    print("solve lumped bdm1_l1_2d")
    run_2d(solve_lumped, folder, "bdm1_l1_2d_lump.tex", d_setup, solver_class)
    # Run the non-lumped case
    print("solve not lumped bdm1_l1_2d")
    run_2d(solve_not_lumped, folder, "bdm1_l1_2d_not_lump.tex", d_setup, solver_class)
