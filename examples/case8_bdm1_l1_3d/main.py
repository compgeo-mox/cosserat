import os
import sys
import numpy as np


sys.path.append("./src")

from functions import *
from solver import SolverBDM1_L1

if __name__ == "__main__":
    np.set_printoptions(precision=2, linewidth=9999)

    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    if not os.path.exists(folder):
        os.makedirs(folder)

    data_setup = setup()

    solver_class = SolverBDM1_L1
    # Run the lumped case
    print("solve lumped case8")
    run_3d(solve_lumped, folder, "case8_lumped.tex", data_setup, solver_class)
    # Run the non-lumped case
    print("solve not lumped case8")
    run_3d(solve_not_lumped, folder, "case8_not_lump.tex", data_setup, solver_class)
