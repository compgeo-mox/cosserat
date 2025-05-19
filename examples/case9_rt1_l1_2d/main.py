import os, sys
import numpy as np

sys.path.append("./src")

from functions import *
from solver import SolverRT1_L1


if __name__ == "__main__":
    np.set_printoptions(precision=2, linewidth=9999)

    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Open a log file
    log_file = open(os.path.join(folder, "case9_output.log"), "w")

    # Redirect stdout and stderr
    sys.stdout = log_file
    sys.stderr = log_file

    data_setup = setup_2d()

    solver_class = SolverRT1_L1
    # Run the lumped case
    # print("solve lumped case9")
    # run_2d(solve_lumped, folder, "case9_lump.tex", data_setup, solver_class)
    # Run the non-lumped case
    print("solve not lumped case9")
    run_2d(solve_not_lumped, folder, "case9_not_lump.tex", data_setup, solver_class)

    log_file.close()
