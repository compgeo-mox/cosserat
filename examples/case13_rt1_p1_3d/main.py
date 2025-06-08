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

    # Open a log file
    log_file = open(os.path.join(folder, "case13_output.log"), "w")

    # Redirect stdout and stderr
    # sys.stdout = log_file
    # sys.stderr = log_file

    data_setup = setup_3d()

    solver_class = SolverRT1_P1
    # Run the lumped case
    print("solve lumped case13")
    run_3d(solve_lumped, folder, "case13_lump.tex", data_setup, solver_class)
    # Run the non-lumped case
    # print("solve not lumped case13")
    # run_3d(solve_not_lumped, folder, "case13_not_lump.tex", data_setup, solver_class)

    log_file.close()
