import numpy as np
import porepy as pp
from analytical_solutions import cosserat_exact_2d, cosserat_exact_3d


def solve_lumped(dim, mesh_size, folder, setup, solver_class):
    print("solve_lumped for mesh_size", mesh_size)
    data, data_pb, key = setup

    solver = solver_class(dim, key)
    solver.create_grid(mesh_size, folder)
    solver.create_family()

    return solver.solve_problem_lumped(data, data_pb)


def solve_not_lumped(dim, mesh_size, folder, setup, solver_class):
    print("solve_not_lumped for mesh_size", mesh_size)
    data, data_pb, key = setup

    solver = solver_class(dim, key)
    solver.create_grid(mesh_size, folder)
    solver.create_family()

    return solver.solve_problem(data, data_pb)


def run_2d(func, folder, file_name, setup, solver_class):
    dim = 2
    mesh_size = np.power(2.0, -np.arange(3, 3 + 5))
    errs = np.vstack([func(dim, h, folder, setup, solver_class) for h in mesh_size])
    errs_latex = make_summary(errs)

    # Write to a file
    with open(folder + "/" + file_name, "w") as file:
        file.write(errs_latex)


def run_3d(func, folder, file_name, setup, solver_class):
    dim = 3
    mesh_size = [0.4, 0.3, 0.2, 0.1, 0.05]
    errs = np.vstack([func(dim, h, folder, setup, solver_class) for h in mesh_size])
    errs_latex = make_summary(errs)

    # Write to a file
    with open(folder + "/" + file_name, "w") as file:
        file.write(errs_latex)


def setup_2d():
    key = "cosserat"

    # return the exact solution and related rhs
    mu_s, mu_sc, lambda_s = 0.5, 0.25, 1
    mu_w = 0.5
    data_pb = cosserat_exact_2d(mu_s, mu_sc, lambda_s, mu_w)
    data = {pp.PARAMETERS: {key: {"mu": mu_s, "lambda": lambda_s, "mu_c": mu_sc}}}

    return data, data_pb, key


def setup_3d():
    key = "cosserat"

    # return the exact solution and related rhs
    mu_s, mu_sc, lambda_s = 0.5, 0.25, 1
    mu_w, mu_wc, lambda_w = 0.5, 0.25, 1
    data_pb = cosserat_exact_3d(mu_s, mu_sc, lambda_s, mu_w, mu_wc, lambda_w)
    data = {pp.PARAMETERS: {key: {"mu": mu_s, "lambda": lambda_s, "mu_c": mu_sc}}}

    return data, data_pb, key


def order(error, diam):
    return np.log(error[:-1] / error[1:]) / np.log(diam[:-1] / diam[1:])


def array_to_latex(arr):
    latex_str = "\\begin{table}[h]\n\\centering\n"
    latex_str += "\\begin{tabular}{" + "|c" * arr.shape[1] + "|}\n\\hline\n"

    intestation = [
        "$h$",
        "$err_\sigma$",
        "ord",
        "$err_w$",
        "ord",
        "$err_u$",
        "ord",
        "$err_r$",
        "ord",
        "$dofs_u$",
        "$dofs_r$",
    ]

    formatted_rows = [" & ".join("{:.2e}".format(num) for num in row) for row in arr]

    latex_str += " & ".join(intestation)
    latex_str += " \\\\\n\\hline\n"
    latex_str += " \\\\\n\\hline\n".join(formatted_rows)
    latex_str += " \\\\\n\\hline\n\\end{tabular}\n\\end{table}"

    return latex_str.replace("-1.00e+00", "-")


def make_summary(errs):
    order_sigma = order(errs[:, 1], errs[:, 0])
    order_w = order(errs[:, 2], errs[:, 0])
    order_u = order(errs[:, 3], errs[:, 0])
    order_r = order(errs[:, 4], errs[:, 0])

    # reorder the arrays for the latex table
    order_sigma = np.hstack([-1, order_sigma])
    order_w = np.hstack([-1, order_w])
    order_u = np.hstack([-1, order_u])
    order_r = np.hstack([-1, order_r])

    errs = np.insert(errs, 2, order_sigma, axis=1)
    errs = np.insert(errs, 4, order_w, axis=1)
    errs = np.insert(errs, 6, order_u, axis=1)
    errs = np.insert(errs, 8, order_r, axis=1)

    return array_to_latex(errs)
