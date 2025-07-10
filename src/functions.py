import numpy as np
import porepy as pp
import copy

import strong_solution_cosserat_elasticity_example_3 as ss


def solve_lumped(dim, mesh_size, folder, setup, solver_class):
    print("solve_lumped for mesh_size", mesh_size)
    data, data_pb, key = setup

    solver = solver_class(dim, key)
    solver.create_grid(mesh_size, folder)
    solver.create_family()
    return solver.solve_problem_lumped(copy.deepcopy(data), data_pb)


def solve_not_lumped(dim, mesh_size, folder, setup, solver_class):
    print("solve_not_lumped for mesh_size", mesh_size)
    data, data_pb, key = setup

    solver = solver_class(dim, key)
    solver.create_grid(mesh_size, folder)
    solver.create_family()

    return solver.solve_problem(copy.deepcopy(data), data_pb)


def run_2d(func, folder, file_name, setup, solver_class):
    dim = 2
    mesh_size = np.power(2.0, -np.arange(3, 3 + 5))
    errs = np.vstack([func(dim, h, folder, setup, solver_class) for h in mesh_size])
    print(errs)
    errs_latex = make_summary(errs)

    # Write to a file
    with open(folder + "/" + file_name, "w") as file:
        file.write(errs_latex)


def run_3d(func, folder, file_name, setup, solver_class):
    dim = 3
    mesh_size = [0.4, 0.3]  # , 0.2, 0.1]
    errs = np.vstack([func(dim, h, folder, setup, solver_class) for h in mesh_size])
    print(errs)
    errs_latex = make_summary(errs)

    # Write to a file
    with open(folder + "/" + file_name, "w") as file:
        file.write(errs_latex)


def setup(dim):
    key = "cosserat"

    # return the exact solution and related rhs
    lambda_, mu, kappa = 1, 1, 0.1

    param = {
        "lambda_s": lambda_,
        "mu_s": mu,
        "kappa_s": kappa,
        "lambda_o": lambda_,
        "mu_o": mu,
        "kappa_o": kappa,
    }

    def stress(pt):
        s = ss.stress(param, dim)(*pt)
        if dim == 2:
            s = np.hstack((s, np.zeros((2, 1))))
        return s

    def couple_stress_scaled(pt):
        s = ss.couple_stress_scaled(param, dim)(*pt)
        if dim == 2:
            s = np.hstack((s[0], 0))
        return s

    def rotation(pt):
        r = ss.rotation(param, dim)(*pt)
        if dim == 2:
            r = r[0]
        return r

    def rhs_scaled_u(pt):
        rhs = ss.rhs_scaled(param, dim)(*pt)
        if dim == 2:
            rhs = rhs[:2]
        return rhs

    def rhs_scaled_r(pt):
        rhs = ss.rhs_scaled(param, dim)(*pt)
        if dim == 2:
            rhs = rhs[-1]
        return rhs

    data_pb = {
        "s_ex": stress,
        "w_ex": couple_stress_scaled,
        "u_ex": lambda pt: ss.displacement(param, dim)(*pt),
        "r_ex": rotation,
        "f_r": rhs_scaled_r,
        "f_u": rhs_scaled_u,
        "ell": lambda pt: ss.gamma_s(dim)(*pt),
    }

    data = {pp.PARAMETERS: {key: {"mu": mu, "lambda": lambda_, "mu_c": kappa}}}

    return data, data_pb, key


def order(error, diam):
    return np.log(error[:-1] / error[1:]) / np.log(diam[:-1] / diam[1:])


def array_to_latex(arr):
    def disp(num, pos):
        if pos % 2 == 0 and pos != 0:
            return "{:.2f}".format(num)
        else:
            return "{:.2e}".format(num)

    latex_str = (
        "& "
        + " \\\\ \n& ".join(
            [" & ".join(disp(num, pos) for pos, num in enumerate(row)) for row in arr]
        )
        + " \\\\"
    )
    return latex_str.replace("-1.00", "-")


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
