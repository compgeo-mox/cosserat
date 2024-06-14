import sympy as sp
from sympy.physics.vector import ReferenceFrame

import sys

sys.path.append("./src")

from exact_operators import compute_all, make_as_lambda


def exact_sol_2d(mu_s, lambda_s, mu_w, lambda_w=0):
    dim = 2
    R = ReferenceFrame("R")
    x, y, _ = R.varlist

    # define the displacement
    u_x = sp.sin(2 * sp.pi * x) * sp.sin(2 * sp.pi * y)
    u_y = u_x
    u = sp.Matrix([u_x, u_y, 0])

    # define the rotation
    r_z = u_x
    r = sp.Matrix([0, 0, r_z])

    # compute the stress and micro stress tensor and the source terms
    sigma, w, f_u, f_r = compute_all(dim, u, r, mu_s, lambda_s, mu_w, lambda_w, R)

    # lambdify the exact solution
    return make_as_lambda(dim, sigma, w, u, r, f_u, f_r, R)


def exact_sol_3d(mu_s, lambda_s, mu_w, lambda_w):
    dim = 3
    R = ReferenceFrame("R")
    x, y, z = R.varlist

    # define the displacement
    u_x = x * (1 - x) * y * (1 - y) * z * (1 - z)
    # sp.sin(2 * sp.pi * x) * sp.sin(2 * sp.pi * y) * sp.sin(2 * sp.pi * z)
    u_y = u_x
    u_z = u_y
    u = sp.Matrix([u_x, u_y, u_z])

    # define the rotation
    r_x = u_x
    r_y = r_x
    r_z = r_x
    r = sp.Matrix([r_x, r_y, r_z])

    # compute the stress and micro stress tensor and the source terms
    sigma, w, f_u, f_r = compute_all(dim, u, r, mu_s, lambda_s, mu_w, lambda_w, R)

    # lambdify the exact solution
    return make_as_lambda(dim, sigma, w, u, r, f_u, f_r, R)
