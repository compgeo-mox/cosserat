import sympy as sp
from sympy.physics.vector import ReferenceFrame

import sys

sys.path.append("./src")

from exact_operators import (
    cosserat_compute_all,
    cosserate_as_lambda,
    elasticity_compute_all,
)


def cosserat_exact_2d(mu_s, lambda_s, mu_w, lambda_w=0):
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
    sigma, w, f_u, f_r = cosserat_compute_all(
        dim, u, r, mu_s, lambda_s, mu_w, lambda_w, R
    )

    # lambdify the exact solution
    return cosserate_as_lambda(dim, sigma, w, u, r, f_u, f_r, R)


def cosserat_exact_3d(mu_s, lambda_s, mu_w, lambda_w):
    dim = 3
    R = ReferenceFrame("R")
    x, y, z = R.varlist

    # define the displacement
    u_x = x * (1 - x) * y * (1 - y) * z * (1 - z)
    u_y = u_x
    u_z = u_y
    u = sp.Matrix([u_x, u_y, u_z])

    # define the rotation
    r_x = u_x
    r_y = r_x
    r_z = r_x
    r = sp.Matrix([r_x, r_y, r_z])

    # compute the stress and micro stress tensor and the source terms
    sigma, w, f_u, f_r = cosserat_compute_all(
        dim, u, r, mu_s, lambda_s, mu_w, lambda_w, R
    )

    # lambdify the exact solution
    return cosserate_as_lambda(dim, sigma, w, u, r, f_u, f_r, R)


def elasticity_exact_3d(mu, lambda_):
    dim = 3
    R = ReferenceFrame("R")
    x, y, z = R.varlist

    # define the displacement
    u_x = x * (1 - x) * y * (1 - y) * z * (1 - z)
    u_y = u_x
    u_z = u_y
    u = sp.Matrix([u_x, u_y, u_z])

    # define the stress
    sigma = sp.Matrix(
        [
            [u_x, 0, 0.5 * u_x],
            [0, u_x, 0],
            [0.5 * u_x, 0, u_x],
        ]
    )

    r = u

    f_s, f_u = elasticity_compute_all(dim, sigma, u, r, mu, lambda_, R)

    w = sp.Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    f_r = sp.Matrix([0, 0, 0])

    # lambdify the exact solution
    return cosserate_as_lambda(dim, sigma, w, u, r, f_s, f_u, f_r, R)
