import sympy as sp
from sympy import cos, sin, diff, pi, Matrix
from sympy.physics.vector import ReferenceFrame


# from sympy.printing.pycode import NumPyPrinter
def to_numpy(expr, R, i):
    return (
        NumPyPrinter()
        .doprint(expr.to_matrix(R)[i])
        .replace("numpy", "np")
        .replace("R_", "")
    )


def vector_gradient(u, R):
    return Matrix(
        [diff(u[i], var) for var in R.varlist for i in range(len(u))]
    ).reshape(3, 3)


def asym(r):
    return Matrix([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])


def asym_T(sigma):
    asym_sigma = sigma - sigma.T
    return Matrix([asym_sigma[2, 1], asym_sigma[0, 2], asym_sigma[1, 0]])


def matrix_divergence(sigma, R):
    x, y, z = R.varlist
    return Matrix(
        [diff(sigma, x).trace(), diff(sigma, y).trace(), diff(sigma, z).trace()]
    )


## -------------------------------------------------------------------##
def main():
    """
    The two-dimensional polynomial case
    """
    R = ReferenceFrame("R")
    x, y, z = R.varlist

    # define the displacement
    u_x = x  # sin(2 * pi * x) * sin(2 * pi * y)
    u_y = y  # sin(2 * pi * x) * sin(2 * pi * y)
    u = Matrix([u_x, u_y, 0])

    # define the rotation
    r_z = x  # sin(2 * pi * x) * sin(2 * pi * y)
    r = Matrix([0, 0, r_z])

    # Stress tensor with, mu = 0.5 and lambda = 0
    sigma = vector_gradient(u, R) + asym(r)
    print("sigma", sigma)
    print("grad u", vector_gradient(u, R))
    print("asym r", asym(r))

    # Micro stress tensor with, mu = 0.5 and lambda = 0
    omega = vector_gradient(r, R)
    print("omega", omega)

    # Compute the source term
    f_u = -matrix_divergence(sigma, R)
    print("f_u", f_u)

    f_r = asym_T(sigma) - matrix_divergence(omega, R)
    print("f_r", f_r)

    # check the asym operator
    print(asym_T(asym(r)) - 2 * r)

    # print("velocity\n", to_numpy(u, R, 0))
    # print("velocity\n", to_numpy(u, R, 1))

    # print("rotation\n", to_numpy(r, R, 2))

    # print("source\n", to_numpy(g, R, 0))
    # print("source\n", to_numpy(g, R, 1))

    # print("curl_r\n", to_numpy(curl_r, R, 0))
    # print("curl_r\n", to_numpy(curl_r, R, 1))

    # print(div_u)


## -------------------------------------------------------------------##

if __name__ == "__main__":
    main()
