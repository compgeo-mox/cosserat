import numpy as np
import sympy as sp

mu_s = mu_w = 1
mu_cs = mu_cw = 0.1
lam_s = lam_w = 1


def funcify(f, x, dim, ravel=True):
    lam_f = sp.lambdify(x, f, "numpy")
    if dim == 2:
        if ravel:
            return lambda x, y, z: lam_f(x, y).ravel()
        else:
            return lambda x, y, z: lam_f(x, y)

    else:  # dim == 3
        if ravel:
            return lambda x, y, z: lam_f(x, y, z).ravel()
        else:
            return lam_f


def compile_funcs(dim, alpha=0):
    func_dict = {}

    if dim == 2:
        x = sp.symbols("x, y")
    else:
        x = sp.symbols("x, y, z")

    ell_hat = alpha + (1 - alpha) * sp.Min(1, sp.Max(0, 3 * x[0] - 1))
    ell = sp.sin(sp.pi / 2 * ell_hat) ** 2
    func_dict["ell"] = funcify(ell, x, dim, False)

    if dim == 2:
        # displacement
        u = sp.Matrix(
            [
                x[1] * (1 - x[1]) * sp.sin(sp.pi * x[0]),
                x[0] * (1 - x[0]) * sp.sin(sp.pi * x[1]),
            ]
        )
    else:
        # displacement
        u = sp.Matrix(
            [
                x[1] * (1 - x[1]) * x[2] * (1 - x[2]) * sp.sin(sp.pi * x[0]),
                x[2] * (1 - x[2]) * x[0] * (1 - x[0]) * sp.sin(sp.pi * x[1]),
                x[0] * (1 - x[0]) * x[1] * (1 - x[1]) * sp.sin(sp.pi * x[2]),
            ]
        )
    func_dict["u"] = funcify(u, x, dim)

    grad_u = sp.Matrix([[sp.diff(u_i, x_j) for x_j in x] for u_i in u])
    div_u = sum([sp.diff(u_i, x_i) for (u_i, x_i) in zip(u, x)])

    # rotation
    if dim == 2:
        r = sp.sin(sp.pi * x[0]) * sp.sin(sp.pi * x[1])
        grad_r = sp.Matrix([sp.diff(r, x_i) for x_i in x])
        asym_r = sp.Matrix(
            [
                [0, -r],
                [r, 0],
            ]
        )
    else:
        r = sp.Matrix(
            [
                x[0] * (1 - x[0]) * sp.sin(sp.pi * x[1]) * sp.sin(sp.pi * x[2]),
                x[1] * (1 - x[1]) * sp.sin(sp.pi * x[2]) * sp.sin(sp.pi * x[0]),
                x[2] * (1 - x[2]) * sp.sin(sp.pi * x[0]) * sp.sin(sp.pi * x[1]),
            ]
        )
        grad_r = sp.Matrix([[sp.diff(r_i, x_j) for x_j in x] for r_i in r])
        asym_r = sp.Matrix(
            [
                [0, -r[2], r[1]],
                [r[2], 0, -r[0]],
                [-r[1], r[0], 0],
            ]
        )
        div_r = sum([sp.diff(r_i, x_i) for (r_i, x_i) in zip(r, x)])

    if dim == 2:
        func_dict["r"] = funcify(r, x, dim, False)
    else:
        func_dict["r"] = funcify(r, x, dim)

    # couple stress
    if dim == 2:
        w = 2 * mu_w * ell * grad_r
        div_ell_w = sum([sp.diff(ell * w_i, x_i) for (w_i, x_i) in zip(w, x)])
        func_dict["w"] = funcify(w, x, dim)

    else:
        tauw = ell * grad_r
        sym_tauw = (tauw + tauw.T) / 2
        skw_tauw = (tauw - tauw.T) / 2
        tr_tauw_I = ell * div_r * sp.Matrix(np.eye(dim))

        w = 2 * mu_w * sym_tauw + 2 * mu_cw * skw_tauw + lam_w * tr_tauw_I
        div_ell_w = sp.Matrix(
            [
                sum([sp.diff(ell * w_i, x_i) for (w_i, x_i) in zip(w[i, :], x)])
                for i in range(3)
            ]
        )
        func_dict["w"] = funcify(w, x, dim, False)

    # Cauchy stress
    tau = grad_u + asym_r
    sym_tau = (tau + tau.T) / 2
    skw_tau = (tau - tau.T) / 2
    tr_tau_I = div_u * sp.Matrix(np.eye(dim))

    sigma = 2 * mu_s * sym_tau + 2 * mu_cs * skw_tau + lam_s * tr_tau_I

    if dim == 2:
        asym_sigma = sigma[1, 0] - sigma[0, 1]
    else:
        asym_sigma = sp.Matrix(
            [
                sigma[2, 1] - sigma[1, 2],
                sigma[0, 2] - sigma[2, 0],
                sigma[1, 0] - sigma[0, 1],
            ]
        )

    div_sigma = sp.Matrix(
        [
            sum([sp.diff(s_ij, x_i) for (s_ij, x_i) in zip(sigma[i, :], x)])
            for i in range(dim)
        ]
    )
    func_dict["s"] = funcify(sigma, x, dim, False)

    # Right-hand sides
    rhs_u = -div_sigma
    rhs_r = asym_sigma - div_ell_w

    func_dict["rhs_u"] = funcify(rhs_u, x, dim)
    if dim == 2:
        func_dict["rhs_r"] = funcify(rhs_r, x, dim, False)
    else:
        func_dict["rhs_r"] = funcify(rhs_r, x, dim)

    return func_dict


# Collect all functions in a dictionary
funcs = {}
for dim in [2, 3]:
    for alpha in [0, 1]:
        funcs[dim, alpha] = compile_funcs(dim, alpha)


def stress(param, dim):
    alpha = param["alpha"]
    return funcs[dim, alpha]["s"]


def couple_stress_scaled(param, dim):
    alpha = param["alpha"]
    return funcs[dim, alpha]["w"]


def displacement(param, dim):
    alpha = param["alpha"]
    return funcs[dim, alpha]["u"]


def rotation(param, dim):
    alpha = param["alpha"]
    return funcs[dim, alpha]["r"]


def rhs_scaled_u(param, dim):
    alpha = param["alpha"]
    return funcs[dim, alpha]["rhs_u"]


def rhs_scaled_r(param, dim):
    alpha = param["alpha"]
    return funcs[dim, alpha]["rhs_r"]


def gamma_s(param, dim):
    alpha = param["alpha"]
    return funcs[dim, alpha]["ell"]
