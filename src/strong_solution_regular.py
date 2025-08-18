import numpy as np
import sympy as sp


mu_s = mu_w = 1
mu_cs = mu_cw = 0.1
lam_s = lam_w = 1


def funcify(f, x, dim, ravel=True):
    lam_f = sp.lambdify(x, f, "numpy")
    if ravel:
        return lambda x: lam_f(*x[:dim]).ravel()
    else:
        return lambda x: lam_f(*x[:dim])


def compile_funcs(dim, alpha=0):
    func_dict = {}

    if dim == 2:
        x = sp.symbols("x, y")

        ell = alpha + (1 - alpha) * sp.Min(1, sp.Max(0, 3 * sp.Max(x[0], x[1]) - 1))

        # displacement
        u = sp.Matrix(
            [
                x[1] * (1 - x[1]) * sp.sin(sp.pi * x[0]),
                x[0] * (1 - x[0]) * sp.sin(sp.pi * x[1]),
            ]
        )
    else:
        x = sp.symbols("x, y, z")

        ell = alpha + (1 - alpha) * sp.Min(
            1, sp.Max(0, 3 * sp.Max(x[0], sp.Max(x[1], x[2])) - 1)
        )

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
    func_dict["r"] = funcify(r, x, dim)

    # couple stress
    if dim == 2:
        phi = ell * (x[0] - x[1]) * sp.sin(3 * sp.pi * x[0]) * sp.sin(3 * sp.pi * x[1])
        w = 2 * mu_w * sp.Matrix([phi, phi])
        div_ell_w = sum([sp.diff(ell * w_i, x_i) for (w_i, x_i) in zip(w, x)])
    else:
        phi = (
            ell
            * (x[0] - x[1])
            * (x[1] - x[2])
            * (x[2] - x[0])
            * sp.sin(3 * sp.pi * x[0])
            * sp.sin(3 * sp.pi * x[1])
            * sp.sin(3 * sp.pi * x[2])
        )
        w = (
            2
            * mu_w
            * sp.Matrix(
                [
                    [phi, 0, 0],
                    [0, phi, 0],
                    [0, 0, phi],
                ]
            )
        )
        div_ell_w = sp.Matrix(
            [
                sum([sp.diff(ell * w_i, x_i) for (w_i, x_i) in zip(w[i, :], x)])
                for i in range(3)
            ]
        )
    func_dict["w"] = funcify(w, x, dim)

    # Cauchy stress
    tau = grad_u + asym_r
    symtau = (tau + tau.T) / 2
    skwtau = (tau - tau.T) / 2
    trtauI = div_u * sp.Matrix(np.eye(dim))

    sigma = 2 * mu_s * symtau + 2 * mu_cs * skwtau + lam_s * trtauI

    div_sigma = sp.Matrix(
        [
            sum([sp.diff(s_ij, x_i) for (s_ij, x_i) in zip(sigma[i, :], x)])
            for i in range(2)
        ]
    )

    func_dict["s"] = funcify(sigma, x, dim, False)

    # Right-hand sides
    rhs_u = -div_sigma

    if dim == 2:
        rhs_w = w / (2 * mu_w) + ell * grad_r

        rhs_r = sigma[1, 0] - sigma[0, 1] - div_ell_w
        rhs_ur = sp.Matrix([*rhs_u, rhs_r])

    func_dict["rhs_w"] = funcify(rhs_w, x, dim, False)
    func_dict["rhs_ur"] = funcify(rhs_ur, x, dim)
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


def rotation(param, dim):
    alpha = param["alpha"]
    return funcs[dim, alpha]["r"]


def body_force(param, dim):
    alpha = param["alpha"]
    return funcs[dim, alpha]["rhs_w"]


def rhs_scaled(param, dim):
    alpha = param["alpha"]
    return funcs[dim, alpha]["rhs_ur"]


# sd = pg.unit_grid(2, 0.5, as_mdg=False)
# sd.compute_geometry()

# S = pg.VecBDM1()
# u_interp = S.interpolate(sd, sig_func)
# pass
