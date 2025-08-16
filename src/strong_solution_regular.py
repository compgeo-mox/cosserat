import numpy as np
import sympy as sp

import pygeon as pg

mu_s = mu_w = 1
mu_cs = mu_cw = 0.1
lam_s = lam_w = 1

# 2D
x = sp.symbols("x, y")

ell = sp.Min(1, sp.Max(0, 3 * sp.Max(x[0], x[1]) - 1))

# displacement
u = sp.Matrix(
    [
        x[1] * (1 - x[1]) * sp.sin(sp.pi * x[0]),
        x[0] * (1 - x[0]) * sp.sin(sp.pi * x[1]),
    ]
)

grad_u = sp.Matrix([[sp.diff(u_i, x_j) for x_j in x] for u_i in u])
div_u = sum([sp.diff(u_i, x_i) for (u_i, x_i) in zip(u, x)])

# rotation
r = sp.sin(sp.pi * x[0]) * sp.sin(sp.pi * x[1])
r_grad = sp.Matrix([sp.diff(r, x_i) for x_i in x])
asym_r = sp.Matrix(
    [
        [0, -r],
        [r, 0],
    ]
)

# couple stress
w = (
    2
    * mu_w
    * sp.Matrix(
        [
            ell * (x[0] - x[1]) * sp.sin(3 * sp.pi * x[0]) * sp.sin(3 * sp.pi * x[1]),
            ell * (x[0] - x[1]) * sp.sin(3 * sp.pi * x[0]) * sp.sin(3 * sp.pi * x[1]),
        ]
    )
)

div_ell_w = sum([sp.diff(ell * omega_i, x_i) for (omega_i, x_i) in zip(w, x)])

# Cauchy stress
tau = grad_u + asym_r
symtau = (tau + tau.T) / 2
skwtau = (tau - tau.T) / 2
trtauI = div_u * sp.Matrix(np.eye(2))

sigma = 2 * mu_s * symtau + 2 * mu_cs * skwtau + lam_s * trtauI

div_sigma = sp.Matrix(
    [sum([sp.diff(s_ij, x_i) for (s_ij, x_i) in zip(sigma[i, :], x)]) for i in range(2)]
)

# extend sigma
sigma_ex = sp.matrix2numpy(sigma)
sigma_ex = np.hstack((sigma_ex, np.zeros((2, 1))))
sigma_ex = sp.Matrix(sigma_ex)

# Right-hand sides
rhs_sigma = 0
rhs_w = w / (2 * mu_w) + ell * r_grad
rhs_u = -div_sigma
rhs_r = sigma[1, 0] - sigma[0, 1] - div_ell_w


# lambda functions
sig_lam = sp.lambdify(x, sigma_ex, "numpy")


def sig_func(x):
    return sig_lam(*x[:2])


def funcify(f):
    lam_f = sp.lambdify(x, f, "numpy")
    return lambda x: lam_f(*x[:2]).ravel()


w_func = funcify(w)
u_func = funcify(u)
r_func = funcify(r)

# sd = pg.unit_grid(2, 0.5, as_mdg=False)
# sd.compute_geometry()

# S = pg.VecBDM1()
# u_interp = S.interpolate(sd, sig_func)
# pass
