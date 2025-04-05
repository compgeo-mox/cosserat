import sympy as sp


def vector_gradient(u, R):
    return sp.simplify(
        sp.Matrix([sp.diff(u_i, var) for u_i in u for var in R.varlist]).reshape(3, 3)
    )


def asym_T(r):
    return sp.simplify(
        sp.Matrix([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
    )


def asym(sigma):
    asym_sigma = sigma - sigma.T
    return sp.simplify(
        sp.Matrix([asym_sigma[2, 1], asym_sigma[0, 2], asym_sigma[1, 0]])
    )


def matrix_divergence(sigma, R):
    x, y, z = R.varlist
    return sp.simplify(
        sp.diff(sigma[:, 0], x) + sp.diff(sigma[:, 1], y) + sp.diff(sigma[:, 2], z)
    )


def skew(sigma):
    return 0.5 * (sigma - sigma.T)


def sym(sigma):
    return 0.5 * (sigma + sigma.T)


def cosserat_compute_all(dim, u, r, mu_s, mu_sc, lambda_s, mu_w, mu_wc, lambda_w, R):
    if dim == 2:
        I = sp.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
    else:
        I = sp.Identity(3).as_explicit()

    # sigma
    tau = vector_gradient(u, R) + asym_T(r)
    sigma = 2 * mu_s * sym(tau) + 2 * mu_sc * skew(tau) + lambda_s * sp.trace(tau) * I

    # w
    tau = vector_gradient(r, R)
    if dim == 2:
        w = 2 * mu_w * tau + lambda_w * sp.trace(tau) * I
    else:
        w = 2 * mu_w * sym(tau) + 2 * mu_wc * skew(tau) + lambda_w * sp.trace(tau) * I

    # Compute the source term
    f_u = -matrix_divergence(sigma, R)
    f_r = asym(sigma) - matrix_divergence(w, R)

    return sigma, w, f_u, f_r


def elasticity_compute_all(dim, sigma, u, r, mu, lambda_, R):
    if dim == 2:
        I = sp.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
    else:
        I = sp.Identity(3).as_explicit()

    coeff = lambda_ / (2 * mu + dim * lambda_)
    Asigma = (sigma - coeff * sp.trace(sigma) * I) / (2 * mu)
    Asigma = sp.simplify(Asigma)

    # Compute the source term
    f_s = Asigma - vector_gradient(u, R) - asym_T(r)
    f_u = -matrix_divergence(sigma, R)

    return f_s, f_u


def cosserate_as_lambda(dim, sigma, w, u, r, f_s, f_u, f_r, R):
    x, y, z = R.varlist

    # lambdify the exact solution
    sigma_lamb = sp.lambdify([x, y, z], sigma)
    sigma_ex = lambda pt: sigma_lamb(*pt)

    w_lamb = sp.lambdify([x, y, z], w)
    if dim == 2:
        w_ex = lambda pt: w_lamb(*pt)[dim]
    else:
        w_ex = lambda pt: w_lamb(*pt)

    u_lamb = sp.lambdify([x, y, z], u)
    u_ex = lambda pt: u_lamb(*pt).ravel()[:dim]

    r_lamb = sp.lambdify([x, y, z], r)
    if dim == 2:
        r_ex = lambda pt: r_lamb(*pt)[dim, 0]
    else:
        r_ex = lambda pt: r_lamb(*pt).ravel()

    f_s_lamb = sp.lambdify([x, y, z], f_s)
    f_s_ex = lambda pt: f_s_lamb(*pt)

    f_u_lamb = sp.lambdify([x, y, z], f_u)
    f_u_ex = lambda pt: f_u_lamb(*pt).ravel()[:dim]

    f_r_lamb = sp.lambdify([x, y, z], f_r)
    if dim == 2:
        f_r_ex = lambda pt: f_r_lamb(*pt)[dim, 0]
    else:
        f_r_ex = lambda pt: f_r_lamb(*pt).ravel()

    return {
        "s_ex": sigma_ex,
        "w_ex": w_ex,
        "u_ex": u_ex,
        "r_ex": r_ex,
        "f_s": f_s_ex,
        "f_u": f_u_ex,
        "f_r": f_r_ex,
    }
