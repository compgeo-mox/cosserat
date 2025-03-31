import os, sys
import numpy as np
import scipy.sparse as sps

import porepy as pp
import pygeon as pg

sys.path.append("./src")

from functions import make_summary
from analytical_solutions import cosserat_exact_2d


def main(mesh_size, folder):
    # return the exact solution and related rhs
    mu_s, mu_sc, lambda_s = 0.5, 0.25, 1
    mu_w = 0.5
    sigma_ex, w_ex, u_ex, r_ex, _, f_u, f_r = cosserat_exact_2d(
        mu_s, mu_sc, lambda_s, mu_w
    )

    mesh_file_name = os.path.join(folder, "grid.msh")
    sd = pg.unit_grid(2, mesh_size, as_mdg=False, file_name=mesh_file_name)
    sd.compute_geometry()

    key = "cosserat"
    vec_bdm1 = pg.VecBDM1(key)
    bdm1 = pg.BDM1(key)
    vec_p0 = pg.VecPwConstants(key)
    p0 = pg.PwConstants(key)

    p1 = pg.PwLinears(key)
    l1 = pg.Lagrange1(key)
    l2 = pg.Lagrange2(key)

    dofs = np.array([vec_p0.ndof(sd), l1.ndof(sd)])
    split_idx = np.cumsum(dofs[:-1])

    data = {pp.PARAMETERS: {key: {"mu": mu_s, "lambda": lambda_s, "mu_c": mu_sc}}}

    Ms = vec_bdm1.assemble_lumped_matrix_cosserat(sd, data)
    Mw = bdm1.assemble_lumped_matrix(sd)  # NOTE attention to the data here
    Mu = vec_p0.assemble_mass_matrix(sd)

    M_p1_lumped = p1.assemble_lumped_matrix(sd)

    proj_l1 = l1.proj_to_pwLinears(sd)
    proj_p0 = p0.proj_to_pwLinears(sd)

    div_s = Mu @ vec_bdm1.assemble_diff_matrix(sd)

    asym_op = vec_bdm1.assemble_asym_matrix(sd, as_pwconstant=False)
    asym = proj_l1.T @ M_p1_lumped @ asym_op

    div_w_op = bdm1.assemble_diff_matrix(sd)
    div_w = proj_l1.T @ M_p1_lumped @ proj_p0 @ div_w_op

    A = sps.block_diag([Ms, Mw], format="csc")
    B = sps.block_array([[-div_s, None], [asym, -div_w]], format="csr")
    Q = sps.linalg.spsolve(A, B.T)
    spp = B @ Q

    bd_faces = sd.tags["domain_boundary_faces"]
    u_bc = vec_bdm1.assemble_nat_bc(sd, u_ex, bd_faces)
    r_bc = bdm1.assemble_nat_bc(sd, r_ex, bd_faces)
    x_bc = np.concatenate([u_bc, r_bc])

    rhs = np.zeros(spp.shape[0])
    rhs[: split_idx[0]] += Mu @ vec_p0.interpolate(sd, f_u)
    rhs[split_idx[0] :] += proj_l1.T @ M_p1_lumped @ p1.interpolate(sd, f_r)
    rhs -= B @ sps.linalg.spsolve(A, x_bc)

    ls = pg.LinearSystem(spp, rhs)
    x = ls.solve()
    u, r = np.split(x, split_idx)

    y = Q @ x + sps.linalg.spsolve(A, x_bc)
    sigma, w = np.split(y, [vec_bdm1.ndof(sd)])

    # compute the error
    err_sigma = vec_bdm1.error_l2(sd, sigma, sigma_ex, data=data)
    err_w = bdm1.error_l2(sd, w, w_ex, data=data)
    err_u = vec_p0.error_l2(sd, u, u_ex)
    r_l2 = l1.proj_to_lagrange2(sd) @ r
    err_r = l2.error_l2(sd, r_l2, r_ex)

    h = np.amax(sd.cell_diameters())
    return h, err_sigma, err_w, err_u, err_r


if __name__ == "__main__":
    np.set_printoptions(precision=2, linewidth=9999)

    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    if not os.path.exists(folder):
        os.makedirs(folder)

    mesh_size = np.power(2.0, -np.arange(3, 3 + 5))
    errs = np.vstack([main(h, folder) for h in mesh_size])
    errs_latex = make_summary(errs)

    # Write to a file
    with open(folder + "/case7_lump.tex", "w") as file:
        file.write(errs_latex)
