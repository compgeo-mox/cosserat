import os, sys
import numpy as np
import scipy.sparse as sps

import porepy as pp
import pygeon as pg

sys.path.append("./src")

from functions import make_summary
from analytical_solutions import cosserat_exact_3d


def main(mesh_size, folder):
    # return the exact solution and related rhs
    mu_s, mu_sc, lambda_s = 0.5, 0.25, 1
    mu_w, mu_wc, lambda_w = 0.5, 0.25, 1
    sigma_ex, w_ex, u_ex, r_ex, _, f_u, f_r = cosserat_exact_3d(
        mu_s, mu_sc, lambda_s, mu_w, mu_wc, lambda_w
    )

    mesh_file_name = os.path.join(folder, "grid.msh")
    sd = pg.unit_grid(3, mesh_size, as_mdg=False, file_name=mesh_file_name)
    sd.compute_geometry()

    key = "cosserat"
    vec_bdm1 = pg.VecBDM1(key)
    vec_p0 = pg.VecPwConstants(key)

    vec_p1 = pg.VecPwLinears(key)
    vec_l1 = pg.VecLagrange1(key)
    vec_l2 = pg.VecLagrange2(key)

    dofs = np.array(
        [vec_bdm1.ndof(sd), vec_bdm1.ndof(sd), vec_p0.ndof(sd), vec_l1.ndof(sd)]
    )
    split_idx = np.cumsum(dofs[:-1])

    data = {pp.PARAMETERS: {key: {"mu": mu_s, "lambda": lambda_s, "mu_c": mu_sc}}}

    Ms = vec_bdm1.assemble_mass_matrix_cosserat(sd, data)
    Mw = Ms.copy()
    Mu = vec_p0.assemble_mass_matrix(sd)

    M_p1 = vec_p1.assemble_mass_matrix(sd)

    proj_l1 = vec_l1.proj_to_pwLinears(sd)
    proj_p0 = vec_p0.proj_to_pwLinears(sd)

    div_op = vec_bdm1.assemble_diff_matrix(sd)
    div_s = Mu @ div_op

    asym_op = vec_bdm1.assemble_asym_matrix(sd, as_pwconstant=False)
    asym = proj_l1.T @ M_p1 @ asym_op

    div_w = proj_l1.T @ M_p1 @ proj_p0 @ div_op

    A = sps.block_diag([Ms, Mw], format="csc")
    B = sps.block_array([[-div_s, None], [asym, -div_w]], format="csr")
    spp = sps.block_array([[A, -B.T], [B, None]], format="csc")

    rhs = np.zeros(spp.shape[0])
    force_u = Mu @ vec_p0.interpolate(sd, f_u)
    force_r = proj_l1.T @ M_p1 @ vec_p1.interpolate(sd, f_r)

    bd_faces = sd.tags["domain_boundary_faces"]
    u_bc = vec_bdm1.assemble_nat_bc(sd, u_ex, bd_faces)
    r_bc = vec_bdm1.assemble_nat_bc(sd, r_ex, bd_faces)

    rhs[: split_idx[0]] += u_bc
    rhs[split_idx[0] : split_idx[1]] += r_bc
    rhs[split_idx[1] : split_idx[2]] += force_u
    rhs[split_idx[2] :] += force_r

    ls = pg.LinearSystem(spp, rhs)
    x = ls.solve()
    sigma, w, u, r = np.split(x, split_idx)

    # compute the error
    err_sigma = vec_bdm1.error_l2(sd, sigma, sigma_ex, data=data)
    err_w = vec_bdm1.error_l2(sd, w, w_ex, data=data)
    err_u = vec_p0.error_l2(sd, u, u_ex)
    r_l2 = vec_l1.proj_to_lagrange2(sd) @ r
    err_r = vec_l2.error_l2(sd, r_l2, r_ex)

    h = np.amax(sd.cell_diameters())
    return h, err_sigma, err_w, err_u, err_r, *dofs


if __name__ == "__main__":
    np.set_printoptions(precision=2, linewidth=9999)

    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    if not os.path.exists(folder):
        os.makedirs(folder)

    mesh_size = [0.4, 0.3, 0.2, 0.1, 0.05, 0.025]
    errs = np.vstack([main(h, folder) for h in mesh_size])
    errs_latex = make_summary(errs)

    # Write to a file
    with open(folder + "/case8_not_lump.tex", "w") as file:
        file.write(errs_latex)
