import os, sys
import numpy as np
import scipy.sparse as sps
import time

import porepy as pp
import pygeon as pg

sys.path.append("./src")

from functions import make_summary
from analytical_solutions import cosserat_exact_3d


class IterationCallback:
    def __init__(self):
        self.iteration_count = 0

    def __call__(self, xk):
        self.iteration_count += 1

    def get_iteration_count(self):
        return self.iteration_count


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

    dofs = np.array(
        [vec_bdm1.ndof(sd), vec_bdm1.ndof(sd), vec_p0.ndof(sd), vec_p0.ndof(sd)]
    )
    split_idx = np.cumsum(dofs[:-1])

    data = {pp.PARAMETERS: {key: {"mu": mu_s, "lambda": lambda_s, "mu_c": mu_sc}}}

    Ms = vec_bdm1.assemble_mass_matrix_cosserat(sd, data)
    Mw = Ms.copy()
    Mu = vec_p0.assemble_mass_matrix(sd)
    Mr = Mu.copy()

    div_s = Mu @ vec_bdm1.assemble_diff_matrix(sd)
    asym = Mr @ vec_bdm1.assemble_asym_matrix(sd)
    div_w = div_s.copy()

    A = sps.block_diag([Ms, Mw], format="csc")
    B = sps.block_array([[-div_s, None], [asym, -div_w]], format="csr")
    spp = sps.block_array([[A, -B.T], [B, None]], format="csc")

    rhs = np.zeros(spp.shape[0])
    force_u = Mu @ vec_p0.interpolate(sd, f_u)
    force_r = Mr @ vec_p0.interpolate(sd, f_r)

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
    err_r = vec_p0.error_l2(sd, r, r_ex)

    h = np.amax(sd.cell_diameters())
    return h, err_sigma, err_w, err_u, err_r, *dofs


if __name__ == "__main__":
    np.set_printoptions(precision=2, linewidth=9999)

    folder = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results_case4_not_lump"
    )
    if not os.path.exists(folder):
        os.makedirs(folder)

    mesh_size = [0.4, 0.3, 0.2, 0.1, 0.05, 0.025]
    errs = np.vstack([main(h, folder) for h in mesh_size])
    errs_latex = make_summary(errs)

    # Write to a file
    with open(folder + "/latex_table.tex", "w") as file:
        file.write(errs_latex)
