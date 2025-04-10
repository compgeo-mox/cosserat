import os, sys
import numpy as np
import scipy.sparse as sps

import porepy as pp
import pygeon as pg

sys.path.append("./src")

from functions import order
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
    vec_rt0 = pg.VecRT0(key)
    vec_bdm1 = pg.VecBDM1(key)
    vec_p0 = pg.VecPwConstants(key)

    dofs = np.array([vec_rt0.ndof(sd)] * 2 + [vec_p0.ndof(sd)] * 2)
    split_idx = np.cumsum(dofs[:-1])

    data = {pp.PARAMETERS: {key: {"mu": mu_s, "lambda": lambda_s, "mu_c": mu_sc}}}

    Ms = vec_rt0.assemble_mass_matrix_cosserat(sd, data)
    Mw = Ms
    Mu = vec_p0.assemble_mass_matrix(sd)
    Mr = Mu

    div_s = Mu @ vec_rt0.assemble_diff_matrix(sd)
    asym = Mr @ vec_rt0.assemble_asym_matrix(sd)
    div_w = div_s

    A = sps.block_diag([Ms, Mw])
    B = sps.bmat([[-div_s, None], [asym, -div_w]])
    spp = sps.bmat([[A, -B.T], [B, None]], format="csc")

    rhs = np.zeros(spp.shape[0])
    force_u = Mu @ vec_p0.interpolate(sd, f_u)
    force_r = Mr @ vec_p0.interpolate(sd, f_r)

    bd_faces = sd.tags["domain_boundary_faces"]
    u_bc = vec_rt0.assemble_nat_bc(sd, u_ex, bd_faces)
    r_bc = vec_rt0.assemble_nat_bc(sd, r_ex, bd_faces)

    rhs[: split_idx[0]] += u_bc
    rhs[split_idx[0] : split_idx[1]] += r_bc
    rhs[split_idx[1] : split_idx[2]] += force_u
    rhs[split_idx[2] :] += force_r

    ls = pg.LinearSystem(spp, rhs)
    x = ls.solve()
    sigma, w, u, r = np.split(x, split_idx)

    # compute the error
    sigma_bdm1 = vec_bdm1.proj_from_RT0(sd) @ sigma
    err_sigma = vec_bdm1.error_l2(sd, sigma_bdm1, sigma_ex, data=data)
    # err_sigma = vec_rt0.error_l2(sd, sigma, sigma_ex, data=data)
    w_bdm1 = vec_bdm1.proj_from_RT0(sd) @ w
    err_w = vec_bdm1.error_l2(sd, w_bdm1, w_ex, data=data)
    # err_w = vec_rt0.error_l2(sd, w, w_ex, data=data)
    err_u = vec_p0.error_l2(sd, u, lambda x: u_ex(x).ravel())
    err_r = vec_p0.error_l2(sd, r, lambda x: r_ex(x).ravel())

    if False:
        cell_sigma = vec_rt0.eval_at_cell_centers(sd) @ sigma
        cell_w = vec_rt0.eval_at_cell_centers(sd) @ w
        cell_u = vec_p0.eval_at_cell_centers(sd) @ u
        cell_r = vec_p0.eval_at_cell_centers(sd) @ r

        # we need to reshape for exporting
        cell_u = cell_u.reshape((3, -1))
        cell_r = cell_r.reshape((3, -1))

        save = pp.Exporter(sd, "sol_cosserat", folder_name=folder)
        save.write_vtu([("cell_u", cell_u), ("cell_r", cell_r)])

    h = np.amax(sd.cell_diameters())
    return err_sigma, err_w, err_u, err_r, h, *dofs, spp.nnz


if __name__ == "__main__":
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    np.set_printoptions(precision=2, linewidth=9999)

    mesh_size = [0.4, 0.3, 0.2, 0.1, 0.05]
    errs = np.vstack([main(h, folder) for h in mesh_size])
    print(errs)

    order_sigma = order(errs[:, 0], errs[:, 4])
    order_w = order(errs[:, 1], errs[:, 4])
    order_u = order(errs[:, 2], errs[:, 4])
    order_r = order(errs[:, 3], errs[:, 4])

    print(order_sigma, order_w, order_u, order_r)

    np.savetxt(os.path.join(folder, "errors.txt"), errs)

    orders = np.vstack([order_sigma, order_w, order_u, order_r])
    np.savetxt(os.path.join(folder, "orders.txt"), orders)
