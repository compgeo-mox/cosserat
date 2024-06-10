import os, sys
import numpy as np
import scipy.sparse as sps

import porepy as pp
import pygeon as pg

sys.path.append("./src")

from functions import order
from analytical_solutions import exact_sol_2d


def main(mesh_size):

    # return the exact solution and related rhs
    mu_s, lambda_s = 0.5, 1
    mu_w = mu_s
    sigma_ex, w_ex, u_ex, r_ex, f_u, f_r = exact_sol_2d(mu_s, lambda_s, mu_w)

    sd = pg.unit_grid(2, mesh_size, as_mdg=False)
    sd.compute_geometry()

    key = "cosserat"
    vec_rt0 = pg.VecRT0(key)
    rt0 = pg.RT0(key)
    vec_p0 = pg.VecPwConstants(key)
    p0 = pg.PwConstants(key)

    dofs = np.array([vec_rt0.ndof(sd), rt0.ndof(sd), vec_p0.ndof(sd), p0.ndof(sd)])
    split_idx = np.cumsum(dofs[:-1])

    data = {pp.PARAMETERS: {key: {"mu": mu_w, "lambda": lambda_s}}}

    Ms = vec_rt0.assemble_mass_matrix(sd, data)
    Mw = rt0.assemble_mass_matrix(sd)  # attention to the data here
    Mu = vec_p0.assemble_mass_matrix(sd)
    Mr = p0.assemble_mass_matrix(sd)

    div_s = Mu @ vec_rt0.assemble_diff_matrix(sd)
    asym = Mr @ vec_rt0.assemble_asym_matrix(sd)
    div_w = Mr @ rt0.assemble_diff_matrix(sd)

    # fmt: off
    spp = sps.bmat([[    Ms,   None, div_s.T,  asym.T],
                    [  None,     Mw,    None, div_w.T],
                    [-div_s,   None,    None,    None],
                    [ -asym, -div_w,    None,    None]], format = "csc")
    # fmt: on

    rhs = np.zeros(spp.shape[0])
    force_u = Mu @ vec_p0.interpolate(sd, lambda x: f_u(x)[: sd.dim].ravel())
    force_r = Mr @ p0.interpolate(sd, lambda x: f_r(x)[sd.dim]).ravel()

    rhs[split_idx[1] : split_idx[2]] += force_u
    rhs[split_idx[2] :] += force_r

    ls = pg.LinearSystem(spp, rhs)
    x = ls.solve()

    sigma, w, u, r = np.split(x, split_idx)

    # compute the error
    err_sigma = vec_rt0.error_l2(sd, sigma, sigma_ex, data=data)
    err_w = rt0.error_l2(sd, w, lambda x: w_ex(x)[sd.dim].ravel())
    err_u = vec_p0.error_l2(sd, u, lambda x: u_ex(x)[: sd.dim].ravel())
    err_r = p0.error_l2(sd, r, lambda x: r_ex(x)[sd.dim][0])

    if False:
        # compute the projections
        cell_sigma = vec_rt0.eval_at_cell_centers(sd) @ sigma
        cell_w = rt0.eval_at_cell_centers(sd) @ w
        cell_u = vec_p0.eval_at_cell_centers(sd) @ u
        cell_r = p0.eval_at_cell_centers(sd) @ r

        # we need to add the z component for the exporting
        cell_u = np.hstack((cell_u, np.zeros(sd.num_cells)))
        cell_u = cell_u.reshape((3, -1))

        cell_w = cell_w.reshape((3, -1), order="F")

        folder = os.path.dirname(os.path.abspath(__file__))
        save = pp.Exporter(sd, "sol_cosserat", folder_name=folder)
        save.write_vtu([("cell_u", cell_u), ("cell_r", cell_r), ("cell_w", cell_w)])

    h = np.amax(sd.cell_diameters())
    return err_sigma, err_w, err_u, err_r, h, *dofs, spp.nnz


if __name__ == "__main__":
    mesh_size = np.power(2.0, -np.arange(3, 3 + 3))  # 5
    errs = np.vstack([main(h) for h in mesh_size])
    print(errs)

    order_sigma = order(errs[:, 0], errs[:, 4])
    order_w = order(errs[:, 1], errs[:, 4])
    order_u = order(errs[:, 2], errs[:, 4])
    order_r = order(errs[:, 3], errs[:, 4])

    print(order_sigma, order_w, order_u, order_r)
