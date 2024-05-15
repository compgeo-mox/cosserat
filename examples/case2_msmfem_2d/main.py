import os, sys
import numpy as np
import scipy.sparse as sps

import porepy as pp
import pygeon as pg

sys.path.append("./src")

from functions import err, order
from analytical_solutions import exact_sol_2d


def main(mesh_size):

    # return the exact solution and related rhs
    sigma_ex, w_ex, u_ex, r_ex, f_u, f_r = exact_sol_2d()

    sd = pg.unit_grid(2, mesh_size, as_mdg=False)
    sd.compute_geometry()

    key = "cosserat"
    vec_bdm1 = pg.VecBDM1(key)
    bdm1 = pg.BDM1(key)
    vec_p0 = pg.VecPwConstants(key)
    p0 = pg.PwConstants(key)

    split_idx = np.cumsum([vec_bdm1.ndof(sd), bdm1.ndof(sd), vec_p0.ndof(sd)])

    data = {pp.PARAMETERS: {key: {"mu": 0.5, "lambda": 0}}}
    Ms = vec_bdm1.assemble_lumped_matrix(sd, data)
    Mw = bdm1.assemble_lumped_matrix(sd, data)
    Mu = vec_p0.assemble_mass_matrix(sd)
    Mr = p0.assemble_mass_matrix(sd)

    div_s = Mu @ vec_bdm1.assemble_diff_matrix(sd)
    asym = Mr @ vec_bdm1.assemble_asym_matrix(sd)
    div_w = Mr @ bdm1.assemble_diff_matrix(sd)

    # fmt: off
    spp = sps.bmat([[    Ms,   None, div_s.T,  asym.T],
                    [  None,     Mw,    None, div_w.T],
                    [-div_s,   None,    None,    None],
                    [ -asym, -div_w,    None,    None]], format = "csc")
    # fmt: on

    b_faces = np.zeros(sd.num_faces, dtype=bool)
    b_faces = np.tile(b_faces, 2 * 2 + 2 * 1)
    b_faces = np.hstack((b_faces, np.zeros((2 + 1) * sd.num_cells, dtype=bool)))

    rhs = np.zeros(spp.shape[0])
    force_u = Mu @ vec_p0.interpolate(sd, lambda x: f_u(x)[: sd.dim].ravel())
    force_r = Mr @ p0.interpolate(sd, lambda x: f_r(x)[sd.dim]).ravel()

    rhs[split_idx[1] : split_idx[2]] += force_u
    rhs[split_idx[2] :] += force_r

    ls = pg.LinearSystem(spp, rhs)
    ls.flag_ess_bc(b_faces, np.zeros(spp.shape[0]))
    x = ls.solve()

    sigma, w, u, r = np.split(x, split_idx)

    # compute the error
    sigma_ex_int = vec_bdm1.interpolate(sd, sigma_ex)
    err_sigma = err(sigma, sigma_ex_int, Ms)

    w_ex_int = bdm1.interpolate(sd, lambda x: w_ex(x)[sd.dim].ravel())
    err_w = err(w, w_ex_int, Mw)

    u_ex_int = vec_p0.interpolate(sd, lambda x: u_ex(x)[: sd.dim].ravel())
    err_u = err(u, u_ex_int, Mu)

    r_ex_int = p0.interpolate(sd, lambda x: r_ex(x)[sd.dim]).ravel()
    err_r = err(r, r_ex_int, Mr)

    if False:
        cell_sigma = vec_bdm1.eval_at_cell_centers(sd) @ sigma
        cell_w = bdm1.eval_at_cell_centers(sd) @ w
        cell_u = vec_p0.eval_at_cell_centers(sd) @ u
        cell_r = p0.eval_at_cell_centers(sd) @ r

        # we need to add the z component for the exporting
        cell_u = np.hstack((cell_u, np.zeros(sd.num_cells)))
        cell_u = cell_u.reshape((3, -1))

        # we need to reshape for exporting
        cell_w = cell_w.reshape((3, -1))

        folder = os.path.dirname(os.path.abspath(__file__))
        save = pp.Exporter(sd, "sol_cosserat", folder_name=folder)
        save.write_vtu([("cell_u", cell_u), ("cell_r", cell_r), ("cell_w", cell_w)])

    return err_sigma, err_w, err_u, err_r, np.amax(sd.cell_diameters())


if __name__ == "__main__":
    mesh_size = np.power(2.0, -np.arange(3, 3 + 5))
    errs = np.vstack([main(h) for h in mesh_size])

    order_sigma = order(errs[:, 0], errs[:, -1])
    order_w = order(errs[:, 1], errs[:, -1])
    order_u = order(errs[:, 2], errs[:, -1])
    order_r = order(errs[:, 3], errs[:, -1])

    print(errs)
    print(order_sigma, order_w, order_u, order_r)
