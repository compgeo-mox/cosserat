import os, sys
import numpy as np
import scipy.sparse as sps
import time

import porepy as pp
import pygeon as pg

sys.path.append("./src")

from functions import order
from analytical_solutions import exact_sol_3d


class IterationCallback:
    def __init__(self):
        self.iteration_count = 0

    def __call__(self, xk):
        self.iteration_count += 1

    def get_iteration_count(self):
        return self.iteration_count


def main(mesh_size):
    # return the exact solution and related rhs
    mu_s, lambda_s = 0.5, 1
    mu_w, lambda_w = mu_s, lambda_s
    sigma_ex, w_ex, u_ex, r_ex, f_u, f_r = exact_sol_3d(mu_s, lambda_s, mu_w, lambda_w)

    sd = pg.unit_grid(3, mesh_size, as_mdg=False)
    sd.compute_geometry()

    key = "cosserat"
    vec_bdm1 = pg.VecBDM1(key)
    vec_p0 = pg.VecPwConstants(key)

    dofs = np.array([vec_p0.ndof(sd), vec_p0.ndof(sd)])
    split_idx = np.cumsum(dofs[:-1])

    data = {pp.PARAMETERS: {key: {"mu": mu_s, "lambda": lambda_s}}}

    Ms = vec_bdm1.assemble_lumped_matrix(sd, data)
    Mw = Ms.copy()
    Mu = vec_p0.assemble_mass_matrix(sd)
    Mr = Mu.copy()

    div_s = Mu @ vec_bdm1.assemble_diff_matrix(sd)
    asym = Mr @ vec_bdm1.assemble_asym_matrix(sd)
    div_w = div_s.copy()

    # fmt: off
    A = sps.block_diag([Ms, Mw], format="csc")
    B = sps.bmat([[-div_s,  None], [asym, -div_w]], format="csr")
    Q = sps.linalg.spsolve(A, B.T)

    spp = B @ Q
    # fmt: on

    bd_faces = sd.tags["domain_boundary_faces"]
    u_bc = vec_bdm1.assemble_nat_bc(sd, u_ex, bd_faces)
    r_bc = vec_bdm1.assemble_nat_bc(sd, r_ex, bd_faces)
    x_bc = np.concatenate([u_bc, r_bc])

    rhs = np.zeros(spp.shape[0])
    force_u = Mu @ vec_p0.interpolate(sd, f_u)
    force_r = Mr @ vec_p0.interpolate(sd, f_r)

    rhs[: split_idx[0]] += force_u
    rhs[split_idx[0] :] += force_r

    # try
    rt0 = pg.RT0(key)
    L = rt0.assemble_lumped_matrix(sd)
    div = rt0.assemble_diff_matrix(sd)
    P = sps.block_diag([div @ sps.linalg.spsolve(L, div.T)] * 6, format="csc")

    callback = IterationCallback()
    start = time.time()
    x_minres, info = sps.linalg.minres(spp, rhs, M=P, callback=callback, rtol=1e-10)

    print("Time minres:", time.time() - start)
    print("Info:", info)
    print(f"Total iterations: {callback.get_iteration_count()}")

    ls = pg.LinearSystem(spp, rhs)
    start = time.time()
    x = ls.solve()
    print("Time direct:", time.time() - start)

    print(np.linalg.norm(x - x_minres) / np.linalg.norm(x))

    u, r = np.split(x, split_idx)

    y = Q @ x + sps.linalg.spsolve(A, x_bc)
    sigma, w = np.split(y, [vec_bdm1.ndof(sd)])

    # compute the error
    err_sigma = vec_bdm1.error_l2(sd, sigma, sigma_ex, data=data)
    err_w = vec_bdm1.error_l2(sd, w, w_ex, data=data)
    err_u = vec_p0.error_l2(sd, u, u_ex)
    err_r = vec_p0.error_l2(sd, r, r_ex)

    if False:
        cell_sigma = vec_bdm1.eval_at_cell_centers(sd) @ sigma
        cell_w = vec_bdm1.eval_at_cell_centers(sd) @ w
        cell_u = vec_p0.eval_at_cell_centers(sd) @ u
        cell_r = vec_p0.eval_at_cell_centers(sd) @ r

        # we need to reshape for exporting
        cell_u = cell_u.reshape((3, -1))
        cell_r = cell_r.reshape((3, -1))
        # cell_w = cell_w.reshape((9, -1))
        # cell_sigma = cell_sigma.reshape((9, -1))

        folder = os.path.dirname(os.path.abspath(__file__))
        save = pp.Exporter(sd, "sol_cosserat", folder_name=folder)
        save.write_vtu([("cell_u", cell_u), ("cell_r", cell_r)])

    h = np.amax(sd.cell_diameters())
    return err_sigma, err_w, err_u, err_r, h, *dofs, spp.nnz


if __name__ == "__main__":
    np.set_printoptions(precision=2, linewidth=9999)

    mesh_size = [0.4, 0.3, 0.2, 0.1]
    errs = np.vstack([main(h) for h in mesh_size])
    print(errs)

    order_sigma = order(errs[:, 0], errs[:, 4])
    order_w = order(errs[:, 1], errs[:, 4])
    order_u = order(errs[:, 2], errs[:, 4])
    order_r = order(errs[:, 3], errs[:, 4])

    print(order_sigma, order_w, order_u, order_r)
