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

    dofs = np.array([vec_p0.ndof(sd), vec_p0.ndof(sd)])
    split_idx = np.cumsum(dofs[:-1])

    data = {pp.PARAMETERS: {key: {"mu": mu_s, "lambda": lambda_s, "mu_c": mu_sc}}}

    Ms = vec_bdm1.assemble_lumped_matrix_cosserat(sd, data)
    Mw = Ms.copy()
    Mu = vec_p0.assemble_mass_matrix(sd)
    Mr = Mu.copy()

    div_s = Mu @ vec_bdm1.assemble_diff_matrix(sd)
    asym = Mr @ vec_bdm1.assemble_asym_matrix(sd)
    div_w = div_s.copy()

    A = sps.block_diag([Ms, Mw], format="csc")
    B = sps.block_array([[-div_s, None], [asym, -div_w]], format="csr")

    # use spsolve only once and put div_s and asym in one single matrix
    diff = sps.hstack([-div_s.T, asym.T], format="csc")

    inv_diff = pg.block_diag_solver(Ms, diff)
    # inv_diff_old = sps.linalg.spsolve(Ms, diff)

    inv_div_T = inv_diff[:, : div_s.shape[0]]
    inv_asym_T = inv_diff[:, div_s.shape[0] :]

    Q = sps.block_array([[inv_div_T, inv_asym_T], [None, inv_div_T]], format="csc")
    spp = B @ Q

    bd_faces = sd.tags["domain_boundary_faces"]
    u_bc = vec_bdm1.assemble_nat_bc(sd, u_ex, bd_faces)
    r_bc = vec_bdm1.assemble_nat_bc(sd, r_ex, bd_faces)
    x_bc = np.concatenate([u_bc, r_bc])

    rhs = np.zeros(spp.shape[0])
    force_u = Mu @ vec_p0.interpolate(sd, f_u)
    force_r = Mr @ vec_p0.interpolate(sd, f_r)

    rhs[: split_idx[0]] += force_u
    rhs[split_idx[0] :] += force_r

    # solve the saddle point problem using minres and a preconditioner
    rt0 = pg.RT0(key)
    L = rt0.assemble_lumped_matrix(sd)
    div = rt0.assemble_diff_matrix(sd)

    # factorize the single and then use it in the pot
    inv_P = div @ sps.linalg.spsolve(L, div.T.tocsc())
    P = sps.linalg.splu(inv_P.tocsc())

    num = 6
    shape = np.array(inv_P.shape) * num
    matvec = lambda x: np.array(
        [P.solve(x_part) for x_part in np.array_split(x, num)]
    ).ravel()
    P_op = sps.linalg.LinearOperator(shape, matvec=matvec)

    callback = IterationCallback()
    start = time.time()
    x, info = sps.linalg.minres(spp, rhs, M=P_op, callback=callback, rtol=1e-5)

    it = callback.get_iteration_count()
    print("Time minres:", time.time() - start)
    print("Info:", info)
    print(f"Total iterations: {it}")

    u, r = np.split(x, split_idx)

    y = Q @ x + sps.linalg.spsolve(A, x_bc)
    sigma, w = np.split(y, [vec_bdm1.ndof(sd)])

    # compute the error
    err_sigma = vec_bdm1.error_l2(sd, sigma, sigma_ex, data=data)
    print(err_sigma)
    err_w = vec_bdm1.error_l2(sd, w, w_ex, data=data)
    err_u = vec_p0.error_l2(sd, u, u_ex)
    err_r = vec_p0.error_l2(sd, r, r_ex)

    h = np.amax(sd.cell_diameters())
    return h, err_sigma, err_w, err_u, err_r, *dofs


if __name__ == "__main__":
    np.set_printoptions(precision=2, linewidth=9999)

    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    if not os.path.exists(folder):
        os.makedirs(folder)

    mesh_size = [0.4, 0.3, 0.2]  # , 0.1, 0.05, 0.025]
    errs = np.vstack([main(h, folder) for h in mesh_size])
    errs_latex = make_summary(errs)

    # Write to a file
    with open(folder + "/case4_lump.tex", "w") as file:
        file.write(errs_latex)
