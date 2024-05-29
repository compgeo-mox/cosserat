import os
import numpy as np
import scipy.sparse as sps

import porepy as pp
import pygeon as pg

""" Spanning tree solve for cosserat materials, made for the ECCOMAS presentation """


def main():
    sd = pg.unit_grid(3, 0.25, as_mdg=False)
    sd.compute_geometry()

    key = "cosserat"
    vec_rt0 = pg.VecRT0(key)
    vec_p0 = pg.VecPwConstants(key)

    data = {pp.PARAMETERS: {key: {"mu": 0.5, "lambda": 0.5}}}
    Ms = vec_rt0.assemble_mass_matrix(sd, data)
    Mw = Ms
    Mu = vec_p0.assemble_mass_matrix(sd)
    Mr = Mu

    div_s = Mu @ vec_rt0.assemble_diff_matrix(sd)
    asym = Mr @ vec_rt0.assemble_asym_matrix(sd)
    div_w = div_s

    # fmt: off
    spp = sps.bmat([[     Ms, None, div_s.T, asym.T],
                    [   None,   Mw, None, div_w.T],
                    [ -div_s,  None, None,   None],
                    [-asym,  -div_w, None,   None]], format = "csc")
    # fmt: on

    A = sps.bmat([[Ms, None], [None, Mw]], format="csc")
    B = sps.bmat([[-div_s, None], [-asym, -div_w]], format="csc")

    # Set essential boundary conditions on all faces except the ones at the bottom
    ess_dof = sd.tags["domain_boundary_faces"].copy()
    ess_dof[np.isclose(sd.face_centers[2, :], 0)] = False
    ess_dof = np.tile(ess_dof, 6)

    to_keep = np.logical_not(ess_dof)
    R_0 = pg.numerics.linear_system.create_restriction(to_keep)

    ess_dof = np.hstack((ess_dof, np.zeros(6 * sd.num_cells, dtype=bool)))

    rhs = np.zeros(spp.shape[0])
    force = lambda x: np.array([0, 0, -1])
    force_p0 = vec_p0.interpolate(sd, force)
    force_rhs = Mu @ force_p0

    split_idx = np.cumsum([vec_rt0.ndof(sd), vec_rt0.ndof(sd), vec_p0.ndof(sd)])

    rhs[split_idx[1] : split_idx[2]] += force_rhs
    # rhs[: vec_rt0.ndof(sd)] = bc

    # Direct solve for comparison
    ls = pg.LinearSystem(spp, rhs)
    ls.flag_ess_bc(ess_dof, np.zeros(spp.shape[0]))
    x = ls.solve()

    sigma, w, u, r = np.split(x, split_idx)

    # Construct spanning tree
    mdg = pg.as_mdg(sd)

    starting_faces = np.logical_and(
        sd.tags["domain_boundary_faces"], np.isclose(sd.face_centers[2, :], 0)
    )
    sptr = pg.SpanningTreeCosserat(mdg, starting_faces=starting_faces)

    # Step 1
    source = rhs[split_idx[1] :]
    sig_f = sptr.solve(source)

    for tol in [1e-5, 1e-6, 1e-7, 1e-8, 1e-9]:

        # Step 2
        SI = sptr.assemble_SI()
        S0 = sps.eye_array(SI.shape[0]) - SI @ B

        A_op = S0.T @ A @ S0  # + self.SI(self.SI_T(x))
        A_op_red = R_0 @ A_op @ R_0.T
        b = R_0 @ S0.T @ (-A @ sig_f)

        sig_0, exit_code = sps.linalg.cg(A_op_red, b, rtol=tol)

        sig_omega = sig_f + S0 @ R_0.T @ sig_0
        sig_cg, w_cg = np.split(sig_omega, 2)

        # Step 3
        u_r = sptr.solve_transpose(A @ sig_omega)
        u_cg, r_cg = np.split(u_r, 2)

        def compute_error(xn, x, M):
            # compute the L2 error
            delta = xn - x
            norm_x = np.sqrt(x @ M @ x)
            return np.sqrt(delta @ M @ delta) / (
                norm_x if not np.isclose(norm_x, 0) else 1
            )

        err = np.zeros(4)
        err[0] = compute_error(sig_cg, sigma, Ms)
        err[1] = compute_error(w_cg, w, Mw)
        err[2] = compute_error(u_cg, u, Mu)
        err[3] = compute_error(r_cg, r, Mr)

        print("Errors: {:.2E}, {:.2E}, {:.2E}, {:.2E}".format(*err))


if __name__ == "__main__":
    main()
