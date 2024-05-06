import os
import numpy as np
import scipy.sparse as sps

import porepy as pp
import pygeon as pg


def main():
    sd = pg.unit_grid(2, 0.1, as_mdg=False)
    sd.compute_geometry()

    key = "cosserat"
    vec_bdm1 = pg.VecBDM1(key)
    bdm1 = pg.BDM1(key)
    vec_p0 = pg.VecPwConstants(key)
    p0 = pg.PwConstants(key)

    data = {pp.PARAMETERS: {key: {"mu": 0.5, "lambda": 0.5}}}
    Ms = vec_bdm1.assemble_lumped_matrix(sd, data)  # TODO the data are not considered
    Mw = bdm1.assemble_lumped_matrix(sd, data)  # TODO the data are not considered
    Mu = vec_p0.assemble_mass_matrix(sd)
    Mr = p0.assemble_mass_matrix(sd)

    div_s = Mu @ vec_bdm1.assemble_diff_matrix(sd)
    asym = Mr @ vec_bdm1.assemble_asym_matrix(sd)
    div_w = Mr @ bdm1.assemble_diff_matrix(sd)

    # fmt: off
    spp = sps.bmat([[     Ms, None, div_s.T, asym.T],
                    [   None,   Mw, None, div_w.T],
                    [ -div_s,  None, None,   None],
                    [-asym,  -div_w, None,   None]], format = "csc")
    # fmt: on

    b_faces = sd.tags["domain_boundary_faces"]
    b_faces[np.isclose(sd.face_centers[1, :], 0)] = False
    b_faces = np.tile(b_faces, 2 * 2 + 2 * 1)
    b_faces = np.hstack((b_faces, np.zeros((2 + 1) * sd.num_cells, dtype=bool)))

    rhs = np.zeros(spp.shape[0])
    force = lambda x: np.array([0, -1])
    force_p0 = vec_p0.interpolate(sd, force)
    force_rhs = Mu @ force_p0

    split_idx = np.cumsum([vec_bdm1.ndof(sd), bdm1.ndof(sd), vec_p0.ndof(sd)])

    rhs[split_idx[1] : split_idx[2]] += force_rhs

    ls = pg.LinearSystem(spp, rhs)
    ls.flag_ess_bc(b_faces, np.zeros(spp.shape[0]))
    x = ls.solve()

    sigma, w, u, r = np.split(x, split_idx)

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


if __name__ == "__main__":
    main()
