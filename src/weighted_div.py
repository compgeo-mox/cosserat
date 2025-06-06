import numpy as np
import pygeon as pg
import scipy.sparse as sps


def compute_weighted_div(
    sd: pg.Grid,
    ell_func: callable,
    stress_space: pg.Discretization,
):
    B = compute_weighted_div_scalar(sd, ell_func, stress_space)
    if sd.dim == 2:
        return B
    elif sd.dim == 3:
        return sps.block_diag([B] * 3).tocsc()
    else:
        raise ValueError("Dimension needs to be 2 or 3.")


def compute_weighted_div_scalar(
    sd: pg.Grid,
    ell_func: callable,
    stress_space: pg.Discretization,
):
    """
    Some additional functionality to calculate the term
    div ell w = ell div w + grad ell dot w
              = term_1 + term_2

    The output is in terms of
        - P1 if the stress space is BDM1
        - P2 if the stress space is RT1
    """
    # If we accidentally give the VecHDiv class, then we extract the base_discr
    if isinstance(stress_space, pg.VecDiscretization):
        stress_space = stress_space.base_discr

    P0 = pg.PwConstants()
    VecP0 = pg.VecPwConstants()
    P1 = pg.PwLinears()
    VecP1 = pg.VecPwLinears()

    # We first compute term_1 = ell * div w
    # Interpolate ell on the L1 space
    L1 = pg.Lagrange1()
    ell = L1.interpolate(sd, ell_func)

    div = stress_space.assemble_diff_matrix(sd)
    ell_P1 = L1.proj_to_pwLinears(sd) @ ell

    if isinstance(stress_space, pg.BDM1):
        div = P0.proj_to_pwLinears(sd) @ div
        term_1 = ell_P1[:, None] * div

    else:  # VecRT1
        div = P1.proj_to_pwQuadratics(sd) @ div
        ell_P2 = P1.proj_to_pwQuadratics(sd) @ ell_P1

        term_1 = ell_P2[:, None] * div

    # Next, we compute term_1 = grad(ell) dot w
    # Interpolate grad ell on the P0 space
    gradL1 = L1.get_range_discr_class(sd.dim)
    grad_ell = L1.assemble_diff_matrix(sd) @ ell
    grad_ell_P0 = gradL1().eval_at_cell_centers(sd) @ grad_ell
    grad_ell_P0 *= np.tile(sd.cell_volumes, 3)

    if sd.dim == 2:  # The 2D differential is a rotated gradient, so we rotate back...
        grad_ell_P0 = np.concatenate(
            (-grad_ell_P0[sd.num_cells : sd.num_cells * 2], grad_ell_P0[: sd.num_cells])
        )

    # Interpolate to the P1 space
    grad_ell_P1 = VecP0.proj_to_pwLinears(sd) @ grad_ell_P0

    if isinstance(stress_space, pg.BDM1):
        grad_ell_i = np.split(grad_ell_P1, sd.dim)
        dot_prod = sps.hstack([sps.diags_array(g_ell) for g_ell in grad_ell_i])

        term_2 = dot_prod @ stress_space.proj_to_VecPwLinears(sd)

    else:  # VecRT1
        grad_ell_P2 = VecP1.proj_to_pwQuadratics(sd) @ grad_ell_P1
        grad_ell_i = np.split(grad_ell_P2, sd.dim)
        dot_prod = sps.hstack([sps.diags_array(g_ell) for g_ell in grad_ell_i])

        term_2 = dot_prod @ stress_space.proj_to_VecPwQuadratics(sd)

    return (term_1 + term_2).tocsc()


if __name__ == "__main__":
    # Examples
    dim = 3

    def ell(x):  # Function describing the length scale ell
        return np.max((x[0], 0.5))

    bdm1 = pg.VecBDM1()
    rt1 = pg.VecRT1()

    sd = pg.unit_grid(dim, 0.5, as_mdg=False)
    sd.compute_geometry()

    div_bdm1 = compute_weighted_div(sd, ell, bdm1)
    div_rt1 = compute_weighted_div(sd, ell, rt1)

    L1 = pg.Lagrange1() if dim == 2 else pg.VecLagrange1()
    P1 = pg.PwLinears() if dim == 2 else pg.VecPwLinears()
    P2 = pg.PwQuadratics() if dim == 2 else pg.VecPwQuadratics()

    M1 = P1.assemble_mass_matrix(sd)
    ML1 = P1.assemble_lumped_matrix(sd)
    M2 = P2.assemble_mass_matrix(sd)

    # EXAMPLES
    B_bdm1_p1 = M1 @ div_bdm1
    B_bdm1_l1 = L1.proj_to_pwLinears(sd).T @ ML1 @ div_bdm1
    B_rt1_p1 = P1.proj_to_pwQuadratics(sd).T @ M2 @ div_rt1
    B_rt1_l1 = L1.proj_to_pwLinears(sd).T @ B_rt1_p1
