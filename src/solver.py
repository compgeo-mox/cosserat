import os
import numpy as np
import scipy.sparse as sps

import porepy as pp
import pygeon as pg
from weighted_div import compute_weighted_div


class Solver:
    def __init__(self, dim, key):
        self.dim = dim
        self.key = key

        self.vec_bdm1 = pg.VecBDM1(key)
        self.bdm1 = pg.BDM1(key)

        self.vec_rt1 = pg.VecRT1(key)
        self.rt1 = pg.RT1(key)

        self.vec_p0 = pg.VecPwConstants(key)
        self.p0 = pg.PwConstants(key)

        self.vec_l1 = pg.VecLagrange1(key)
        self.l1 = pg.Lagrange1(key)

        self.p1 = pg.PwLinears(key)
        self.vec_p1 = pg.VecPwLinears(key)

        self.p2 = pg.PwQuadratics(key)
        self.vec_p2 = pg.VecPwQuadratics(key)

        self.l2 = pg.Lagrange2(key)
        self.vec_l2 = pg.VecLagrange2(key)

        self.dis_s = None
        self.dis_w = None
        self.dis_u = None
        self.dis_r = None

    def create_grid(self, mesh_size, folder):
        mesh_file_name = os.path.join(folder, "grid.msh")
        self.sd = pg.unit_grid(
            self.dim, mesh_size, as_mdg=False, file_name=mesh_file_name
        )
        self.sd.compute_geometry()

    def solve_problem(self, data, data_pb, tol=1e-9):
        # Compute the number of dofs
        dofs = np.array(
            [
                self.dis_s.ndof(self.sd),
                self.dis_w.ndof(self.sd),
                self.dis_u.ndof(self.sd),
                self.dis_r.ndof(self.sd),
            ]
        )
        split_idx = np.cumsum(dofs[:-1])

        # Build the mass matrices
        M_s, M_w, M_u, M_r = self.build_mass(data, is_lumped=False)

        # Build the diff matrices
        div_s, asym, div_w = self.build_diff(data_pb, M_u, M_r)

        # Build the global system
        A = sps.block_diag([M_s, M_w])
        B = sps.block_array([[-div_s, None], [asym, -div_w]])
        spp = sps.block_array([[A, -B.T], [-B, None]]).tocsc()

        # Assemble the source terms
        r_for = self.build_bc_for(M_u, M_r, data_pb)

        # Assemble the right-hand side
        rhs = np.zeros(spp.shape[0])
        rhs[split_idx[2] :] -= r_for

        x, _ = sps.linalg.bicgstab(spp, rhs)
        s, w, u, r = np.split(x, split_idx)

        # compute the error
        err = self.compute_err(s, w, u, r, data, data_pb)
        h = np.amax(self.sd.cell_diameters())

        return h, *err, *dofs

    def solve_problem_lumped(self, data, data_pb, tol=1e-9):
        # Step 1: Solve the lumped system
        info = self.step1(data, data_pb)
        return self.step2(*info, tol)

    def step1(self, data, data_pb):
        # Compute the number of dofs
        dofs = np.array(
            [
                self.dis_u.ndof(self.sd),
                self.dis_r.ndof(self.sd),
            ]
        )
        split_idx = np.cumsum(dofs[:-1])

        # Build the mass matrices
        M_s, M_w, M_u, M_r = self.build_mass(data, is_lumped=True)

        # Build the diff matrices
        div_s, asym, div_w = self.build_diff(data_pb, M_u, M_r)

        # Build the global matrices
        A = sps.block_diag([M_s, M_w]).tocsc()
        B = sps.block_array([[-div_s, None], [asym, -div_w]]).tocsr()

        # solve the block diagonal system
        ls = pg.LinearSystem(A, B.T)
        inv_ABT = ls.solve(pg.block_diag_solver)

        # Assemble the source terms
        r_for, u_for = self.build_bc_for(M_u, M_r, data_pb)

        # Assemble the right-hand side
        rhs = np.zeros(dofs.sum())
        rhs[: split_idx[0]] += u_for
        rhs[split_idx[0] :] += r_for

        return inv_ABT, B, rhs, split_idx, data, data_pb, dofs

    def step2(self, inv_ABT, B, rhs, split_idx, data, data_pb, dofs, tol):
        # solve the saddle point problem using bcgstab
        spp = B @ inv_ABT

        # x, _ = sps.linalg.bicgstab(spp, rhs, rtol=tol)
        x = sps.linalg.spsolve(spp, rhs)
        u, r = np.split(x, split_idx)
        s, w = np.split(inv_ABT @ x, [self.dis_s.ndof(self.sd)])

        # compute the error
        err = self.compute_err(s, w, u, r, data, data_pb)
        h = np.amax(self.sd.cell_diameters())

        return h, *err, *dofs

    def build_mass(self, data, is_lumped):
        if is_lumped:
            M_s = self.dis_s.assemble_lumped_matrix_cosserat(self.sd, data)
            if self.dim == 2:
                M_w = self.dis_w.assemble_lumped_matrix(self.sd, data)
            else:
                M_w = self.dis_w.assemble_lumped_matrix_cosserat(self.sd, data)
        else:
            M_s = self.dis_s.assemble_mass_matrix_cosserat(self.sd, data)
            if self.dim == 2:
                M_w = self.dis_w.assemble_mass_matrix(self.sd, data)
            else:
                M_w = self.dis_w.assemble_mass_matrix_cosserat(self.sd, data)

        M_u = self.dis_u.assemble_mass_matrix(self.sd)
        M_r = self.dis_r.assemble_mass_matrix(self.sd)

        return M_s, M_w, M_u, M_r


class SolverBDM1_P0(Solver):
    def create_family(self):
        self.dis_s = self.vec_bdm1
        self.dis_w = self.bdm1 if self.dim == 2 else self.vec_bdm1
        self.dis_u = self.vec_p0
        self.dis_r = self.p0 if self.dim == 2 else self.vec_p0

    def build_diff(self, data_pb, M_u, M_r):
        # Build the differential matrices
        div_s = M_u @ self.dis_s.assemble_diff_matrix(self.sd)
        asym = M_r @ self.dis_s.assemble_asym_matrix(self.sd, as_pwconstant=True)
        div_w = M_r @ self.dis_w.assemble_diff_matrix(self.sd)

        return div_s, asym, div_w

    def build_bc_for(self, M_u, M_r, data_pb):
        f_u, f_r = data_pb["f_u"], data_pb["f_r"]

        # Assemble the source terms
        u_for = M_u @ self.dis_u.interpolate(self.sd, f_u)
        r_for = M_r @ self.dis_r.interpolate(self.sd, f_r)

        # Assemble the boundary conditions
        bd_faces = self.sd.tags["domain_boundary_faces"]
        u_ex, r_ex = data_pb["u_ex"], data_pb["r_ex"]
        u_bc = self.dis_s.assemble_nat_bc(self.sd, u_ex, bd_faces)
        r_bc = self.dis_w.assemble_nat_bc(self.sd, r_ex, bd_faces)

        return u_bc, r_bc, u_for, r_for

    def compute_err(self, s, w, u, r, data, data_pb):
        # compute the error
        s_ex, w_ex = data_pb["s_ex"], data_pb["w_ex"]
        u_ex, r_ex = data_pb["u_ex"], data_pb["r_ex"]

        err_s = self.dis_s.error_l2(self.sd, s, s_ex)
        err_w = self.dis_w.error_l2(self.sd, w, w_ex)
        err_u = self.dis_u.error_l2(self.sd, u, u_ex)
        err_r = self.dis_r.error_l2(self.sd, r, r_ex)

        return err_s, err_w, err_u, err_r


class SolverBDM1_L1(Solver):
    def create_family(self):
        self.dis_s = self.vec_bdm1
        self.dis_w = self.bdm1 if self.dim == 2 else self.vec_bdm1
        self.dis_u = self.vec_p0
        self.dis_r = self.l1 if self.dim == 2 else self.vec_l1

    def build_diff(self, data_pb, M_u, M_r):
        ell = data_pb["ell"]

        if self.dim == 2:
            M_p1 = self.p1.assemble_lumped_matrix(self.sd)
        else:
            M_p1 = self.vec_p1.assemble_lumped_matrix(self.sd)

        proj_l1 = self.dis_r.proj_to_pwLinears(self.sd)
        div_s = M_u @ self.dis_s.assemble_diff_matrix(self.sd)

        asym_op = self.dis_s.assemble_asym_matrix(self.sd, as_pwconstant=False)
        asym = proj_l1.T @ M_p1 @ asym_op

        div_op = compute_weighted_div(self.sd, ell, self.vec_bdm1)
        div_w = proj_l1.T @ M_p1 @ div_op

        return div_s, asym, div_w

    def build_bc_for(self, M_u, M_r, data_pb):
        f_r = data_pb["f_r"]
        f_u = data_pb["f_u"]

        if self.dim == 2:
            M_p1 = self.p1.assemble_lumped_matrix(self.sd)
            f_r_int = self.p1.interpolate(self.sd, f_r)
        else:
            M_p1 = self.vec_p1.assemble_lumped_matrix(self.sd)
            f_r_int = self.vec_p1.interpolate(self.sd, f_r)

        proj_l1 = self.dis_r.proj_to_pwLinears(self.sd)

        # Assemble the source terms
        r_for = proj_l1.T @ M_p1 @ f_r_int

        u_for = M_u @ self.dis_u.interpolate(self.sd, f_u)
        return r_for, u_for

    def compute_err(self, s, w, u, r, data, data_pb):
        # compute the error
        s_ex, w_ex = data_pb["s_ex"], data_pb["w_ex"]
        u_ex, r_ex = data_pb["u_ex"], data_pb["r_ex"]

        err_s = self.dis_s.error_l2(self.sd, s, s_ex)
        err_w = self.dis_w.error_l2(self.sd, w, w_ex)
        err_u = self.dis_u.error_l2(self.sd, u, u_ex)
        r_l2 = self.dis_r.proj_to_lagrange2(self.sd) @ r
        if self.dim == 2:
            err_r = self.l2.error_l2(self.sd, r_l2, r_ex)
        else:
            err_r = self.vec_l2.error_l2(self.sd, r_l2, r_ex)

        u_ex_int = self.dis_u.interpolate(self.sd, u_ex)
        r_ex_int = self.dis_r.interpolate(self.sd, r_ex)

        proj_u = self.vec_p0.eval_at_cell_centers(self.sd)
        u = (proj_u @ u).reshape((2, -1))
        u = np.vstack((u, np.zeros(u.shape[1])))

        u_ex_int = (proj_u @ u_ex_int).reshape((2, -1))
        u_ex_int = np.vstack((u_ex_int, np.zeros(u_ex_int.shape[1])))

        save = pp.Exporter(self.sd, "sol", folder_name="/home/elle/")
        save.write_vtu(
            [("u", u), ("u_ex", u_ex_int)], data_pt=[("r", r), ("r_ex", r_ex_int)]
        )

        return err_s, err_w, err_u, err_r


class SolverRT1_L1(Solver):
    def create_family(self):
        self.dis_s = self.vec_rt1
        self.dis_w = self.rt1 if self.dim == 2 else self.vec_rt1
        self.dis_u = self.vec_p1
        self.dis_r = self.l1 if self.dim == 2 else self.vec_l1

    def build_diff(self, data_pb, M_u, M_r):
        proj_l1_p2 = self.dis_r.proj_to_lagrange2(self.sd)

        if self.dim == 2:
            M_p2 = self.p2.assemble_lumped_matrix(self.sd)
            M_p1 = self.p1.assemble_mass_matrix(self.sd)
            proj_l1_p2 = self.p2.proj_to_pwQuadratics(self.sd) @ proj_l1_p2

        else:
            M_p2 = self.vec_p2.assemble_lumped_matrix(self.sd)
            M_p1 = self.vec_p1.assemble_mass_matrix(self.sd)
            proj_l1_p2 = self.vec_l2.proj_to_pwQuadratics(self.sd) @ proj_l1_p2

        div_s = M_u @ self.dis_s.assemble_diff_matrix(self.sd)

        asym_op = self.dis_s.assemble_asym_matrix(self.sd)
        asym = proj_l1_p2.T @ M_p2 @ asym_op

        proj_l1_p1 = self.dis_r.proj_to_pwLinears(self.sd)
        div_w_op = self.dis_w.assemble_diff_matrix(self.sd)

        div_w = proj_l1_p1.T @ M_p1 @ div_w_op

        return div_s, asym, div_w

    def build_bc_for(self, M_u, M_r, data_pb):
        f_u, f_r = data_pb["f_u"], data_pb["f_r"]
        proj_l1_p2 = self.dis_r.proj_to_lagrange2(self.sd)

        if self.dim == 2:
            M_p2 = self.p2.assemble_lumped_matrix(self.sd)
            r_interp = self.p2.interpolate(self.sd, f_r)
            proj_l1_p2 = self.p2.proj_to_pwQuadratics(self.sd) @ proj_l1_p2
        else:
            M_p2 = self.vec_p2.assemble_lumped_matrix(self.sd)
            r_interp = self.vec_p2.interpolate(self.sd, f_r)
            proj_l1_p2 = self.vec_l2.proj_to_pwQuadratics(self.sd) @ proj_l1_p2

        # Assemble the source terms
        u_for = M_u @ self.dis_u.interpolate(self.sd, f_u)
        r_for = proj_l1_p2.T @ M_p2 @ r_interp

        # Assemble the boundary conditions
        u_ex, r_ex = data_pb["u_ex"], data_pb["r_ex"]
        bd_faces = self.sd.tags["domain_boundary_faces"]
        u_bc = self.dis_s.assemble_nat_bc(self.sd, u_ex, bd_faces)
        r_bc = self.dis_w.assemble_nat_bc(self.sd, r_ex, bd_faces)

        return u_bc, r_bc, u_for, r_for

    def compute_err(self, s, w, u, r, data, data_pb):
        # compute the error
        s_ex, w_ex = data_pb["s_ex"], data_pb["w_ex"]
        u_ex, r_ex = data_pb["u_ex"], data_pb["r_ex"]

        err_s = self.dis_s.error_l2(self.sd, s, s_ex)
        err_w = self.dis_w.error_l2(self.sd, w, w_ex)
        err_u = self.dis_u.error_l2(self.sd, u, u_ex)
        r_l2 = self.dis_r.proj_to_lagrange2(self.sd) @ r
        if self.dim == 2:
            err_r = self.p2.error_l2(self.sd, r_l2, r_ex)
        else:
            err_r = self.vec_l2.error_l2(self.sd, r_l2, r_ex)

        return err_s, err_w, err_u, err_r


class SolverRT1_P1(Solver):
    def create_family(self):
        self.dis_s = self.vec_rt1
        self.dis_w = self.rt1 if self.dim == 2 else self.vec_rt1
        self.dis_u = self.vec_p1
        self.dis_r = self.p1 if self.dim == 2 else self.vec_p1

    def build_diff(self, data_pb, M_u, M_r):
        proj_p1_p2 = self.dis_r.proj_to_pwQuadratics(self.sd)

        if self.dim == 2:
            M_p2 = self.p2.assemble_mass_matrix(self.sd)
            M_p1 = self.p1.assemble_mass_matrix(self.sd)

        else:
            M_p2 = self.vec_p2.assemble_mass_matrix(self.sd)
            M_p1 = self.vec_p1.assemble_mass_matrix(self.sd)

        div_s = M_u @ self.dis_s.assemble_diff_matrix(self.sd)

        asym_op = self.dis_s.assemble_asym_matrix(self.sd)
        asym = proj_p1_p2.T @ M_p2 @ asym_op

        div_w_op = self.dis_w.assemble_diff_matrix(self.sd)
        div_w = M_p1 @ div_w_op

        return div_s, asym, div_w

    def build_bc_for(self, M_u, M_r, data_pb):
        f_u, f_r = data_pb["f_u"], data_pb["f_r"]

        if self.dim == 2:
            M_p1 = self.p1.assemble_mass_matrix(self.sd)
            r_interp = self.p1.interpolate(self.sd, f_r)
        else:
            M_p1 = self.vec_p1.assemble_mass_matrix(self.sd)
            r_interp = self.vec_p1.interpolate(self.sd, f_r)

        # Assemble the source terms
        u_for = M_u @ self.dis_u.interpolate(self.sd, f_u)
        r_for = M_p1 @ r_interp

        # Assemble the boundary conditions
        u_ex, r_ex = data_pb["u_ex"], data_pb["r_ex"]
        bd_faces = self.sd.tags["domain_boundary_faces"]
        u_bc = self.dis_s.assemble_nat_bc(self.sd, u_ex, bd_faces)
        r_bc = self.dis_w.assemble_nat_bc(self.sd, r_ex, bd_faces)

        return u_bc, r_bc, u_for, r_for

    def compute_err(self, s, w, u, r, data, data_pb):
        # compute the error
        s_ex, w_ex = data_pb["s_ex"], data_pb["w_ex"]
        u_ex, r_ex = data_pb["u_ex"], data_pb["r_ex"]

        err_s = self.dis_s.error_l2(self.sd, s, s_ex)
        err_w = self.dis_w.error_l2(self.sd, w, w_ex)
        err_u = self.dis_u.error_l2(self.sd, u, u_ex)

        r_p2 = self.dis_r.proj_to_pwQuadratics(self.sd) @ r
        if self.dim == 2:
            err_r = self.p2.error_l2(self.sd, r_p2, r_ex)
        else:
            err_r = self.vec_p2.error_l2(self.sd, r_p2, r_ex)

        return err_s, err_w, err_u, err_r

    def create_grid(self, mesh_size, folder):
        mesh_file_name = os.path.join(folder, "grid.msh")
        self.sd = pg.unit_grid(
            self.dim, mesh_size, as_mdg=False, file_name=mesh_file_name
        )
        self.sd.compute_geometry()
        self.sd = pg.barycentric_split(self.sd)
        self.sd.compute_geometry()
