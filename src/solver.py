import os
import numpy as np
import scipy.sparse as sps
import pyamg

import pygeon as pg


class IterationCallback:
    def __init__(self):
        self.iteration_count = 0

    def __call__(self, xk):
        self.iteration_count += 1

    def get_iteration_count(self):
        return self.iteration_count


class Solver:
    def __init__(self, dim, key):
        self.dim = dim
        self.key = key

        self.vec_bdm1 = pg.VecBDM1(key)
        self.bdm1 = pg.BDM1(key)

        self.vec_p0 = pg.VecPwConstants(key)
        self.p0 = pg.PwConstants(key)

        self.vec_l1 = pg.VecLagrange1(key)
        self.l1 = pg.Lagrange1(key)

        self.p1 = pg.PwLinears(key)
        self.vec_p1 = pg.VecPwLinears(key)

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

    def solve_problem(self, data, data_pb):
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
        div_s, asym, div_w = self.build_diff(M_u, M_r)

        # Build the global system
        A = sps.block_diag([M_s, M_w])
        B = sps.block_array([[-div_s, None], [asym, -div_w]])
        spp = sps.block_array([[A, -B.T], [B, None]]).tocsc()

        # Assemble the source terms
        u_bc, r_bc, u_for, r_for = self.build_bc_for(M_u, M_r, data_pb)

        # Assemble the right-hand side
        rhs = np.zeros(spp.shape[0])
        rhs[: split_idx[0]] += u_bc
        rhs[split_idx[0] : split_idx[1]] += r_bc
        rhs[split_idx[1] : split_idx[2]] += u_for
        rhs[split_idx[2] :] += r_for

        ls = pg.LinearSystem(spp, rhs)
        x = ls.solve()
        s, w, u, r = np.split(x, split_idx)

        # compute the error
        err = self.compute_err(s, w, u, r, data, data_pb)
        h = np.amax(self.sd.cell_diameters())

        return h, *err, *dofs

    def solve_problem_lumped(self, data, data_pb):
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
        div_s, asym, div_w = self.build_diff(M_u, M_r)

        # Build the global matrices
        A = sps.block_diag([M_s, M_w]).tocsc()
        B = sps.block_array([[-div_s, None], [asym, -div_w]]).tocsr()

        # solve the block diagonal system
        ls = pg.LinearSystem(A, B.T)
        inv_ABT = ls.solve(pg.block_diag_solver)
        spp = B @ inv_ABT

        # Assemble the source terms
        u_bc, r_bc, u_for, r_for = self.build_bc_for(M_u, M_r, data_pb)
        x_bc = np.concatenate([u_bc, r_bc])

        # Assemble the right-hand side
        rhs = np.zeros(spp.shape[0])
        rhs[: split_idx[0]] += u_for
        rhs[split_idx[0] :] += r_for

        ls = pg.LinearSystem(A, x_bc)
        x_bc = ls.solve(pg.block_diag_solver_dense)
        rhs -= B @ x_bc

        P_op = self.create_precond(spp)

        callback = IterationCallback()
        x, _ = sps.linalg.minres(spp, rhs, M=P_op, callback=callback, rtol=1e-7)
        it = callback.get_iteration_count()
        u, r = np.split(x, split_idx)

        y = inv_ABT @ x + x_bc
        s, w = np.split(y, [self.dis_s.ndof(self.sd)])

        # compute the error
        err = self.compute_err(s, w, u, r, data, data_pb)
        h = np.amax(self.sd.cell_diameters())

        return h, *err, *dofs, it

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

    def build_diff(self, M_u, M_r):
        # Build the differential matrices
        div_s = M_u @ self.dis_s.assemble_diff_matrix(self.sd)
        asym = M_r @ self.dis_s.assemble_asym_matrix(self.sd)
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

        err_s = self.dis_s.error_l2(self.sd, s, s_ex, data=data)
        err_w = self.dis_w.error_l2(self.sd, w, w_ex, data=data)
        err_u = self.dis_u.error_l2(self.sd, u, u_ex)
        err_r = self.dis_r.error_l2(self.sd, r, r_ex)

        return err_s, err_w, err_u, err_r

    def create_precond(self, spp):
        # solve the saddle point problem using minres and a preconditioner
        rt0 = pg.RT0(self.key)
        L = rt0.assemble_lumped_matrix(self.sd)
        div = rt0.assemble_diff_matrix(self.sd)

        # factorize the single and then use it in the pot
        inv_P = sps.csr_matrix(div @ sps.linalg.spsolve(L, div.T.tocsc()))
        inv_P.indices = inv_P.indices.astype(np.int32)
        inv_P.indptr = inv_P.indptr.astype(np.int32)

        amg_u = pyamg.smoothed_aggregation_solver(inv_P)
        P_P0 = amg_u.aspreconditioner()

        num = 3 if self.dim == 2 else 6

        def matvec(x):
            return np.array(
                [P_P0.matvec(x_part) for x_part in np.array_split(x, num)]
            ).ravel()

        return sps.linalg.LinearOperator(spp.shape, matvec=matvec)


class SolverBDM1_L1(Solver):
    def create_family(self):
        self.dis_s = self.vec_bdm1
        self.dis_w = self.bdm1 if self.dim == 2 else self.vec_bdm1
        self.dis_u = self.vec_p0
        self.dis_r = self.l1 if self.dim == 2 else self.vec_l1

    def build_diff(self, M_u, M_r):
        if self.dim == 2:
            M_p1 = self.p1.assemble_lumped_matrix(self.sd)
            proj_p0 = self.p0.proj_to_pwLinears(self.sd)
        else:
            M_p1 = self.vec_p1.assemble_lumped_matrix(self.sd)
            proj_p0 = self.vec_p0.proj_to_pwLinears(self.sd)

        proj_l1 = self.dis_r.proj_to_pwLinears(self.sd)
        div_s = M_u @ self.dis_s.assemble_diff_matrix(self.sd)

        asym_op = self.dis_s.assemble_asym_matrix(self.sd, as_pwconstant=False)
        asym = proj_l1.T @ M_p1 @ asym_op

        div_w_op = self.dis_w.assemble_diff_matrix(self.sd)
        div_w = proj_l1.T @ M_p1 @ proj_p0 @ div_w_op

        return div_s, asym, div_w

    def build_bc_for(self, M_u, M_r, data_pb):
        f_u, f_r = data_pb["f_u"], data_pb["f_r"]

        if self.dim == 2:
            M_p1 = self.p1.assemble_lumped_matrix(self.sd)
            r_interp = self.p1.interpolate(self.sd, f_r)
        else:
            M_p1 = self.vec_p1.assemble_lumped_matrix(self.sd)
            r_interp = self.vec_p1.interpolate(self.sd, f_r)

        proj_l1 = self.dis_r.proj_to_pwLinears(self.sd)

        # Assemble the source terms
        u_for = M_u @ self.dis_u.interpolate(self.sd, f_u)
        r_for = proj_l1.T @ M_p1 @ r_interp

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

        err_s = self.dis_s.error_l2(self.sd, s, s_ex, data=data)
        err_w = self.dis_w.error_l2(self.sd, w, w_ex, data=data)
        err_u = self.dis_u.error_l2(self.sd, u, u_ex)
        r_l2 = self.dis_r.proj_to_lagrange2(self.sd) @ r
        if self.dim == 2:
            err_r = self.l2.error_l2(self.sd, r_l2, r_ex)
        else:
            err_r = self.vec_l2.error_l2(self.sd, r_l2, r_ex)

        return err_s, err_w, err_u, err_r

    def create_precond(self, spp):
        # solve the saddle point problem using minres and a preconditioner
        rt0 = pg.RT0(self.key)
        L = rt0.assemble_lumped_matrix(self.sd)
        div = rt0.assemble_diff_matrix(self.sd)

        # factorize the single and then use it in the pot
        inv_Pu = sps.csr_matrix(div @ sps.linalg.spsolve(L, div.T.tocsc()))
        inv_Pu.indices = inv_Pu.indices.astype(np.int32)
        inv_Pu.indptr = inv_Pu.indptr.astype(np.int32)

        amg_u = pyamg.smoothed_aggregation_solver(inv_Pu)
        P_P0 = amg_u.aspreconditioner()

        l1_stiff = self.l1.assemble_stiff_matrix(self.sd)
        inv_Pr = sps.csr_matrix(l1_stiff)

        amg_r = pyamg.smoothed_aggregation_solver(inv_Pr)
        P_L1 = amg_r.aspreconditioner()

        num_r = 1 if self.dim == 2 else 3

        def matvec(x):
            x_p0 = x[: self.vec_p0.ndof(self.sd)]
            u = np.array(
                [P_P0.matvec(x_part) for x_part in np.array_split(x_p0, self.sd.dim)]
            ).ravel()

            x_l1 = x[self.vec_p0.ndof(self.sd) :]
            r = np.array(
                [P_L1.matvec(x_part) for x_part in np.array_split(x_l1, num_r)]
            ).ravel()

            return np.concatenate([u, r])

        return sps.linalg.LinearOperator(spp.shape, matvec=matvec)
