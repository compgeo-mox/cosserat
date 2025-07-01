import sys

sys.path.append("./src")

import strong_solution_cosserat_elasticity_example_3 as ss


def cosserat_exact(param, dim):
    def sigma(pt):
        return ss.stress(param, dim)(*pt)

    def u(pt):
        return ss.displacement(param, dim)(*pt)

    def r(pt):
        return ss.rotation(param, dim)(*pt)

    def w(pt):
        return ss.couple_stress_scaled(param, dim)(*pt)

    def f_r(pt):
        return ss.rhs_scaled(param, dim)(*pt)

    return {
        "s_ex": sigma,
        "w_ex": w,
        "u_ex": u,
        "r_ex": r,
        "f_r": f_r,
    }
