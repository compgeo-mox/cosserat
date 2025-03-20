import numpy as np


def order(error, diam):
    return np.log(error[:-1] / error[1:]) / np.log(diam[:-1] / diam[1:])

def array_to_latex(arr):
    latex_str = "\\begin{table}[h]\n\\centering\n"
    latex_str += "\\begin{tabular}{" + "|c" * arr.shape[1] + "|}\n\\hline\n"
    
    intestation = ["$h$", "$err_\sigma$", "ord", "$err_w$", "ord", "$err_u$", "ord", "$err_r$", "ord"]

    formatted_rows = [
        " & ".join("{:.2e}".format(num) for num in row) for row in arr
    ]
    
    latex_str +=  " & ".join(intestation)
    latex_str += " \\\\\n\\hline\n"
    latex_str += " \\\\\n\\hline\n".join(formatted_rows)
    latex_str += " \\\\\n\\hline\n\\end{tabular}\n\\end{table}"
    
    return latex_str.replace("-1.00e+00", "-")

def make_summary(errs):
    order_sigma = order(errs[:, 1], errs[:, 0])
    order_w = order(errs[:, 2], errs[:, 0])
    order_u = order(errs[:, 3], errs[:, 0])
    order_r = order(errs[:, 4], errs[:, 0])

    # reorder the arrays for the latex table
    order_sigma = np.hstack([-1, order_sigma])
    order_w = np.hstack([-1, order_w])
    order_u = np.hstack([-1, order_u])
    order_r = np.hstack([-1, order_r])

    errs = np.insert(errs, 2, order_sigma, axis=1)
    errs = np.insert(errs, 4, order_w, axis=1)
    errs = np.insert(errs, 6, order_u, axis=1)
    errs = np.insert(errs, 8, order_r, axis=1)

    return array_to_latex(errs)