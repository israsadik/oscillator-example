"""
Run waveform relaxation experiments with iteration tracking

compares linear interpolation vs time averaging for all methods.
also runs different tolerances
"""

import numpy as np
import pandas as pd
import compute_partitioned_errors as cpe
from utility import max_norm

t_stop = 1
N_list = np.array([25 * 2**i for i in range(9)])
dt_list = np.array([t_stop / N for N in N_list])

method_name_and_func = {
    "newmark" : cpe.compute_newmark_error,
    "alpha": cpe.compute_alpha_error,
    "erk2": cpe.compute_erk2_error,
    "erk3": cpe.compute_erk3_error,
    "erk4": cpe.compute_erk4_error,
    "sie": cpe.compute_sie_error,
    "mid": cpe.compute_mid_error,
    "trap": cpe.compute_trap_error,
    "radau_iia3": cpe.compute_radau_iia3_error,
}

coupling_scheme = "implicit-cps"

#Linear VS Average, default tolerance

for interp_name, interp_order in [("linear", 1), ("average", "average")]:
    for method_name, method_func in method_name_and_func.items():
        rows = []
        for N in N_list:
            dt = t_stop / N
            error_matrix, iter_counts = method_func(
                t_stop, N, coupling_scheme,
                interpolation_order=interp_order,
            )
            err = max_norm(error_matrix)
            rows.append({
                "dt": dt,
                "error": err,
                "iter_avg": np.mean(iter_counts) if iter_counts else 0,
                "iter_max": np.max(iter_counts) if iter_counts else 0,
                "iter_total": np.sum(iter_counts) if iter_counts else 0,
            })
        df = pd.DataFrame(rows)
        fname = f"results/iterations_{method_name}_{interp_name}.csv"
        df.to_csv(fname, index=False)
        print(f"Saved {fname}")
        # Print summary
        print(f"  {method_name} ({interp_name}): order ~ {np.log2(df['error'].values[-2]/df['error'].values[-1]):.2f}, "
              f"avg iters: {df['iter_avg'].values[-1]:.1f}")

#Difference tolerances ( time averaging only)
        
print("\n=== Tolerance study (time averaging) ===")
tolerances = [1e-6, 1e-8, 1e-10, 1e-12]

for tol in tolerances:
    print(f"\n--- tol = {tol} ---")
    for method_name, method_func in method_name_and_func.items():
        rows = []
        for N in N_list:
            dt = t_stop / N
            error_matrix, iter_counts = method_func(
                t_stop, N, coupling_scheme,
                interpolation_order="average",
                tol=tol,
            )
            err = max_norm(error_matrix)
            rows.append({
                "dt": dt,
                "error": err,
                "iter_avg": np.mean(iter_counts) if iter_counts else 0,
                "iter_max": np.max(iter_counts) if iter_counts else 0,
                "iter_total": np.sum(iter_counts) if iter_counts else 0,
            })
        df = pd.DataFrame(rows)
        fname = f"results/iterations_{method_name}_avg_tol{tol:.0e}.csv"
        df.to_csv(fname, index=False)
        order = np.log2(df['error'].values[-2] / df['error'].values[-1])
        print(f"  {method_name}: order ~ {order:.2f}, "
              f"avg iters @ finest dt: {df['iter_avg'].values[-1]:.1f}, "
              f"max iters: {df['iter_max'].values[-1]}")

print("\nDone! All results in results/ folder.")
       