import compute_partitioned_errors as cpe
import numpy as np
import pandas as pd
from utility import max_norm, comment_meta_information

if __name__ == "__main__":
    t_stop = 1
    N_list = np.array([25 * 2**i for i in range(9)])
    dt_list = np.array([t_stop / N for N in N_list])

    errors_df = pd.DataFrame(index=dt_list)
    errors_df.index.name = "dt"

    method_name_and_func = {
        "newmark": cpe.compute_newmark_error,
        "alpha": cpe.compute_alpha_error,
        "erk4": cpe.compute_erk4_error,
        "sie": cpe.compute_sie_error,
        "mid": cpe.compute_mid_error,
        "trap": cpe.compute_trap_error,
        "erk2": cpe.compute_erk2_error,
    }

    coupling_scheme = "cps"  # no iterations

    for method_name, method_func in method_name_and_func.items():
        errors_df["error"] = np.array(
            [
                max_norm(
                    method_func(
                        t_stop,
                        N,
                        coupling_scheme,
                        interpolation_order="lagged_average",
                    )[0]
                )
                for N in N_list
            ]
        )
        filename = f"results/partitioned_{method_name}_lagged_average.csv"
        errors_df.to_csv(filename)
        comment_meta_information(method_name + "_lagged_average", __file__, filename)

        print(f"Saved {filename}")