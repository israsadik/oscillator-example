import compute_partitioned_errors as cpe
import numpy as np
import pandas as pd
from utility import comment_meta_information, max_norm

if __name__ == "__main__":
    t_stop = 1
    N_list = np.array([25 * 2**i for i in range(9)])
    dt_list = np.array([t_stop / N for N in N_list])

    coupling_scheme = "implicit-cps"

    for interp_name, interp_order in [("linear", 1), ("average", "average")]:
        rows = []

        for N in N_list:
            dt = t_stop / N

            error_matrix, iter_counts = cpe.compute_exact_error(
                t_stop,
                N,
                coupling_scheme,
                interpolation_order=interp_order,
            )

            rows.append({
                "dt": dt,
                "error": max_norm(error_matrix),
                "iter_avg": np.mean(iter_counts),
                "iter_max": np.max(iter_counts),
            })

        df = pd.DataFrame(rows)
        filename = f"results/partitioned_exact_{interp_name}.csv"
        df.to_csv(filename, index=False)

        print(f"Saved {filename}")
        print(df)
        