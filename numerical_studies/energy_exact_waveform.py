import numpy as np
import pandas as pd
import run_partitioned_simulation as rps

t_stop = 5
N = 1000
dt = t_stop / N
timesteps = np.arange(0, t_stop + dt, dt)

coupling_scheme = "implicit-cps"

cases = {
    "linear": 1,
    "avg": "average",
}

for case_name, interpolation_order in cases.items():
    num_sol, iter_counts = rps.partitioned_exact(
        t_stop,
        N,
        coupling_scheme,
        interpolation_order=interpolation_order,
    )

    trajectory_df = pd.DataFrame(index=timesteps)
    trajectory_df.index.name = "t"

    trajectory_df["u1"] = num_sol[0]
    trajectory_df["u2"] = num_sol[1]
    trajectory_df["v1"] = num_sol[2]
    trajectory_df["v2"] = num_sol[3]

    filename = f"results/energy_exact_waveform_{case_name}.csv"
    trajectory_df.to_csv(filename)

    print(f"Saved {filename}")