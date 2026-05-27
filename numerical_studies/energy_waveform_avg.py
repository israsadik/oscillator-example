import compute_trajectories as ct
import numpy as np
import pandas as pd
from pathlib import Path

if __name__ == "__main__":
    """
    Runs partitioned experiment for oscillator example with different time stepping schemes
    using waveform iterations with time-averaged interface data.
    Outputs trajectories for energy/phase computation.
    """
    Path("results").mkdir(exist_ok=True)

    t_stop = 5
    N = 1000
    dt = t_stop / N
    timesteps = np.arange(0, t_stop + dt, dt)
    sampling_frequency = 1
    timesteps = timesteps[::sampling_frequency]

    method_name_and_func = {
        "newmark": ct.compute_newmark_trajectory,
        "alpha": ct.compute_alpha_trajectory,
        "erk2": ct.compute_erk2_trajectory,
        "erk4": ct.compute_erk4_trajectory,
        "sie": ct.compute_sie_trajectory,
        "mid": ct.compute_mid_trajectory,
        "trap": ct.compute_trap_trajectory,
    }

    coupling_scheme = "implicit-cps"

    for method_name, method_func in method_name_and_func.items():
        solution, iteration_counts = method_func(
            t_stop,
            N,
            coupling_scheme,
            interpolation_order="average",
        )

        trajectory_df = pd.DataFrame(index=timesteps)
        trajectory_df.index.name = "t"
        trajectory_df["u1"] = solution[0, ::sampling_frequency]
        trajectory_df["u2"] = solution[1, ::sampling_frequency]
        trajectory_df["v1"] = solution[2, ::sampling_frequency]
        trajectory_df["v2"] = solution[3, ::sampling_frequency]

        trajectory_filename = f"results/energy_{method_name}_waveform_avg.csv"
        trajectory_df.to_csv(trajectory_filename)
        print(f"Saved {trajectory_filename}")

        iter_df = pd.DataFrame({"iter_count": iteration_counts})
        iter_filename = f"results/iters_{method_name}_waveform_avg.csv"
        iter_df.to_csv(iter_filename, index=False)
        print(f"Saved {iter_filename}")