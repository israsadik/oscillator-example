import pandas as pd
import matplotlib.pyplot as plt
from oscillator import compute_energy, analytical_solution


def clean_axes(ax):
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.grid(False)


def plot_energy(df_lin, df_avg, filename):
    t_stop = df_lin["t"].iloc[-1]
    N = len(df_lin["t"]) - 1

    exact_sol = analytical_solution(t_stop, N)

    E_exact = compute_energy(exact_sol[0], exact_sol[1],
                             exact_sol[2], exact_sol[3])
    E_lin = compute_energy(df_lin["u1"], df_lin["u2"],
                           df_lin["v1"], df_lin["v2"])
    E_avg = compute_energy(df_avg["u1"], df_avg["u2"],
                           df_avg["v1"], df_avg["v2"])

    fig, ax = plt.subplots(figsize=(5.0, 3.2))

    ax.plot(df_lin["t"], E_exact, "k--", label="analytical")
    ax.plot(df_lin["t"], E_lin, label="linear interpolation")
    ax.plot(df_avg["t"], E_avg, label="time averaging")

    ax.set_xlabel(r"$t$")
    ax.set_ylabel("Energy")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
          ncol=3, fontsize=7, frameon=False)
    clean_axes(ax)

    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {filename}")


# SIE
df_lin = pd.read_csv("results/energy_sie_waveform.csv")
df_avg = pd.read_csv("results/energy_sie_waveform_avg.csv")
plot_energy(df_lin, df_avg, "results/energy_linear_vs_avg_sie.pdf")


# Exact subsystem solver
df_lin = pd.read_csv("results/energy_exact_waveform_linear.csv")
df_avg = pd.read_csv("results/energy_exact_waveform_avg.csv")
plot_energy(df_lin, df_avg, "results/energy_exact_linear_vs_avg.pdf")