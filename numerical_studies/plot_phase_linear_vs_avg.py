import pandas as pd
import matplotlib.pyplot as plt
from oscillator import analytical_solution


def clean_axes(ax):
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.grid(False)


def plot_phase(df_lin, df_avg, filename):
    t_stop = df_lin["t"].iloc[-1]
    N = len(df_lin["t"]) - 1
    exact_sol = analytical_solution(t_stop, N)

    fig, ax = plt.subplots(figsize=(5.0, 3.2))

    ax.plot(exact_sol[0], exact_sol[2], "k--", label="analytical")
    ax.plot(df_lin["u1"], df_lin["v1"], label="linear interpolation")
    ax.plot(df_avg["u1"], df_avg["v1"], label="time averaging")

    ax.set_xlabel(r"$u_1$")
    ax.set_ylabel(r"$\dot u_1$")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-12, 12)
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
plot_phase(df_lin, df_avg, "results/phase_linear_vs_avg_sie.pdf")


# Exact subsystem solver
df_lin = pd.read_csv("results/energy_exact_waveform_linear.csv")
df_avg = pd.read_csv("results/energy_exact_waveform_avg.csv")
plot_phase(df_lin, df_avg, "results/phase_exact_linear_vs_avg.pdf")