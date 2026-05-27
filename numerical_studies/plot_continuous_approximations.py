import numpy as np
import matplotlib.pyplot as plt

# Schematic one-step interval
t = np.linspace(0, 1, 400)

def u(t):
    return 1.4 + 0.9 * np.cos(1.25 * np.pi * t - 0.3)

u_n = u(0)
u_np1 = u(1)

eta_lin = (1 - t) * u_n + t * u_np1
eta_avg = 0.5 * (u_n + u_np1) * np.ones_like(t)


def clean_axis(ax):
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$u$")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([r"$t_n$", r"$t_{n+1}$"])
    ax.set_yticks([])
    ax.spines["top"].set_color("none")
    ax.spines["right"].set_color("none")
    ax.grid(False)
    ax.legend(frameon=False)


# Figure 1: linear interpolation
fig, ax = plt.subplots(figsize=(5.0, 3.2))

ax.plot(t, u(t), label=r"$u(t)$")
ax.plot(t, eta_lin, "--", label=r"$\tilde{\eta}_{\mathrm{lin}}$")
ax.scatter([0, 1], [u_n, u_np1], s=25, color="black", zorder=3)

clean_axis(ax)

fig.tight_layout()
fig.savefig("linear_interpolation_schematic.pdf", bbox_inches="tight")
plt.close(fig)


# Figure 2: time-averaged approximation
fig, ax = plt.subplots(figsize=(5.0, 3.2))

ax.plot(t, u(t), label=r"$u(t)$")
ax.plot(t, eta_avg, "--", label=r"$\tilde{\eta}_{\mathrm{avg}}$")
ax.scatter([0, 1], [u_n, u_np1], s=25, color="black", zorder=3)

clean_axis(ax)

fig.tight_layout()
fig.savefig("time_avg_schematic.pdf", bbox_inches="tight")
plt.close(fig)

print("Saved linear_interpolation_schematic.pdf")
print("Saved time_avg_schematic.pdf")