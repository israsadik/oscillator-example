import numpy as np
import matplotlib.pyplot as plt
from oscillator import analytical_solution


def clean_axes(ax):
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.grid(False)


# Figure 1: displacement plot
t = np.linspace(0, 1, 1200)

u1 = 0.5 * (np.cos(2*np.pi*t) + np.cos(6*np.pi*t))
u2 = 0.5 * (np.cos(2*np.pi*t) - np.cos(6*np.pi*t))

fig, ax = plt.subplots(figsize=(5.0, 3.2))

ax.plot(t, u1, label=r"$u_1$")
ax.plot(t, u2, "--", label=r"$u_2$")
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$u(t)$")
ax.set_xlim(0, 1)
ax.set_ylim(-1.1, 1.1)
ax.legend(frameon=False)
clean_axes(ax)

fig.tight_layout()
fig.savefig("results/oscillator_solution.pdf", bbox_inches="tight")
plt.close(fig)


# Figure 2: analytical phase portrait
t_stop = 5
N = 1000
exact_sol = analytical_solution(t_stop, N)

fig, ax = plt.subplots(figsize=(5.0, 3.2))

ax.plot(exact_sol[0], exact_sol[2], "k--")
ax.set_xlabel(r"$u_1$")
ax.set_ylabel(r"$\dot u_1$")
ax.set_xlim(-2, 2)
ax.set_ylim(-12, 12)
clean_axes(ax)

fig.tight_layout()
fig.savefig("results/oscillator_phase.pdf", bbox_inches="tight")
plt.close(fig)

print("Saved results/oscillator_solution.pdf")
print("Saved results/oscillator_phase.pdf")