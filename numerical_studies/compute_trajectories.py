"""
compute_*_trajectory() functions for use in the energy_*.py scripts
"""

import run_partitioned_simulation as rps


def compute_newmark_trajectory(t_stop: float, N: int, coupling_scheme: str, **kwargs):
    return rps.partitioned_newmark_beta(t_stop, N, coupling_scheme, **kwargs)


def compute_alpha_trajectory(t_stop: float, N: int, coupling_scheme: str, **kwargs):
    return rps.partitioned_generalized_alpha(t_stop, N, coupling_scheme, **kwargs)


def compute_erk2_trajectory(t_stop: float, N: int, coupling_scheme: str, **kwargs):
    return rps.partitioned_erk(t_stop, N, 2, coupling_scheme, **kwargs)


def compute_erk4_trajectory(t_stop: float, N: int, coupling_scheme: str, **kwargs):
    return rps.partitioned_erk(t_stop, N, 4, coupling_scheme, **kwargs)


def compute_sie_trajectory(t_stop: float, N: int, coupling_scheme: str, **kwargs):
    return rps.partitioned_semi_implicit_euler(t_stop, N, coupling_scheme, **kwargs)


def compute_mid_trajectory(t_stop: float, N: int, coupling_scheme: str, **kwargs):
    return rps.partitioned_implicit_midpoint(t_stop, N, coupling_scheme, **kwargs)


def compute_trap_trajectory(t_stop: float, N: int, coupling_scheme: str, **kwargs):
    return rps.partitioned_trapezoidal(t_stop, N, coupling_scheme, **kwargs)