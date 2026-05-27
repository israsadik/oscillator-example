"""
collection of compute_*_error() functions for the partitioned experiments
"""

import run_partitioned_simulation as rps
from oscillator import analytical_solution


def compute_newmark_error(t_stop: float, N: int, coupling_scheme: str, **kwargs):
    true_sol = analytical_solution(t_stop, N)
    num_sol, iter_counts = rps.partitioned_newmark_beta(t_stop, N, coupling_scheme, **kwargs)
    return true_sol - num_sol, iter_counts


def compute_alpha_error(t_stop: float, N: int, coupling_scheme: str, **kwargs):
    true_sol = analytical_solution(t_stop, N)
    num_sol, iter_counts = rps.partitioned_generalized_alpha(t_stop, N, coupling_scheme, **kwargs)
    return true_sol - num_sol, iter_counts


def compute_erk4_error(t_stop: float, N: int, coupling_scheme: str, **kwargs):
    true_sol = analytical_solution(t_stop, N)
    num_sol, iter_counts = rps.partitioned_erk(t_stop, N, 4, coupling_scheme, **kwargs)
    return true_sol - num_sol, iter_counts


def compute_sie_error(t_stop: float, N: int, coupling_scheme: str, **kwargs):
    true_sol = analytical_solution(t_stop, N)
    num_sol, iter_counts = rps.partitioned_semi_implicit_euler(t_stop, N, coupling_scheme, **kwargs)
    return true_sol - num_sol, iter_counts

def compute_mid_error(t_stop: float, N: int, coupling_scheme: str, **kwargs):
    true_sol = analytical_solution(t_stop, N)
    num_sol, iter_counts = rps.partitioned_implicit_midpoint(t_stop, N, coupling_scheme, **kwargs)
    return true_sol - num_sol, iter_counts

def compute_erk2_error(t_stop: float, N: int, coupling_scheme: str, **kwargs):
    true_sol = analytical_solution(t_stop, N)
    num_sol, iter_counts = rps.partitioned_erk(t_stop, N, order=2, coupling_scheme_str=coupling_scheme, **kwargs)
    return true_sol - num_sol, iter_counts

def compute_trap_error(t_stop: float, N: int, coupling_scheme: str, **kwargs):
    true_sol = analytical_solution(t_stop, N)
    num_sol, iter_counts = rps.partitioned_trapezoidal(t_stop, N, coupling_scheme, **kwargs)
    return true_sol - num_sol, iter_counts

def compute_erk3_error(t_stop: float, N: int, coupling_scheme: str, **kwargs):
    true_sol = analytical_solution(t_stop, N)
    num_sol, iter_counts = rps.partitioned_erk(t_stop, N, order=3, coupling_scheme_str=coupling_scheme, **kwargs)
    return true_sol - num_sol, iter_counts


def compute_exact_error(t_stop: float, N: int, coupling_scheme: str, **kwargs):
    import run_partitioned_simulation as rps
    from oscillator import analytical_solution
    true_sol = analytical_solution(t_stop, N)
    num_sol, iter_counts = rps.partitioned_exact(
        t_stop, N, coupling_scheme, **kwargs
    )

    return true_sol - num_sol, iter_counts
