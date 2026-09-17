"""Nearby initial conditions: an illustration, not a Lyapunov-exponent estimate."""

import numpy as np

from double_pendulum import cartesian_coordinates, simulate


def compare_initial_conditions(initial_state, perturbation=np.deg2rad(0.01), **settings):
    """Run two systems differing only in theta1 and measure second-bob separation.

    Cartesian distance is intuitive and avoids artificial jumps when angles
    cross +/-pi. It is a projected, bounded distance, not a full phase-space
    norm; it can shrink again even after the trajectories have diverged.
    """
    if not np.isfinite(perturbation) or perturbation == 0:
        raise ValueError("The angle perturbation must be finite and nonzero.")
    state_a = np.asarray(initial_state, dtype=float).copy()
    state_b = state_a.copy()
    if state_b.shape != (4,):
        raise ValueError("Initial state must contain four numbers.")
    state_b[0] += perturbation
    if state_a[0] == state_b[0]:
        raise ValueError("Perturbation is too small to represent at this initial angle.")
    times, states_a = simulate(state_a, **settings)
    _, states_b = simulate(state_b, **settings)
    L1, L2 = settings.get("L1", 1.0), settings.get("L2", 1.0)
    _, _, x_a, y_a = cartesian_coordinates(states_a, L1, L2)
    _, _, x_b, y_b = cartesian_coordinates(states_b, L1, L2)
    separation = np.hypot(x_b - x_a, y_b - y_a)
    return times, states_a, states_b, separation
