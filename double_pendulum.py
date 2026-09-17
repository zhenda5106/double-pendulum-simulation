"""Planar point-mass double pendulum; angles are measured from downward vertical.

All inputs use SI units: kg, m, s, radians, and radians/second.
See README.md for the Lagrangian and coupled acceleration equations.
"""

import numpy as np
from scipy.integrate import solve_ivp


def derivatives(time, state, m1, m2, L1, L2, g):
    """Return [omega1, alpha1, omega2, alpha2] for one state.

    The equations have no explicit time dependence. ``time`` is nevertheless
    required by solve_ivp. Solve the two coupled equations algebraically,
    rather than assuming that either rod behaves like an independent pendulum.
    """
    theta1, omega1, theta2, omega2 = state
    delta = theta1 - theta2

    # Euler-Lagrange equations: A*alpha1 + B*alpha2 = F1,
    #                           B*alpha1 + C*alpha2 = F2.
    A = (m1 + m2) * L1**2
    B = m2 * L1 * L2 * np.cos(delta)
    C = m2 * L2**2
    F1 = -m2 * L1 * L2 * omega2**2 * np.sin(delta)
    F1 -= (m1 + m2) * g * L1 * np.sin(theta1)
    F2 = m2 * L1 * L2 * omega1**2 * np.sin(delta)
    F2 -= m2 * g * L2 * np.sin(theta2)

    # A*C-B**2, written in a form avoiding subtractive cancellation.
    determinant = m2 * L1**2 * L2**2 * (m1 + m2 * np.sin(delta)**2)
    alpha1 = (C * F1 - B * F2) / determinant
    alpha2 = (A * F2 - B * F1) / determinant
    # Angle derivatives are angular velocities; velocity derivatives are
    # angular accelerations. The ordering matches the input state exactly.
    return np.array([omega1, alpha1, omega2, alpha2])


def simulate(initial_state, duration=30.0, dt=0.01, m1=1.0, m2=1.0,
             L1=1.0, L2=1.0, g=9.81, rtol=1e-10, atol=1e-12):
    """Integrate the motion and return time (N,) and states (4, N).

    dt sets the maximum spacing of saved samples, not the adaptive solver's
    internal step size. The endpoint is included using a uniform sample grid.
    """
    positive_values = (duration, dt, m1, m2, L1, L2, g, rtol, atol)
    if not all(np.isfinite(value) and value > 0 for value in positive_values):
        raise ValueError("Duration, dt, masses, lengths, g and tolerances must be positive and finite.")
    initial_state = np.asarray(initial_state, dtype=float)
    if initial_state.shape != (4,) or not np.all(np.isfinite(initial_state)):
        raise ValueError("Initial state must contain four finite numbers.")

    sample_count = max(2, int(np.ceil(duration / dt)) + 1)
    times = np.linspace(0.0, duration, sample_count)
    solution = solve_ivp(
        derivatives, (0.0, duration), initial_state, t_eval=times,
        args=(m1, m2, L1, L2, g), method="DOP853", rtol=rtol, atol=atol,
    )
    if not solution.success:
        raise RuntimeError(f"Integration failed: {solution.message}")
    return solution.t, solution.y


def cartesian_coordinates(states, L1=1.0, L2=1.0):
    """Return x1, y1, x2, y2 with the pivot at the origin and y upward."""
    theta1, _, theta2, _ = states
    x1 = L1 * np.sin(theta1)
    y1 = -L1 * np.cos(theta1)
    x2 = x1 + L2 * np.sin(theta2)
    y2 = y1 - L2 * np.cos(theta2)
    return x1, y1, x2, y2


def total_energy(states, m1=1.0, m2=1.0, L1=1.0, L2=1.0, g=9.81):
    """Return kinetic plus potential energy (J); potential is zero at pivot height."""
    theta1, omega1, theta2, omega2 = states
    # Bob 2 moves because both rods rotate: its squared speed has a cross term.
    kinetic = 0.5 * (m1 + m2) * L1**2 * omega1**2
    kinetic += 0.5 * m2 * L2**2 * omega2**2
    kinetic += m2 * L1 * L2 * omega1 * omega2 * np.cos(theta1 - theta2)
    potential = -(m1 + m2) * g * L1 * np.cos(theta1)
    potential -= m2 * g * L2 * np.cos(theta2)
    return kinetic + potential


def relative_energy_drift(energy, energy_scale):
    """Return (E-E0)/abs(E0), or None if E0 is too close to zero.

    Near-zero E0 makes this relative diagnostic misleading or undefined.
    The caller then plots absolute drift in joules instead.
    """
    energy = np.asarray(energy)
    if abs(energy[0]) <= 1e-12 * energy_scale:
        return None
    return (energy - energy[0]) / abs(energy[0])
