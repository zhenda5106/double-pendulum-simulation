"""Small physics checks. Run with: python verify.py (no extra dependencies)."""

import unittest

import numpy as np

from double_pendulum import (
    cartesian_coordinates, derivatives, relative_energy_drift, simulate, total_energy,
)
from sensitivity import compare_initial_conditions


class PhysicsChecks(unittest.TestCase):
    """Check independent physical identities and the numerical experiment."""

    def test_downward_equilibrium(self):
        """Both bobs at rest below the pivot must remain at rest."""
        _, states = simulate([0, 0, 0, 0], duration=1)
        np.testing.assert_array_equal(states, 0)
        self.assertAlmostEqual(total_energy(states)[0], -3 * 9.81)

    def test_cartesian_energy_and_rod_lengths(self):
        """Compare the angular energy formula with Cartesian speeds and heights."""
        state = np.array([0.8, 0.7, -0.4, -1.2])
        m1, m2, L1, L2, g = 1.3, 0.8, 0.7, 1.4, 9.81
        x1, y1, x2, y2 = cartesian_coordinates(state, L1, L2)
        self.assertAlmostEqual(np.hypot(x1, y1), L1)
        self.assertAlmostEqual(np.hypot(x2 - x1, y2 - y1), L2)
        # A centered difference of positions yields an independent velocity.
        step = 1e-6
        shift = np.array([state[1], 0, state[3], 0]) * step
        ahead = np.array(cartesian_coordinates(state + shift, L1, L2))
        behind = np.array(cartesian_coordinates(state - shift, L1, L2))
        vx1, vy1, vx2, vy2 = (ahead - behind) / (2 * step)
        expected = 0.5 * m1 * (vx1**2 + vy1**2) + 0.5 * m2 * (vx2**2 + vy2**2)
        expected += m1 * g * y1 + m2 * g * y2
        self.assertAlmostEqual(total_energy(state, m1, m2, L1, L2, g), expected, places=7)

    def test_equations_conserve_energy_instantaneously(self):
        """Check grad(E) dot state_derivative = 0 at several general states."""
        parameters = (1.3, 0.8, 0.7, 1.4, 9.81)
        for state in np.random.default_rng(7).uniform(-2, 2, (8, 4)):
            gradient = np.zeros(4)
            for index in range(4):
                offset = np.zeros(4)
                offset[index] = 1e-6
                gradient[index] = (total_energy(state + offset, *parameters)
                                   - total_energy(state - offset, *parameters)) / 2e-6
            rate = gradient @ derivatives(0, state, *parameters)
            self.assertLess(abs(rate), 1e-6)

    def test_energy_drift_and_short_time_convergence(self):
        """Tight integration should conserve energy and converge over a short interval."""
        initial = [np.deg2rad(120), 0, np.deg2rad(-10), 0]
        _, standard = simulate(initial, duration=2)
        _, tighter = simulate(initial, duration=2, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(standard, tighter, atol=1e-7, rtol=1e-7)
        energy = total_energy(standard)
        self.assertLess(np.max(np.abs(energy - energy[0])), 1e-7)

    def test_sensitivity_changes_only_one_initial_angle(self):
        """The two runs differ only in theta1, by 0.01 degrees."""
        initial = np.array([np.deg2rad(120), 0, np.deg2rad(-10), 0])
        original = initial.copy()
        offset = np.deg2rad(0.01)
        _, a, b, distance = compare_initial_conditions(initial, offset, duration=0.2)
        np.testing.assert_array_equal(initial, original)
        np.testing.assert_allclose(b[:, 0] - a[:, 0], [offset, 0, 0, 0], atol=1e-15)
        self.assertAlmostEqual(distance[0], 2 * np.sin(offset / 2), places=12)

    def test_near_zero_energy_diagnostic(self):
        """Zero initial energy must not produce infinity in relative drift."""
        self.assertIsNone(relative_energy_drift(np.array([0.0, 1e-10]), 29.43))
        np.testing.assert_allclose(relative_energy_drift(np.array([-2.0, -1.0]), 29.43), [0, 0.5])

    def test_invalid_inputs(self):
        """Reject parameters that are not valid for the point-mass model."""
        for options in ({"m1": 0}, {"L2": -1}, {"dt": 0}, {"duration": np.nan}):
            with self.assertRaises(ValueError):
                simulate([0, 0, 0, 0], **options)


if __name__ == "__main__":
    unittest.main(verbosity=2)
