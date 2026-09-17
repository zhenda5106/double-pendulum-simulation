"""Static figures and an optional animation, separate from the physics code."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter, writers
from matplotlib.collections import LineCollection

from double_pendulum import cartesian_coordinates


def save_figure(fig, directory, filename):
    """Save one clearly labeled figure and release its memory."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(directory / filename, dpi=160)
    plt.close(fig)


def plot_motion(times, states, L1, L2, directory):
    """Save angles, angular velocities, and the trajectory of the second bob."""
    for rows, ylabel, title, filename, labels in [
        ((0, 2), "Angle (rad)", "Angular motion", "angles.png", (r"$\theta_1$", r"$\theta_2$")),
        ((1, 3), "Angular velocity (rad/s)", "Angular velocities", "angular_velocities.png", (r"$\omega_1$", r"$\omega_2$")),
    ]:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for row, label in zip(rows, labels):
            ax.plot(times, states[row], label=label, linewidth=1.0)
        ax.set(xlabel="Time (s)", ylabel=ylabel, title=title)
        ax.legend()
        ax.grid(alpha=0.2)
        save_figure(fig, directory, filename)

    _, _, x2, y2 = cartesian_coordinates(states, L1, L2)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(x2, y2, linewidth=0.65, color="#002a5c", alpha=0.75, label="Second bob")
    ax.plot(0, 0, "k+", markersize=9, label="Pivot")
    ax.plot(x2[0], y2[0], "o", color="#b45122", markersize=5, label="Start")
    ax.set(xlabel="x (m)", ylabel="y (m)", title="Trajectory of the second mass")
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    ax.grid(alpha=0.2)
    save_figure(fig, directory, "trajectory.png")


def plot_energy(times, energy, drift, directory):
    """Save total energy and relative drift (absolute drift when E0 is near zero)."""
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(times, energy, color="#002a5c")
    axes[0].axhline(energy[0], color="#b45122", linestyle="--", label="Initial energy")
    axes[0].set(ylabel="Energy (J)", title="Mechanical energy and integration error")
    axes[0].legend()
    axes[0].ticklabel_format(axis="y", useOffset=False)
    values = energy - energy[0] if drift is None else drift
    axes[1].plot(times, values, color="#002a5c", linewidth=1)
    axes[1].set(xlabel="Time (s)", ylabel="Absolute drift (J)" if drift is None else "Relative energy drift")
    for ax in axes:
        ax.grid(alpha=0.2)
    save_figure(fig, directory, "energy.png")


def plot_sensitivity(times, separation, perturbation_degrees, directory):
    """Plot bob-2 distance on a log scale without interpreting its slope as an exponent."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    # Mask exact zeros rather than inventing a nonzero measured separation.
    positive_separation = np.ma.masked_less_equal(separation, 0)
    ax.semilogy(times, positive_separation, color="#002a5c", linewidth=1)
    ax.set(xlabel="Time (s)", ylabel="Second-bob separation (m)",
           title=f"Initial-angle sensitivity: theta1 offset = {perturbation_degrees:g} degrees")
    ax.grid(which="both", alpha=0.2)
    save_figure(fig, directory, "sensitivity.png")


def animate_pendulum(times, states, L1, L2, fps=30, trail_seconds=1.5):
    """Return (figure, animation) showing rods, masses, time, and a fading trail.

    Resample for a constant playback frame rate independently of saved dt.
    Linear interpolation is for display only; use sufficiently small dt for
    accurate-looking playback. Numerical analysis always uses solver output.
    """
    frame_times = np.arange(0, times[-1] + 1e-12, 1.0 / fps)
    if len(frame_times) < 2:
        frame_times = np.array([0.0, times[-1]])
    frame_states = np.array([np.interp(frame_times, times, row) for row in states])
    x1, y1, x2, y2 = cartesian_coordinates(frame_states, L1, L2)
    fig, ax = plt.subplots(figsize=(6, 6))
    radius = 1.1 * (L1 + L2)
    ax.set(xlim=(-radius, radius), ylim=(-radius, radius), xlabel="x (m)",
           ylabel="y (m)", title="Double pendulum")
    ax.set_aspect("equal", adjustable="box")
    ax.plot(0, 0, "k+", markersize=12, zorder=4)
    rods, = ax.plot([], [], "-o", color="#002a5c", linewidth=2, markersize=9, zorder=3)
    trail = LineCollection([], linewidths=1.5, zorder=2)
    ax.add_collection(trail)
    clock = ax.text(0.03, 0.95, "", transform=ax.transAxes)
    trail_frames = max(2, int(trail_seconds * fps))

    def update(frame):
        """Update artists for one animation frame."""
        rods.set_data([0, x1[frame], x2[frame]], [0, y1[frame], y2[frame]])
        start = max(0, frame - trail_frames)
        points = np.column_stack((x2[start:frame + 1], y2[start:frame + 1]))
        segments = np.stack((points[:-1], points[1:]), axis=1)
        trail.set_segments(segments)
        colors = np.zeros((len(segments), 4))
        colors[:, :3] = [0.13, 0.42, 0.65]
        colors[:, 3] = np.linspace(0.03, 0.8, len(segments))
        trail.set_color(colors)
        clock.set_text(f"t = {frame_times[frame]:.2f} s")
        return rods, trail, clock

    animation = FuncAnimation(fig, update, frames=len(frame_times),
                              interval=1000 / fps, blit=True, cache_frame_data=False)
    fig.tight_layout()
    return fig, animation


def save_gif(animation, path, fps=30):
    """Try GIF export; a missing writer or export error does not lose static plots."""
    if not writers.is_available("pillow"):
        print("GIF export unavailable: install Pillow. Static plots are saved.")
        return False
    try:
        animation.save(str(path), writer=PillowWriter(fps=fps), dpi=90)
    except (ImportError, OSError, RuntimeError) as error:
        print(f"GIF export failed: {error}. Static plots are saved.")
        return False
    return True
