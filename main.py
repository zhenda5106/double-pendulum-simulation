"""Run the simulation, save five figures, and optionally show/export animation."""

import argparse
import json
from pathlib import Path

import numpy as np

from double_pendulum import relative_energy_drift, total_energy
from sensitivity import compare_initial_conditions


def parse_arguments():
    """Read initial conditions and simulation settings from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theta1", type=float, default=120.0, help="Initial first angle (degrees)")
    parser.add_argument("--theta2", type=float, default=-10.0, help="Initial second angle (degrees)")
    parser.add_argument("--omega1", type=float, default=0.0, help="Initial first angular velocity (rad/s)")
    parser.add_argument("--omega2", type=float, default=0.0, help="Initial second angular velocity (rad/s)")
    for name, default, unit in [("m1", 1.0, "kg"), ("m2", 1.0, "kg"),
                                ("L1", 1.0, "m"), ("L2", 1.0, "m"), ("g", 9.81, "m/s^2")]:
        parser.add_argument(f"--{name}", type=float, default=default, help=f"{name} ({unit})")
    parser.add_argument("--duration", type=float, default=30.0, help="Duration (s)")
    parser.add_argument("--dt", type=float, default=0.01, help="Maximum output sample spacing (s)")
    parser.add_argument("--perturbation", type=float, default=0.01, help="Offset added to theta1 in run B (degrees)")
    parser.add_argument("--rtol", type=float, default=1e-10, help="Solver relative tolerance")
    parser.add_argument("--atol", type=float, default=1e-12, help="Solver absolute tolerance")
    parser.add_argument("--animate", action="store_true", help="Display the animation in a GUI window")
    parser.add_argument("--save-gif", action="store_true", help="Save results/double_pendulum.gif")
    parser.add_argument("--fps", type=int, default=30, help="Animation frames per second")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent,
                        help="Parent folder for figures/ and results/ (default: project folder)")
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be positive")
    return args


def main():
    """Compute both trajectories, diagnostics, figures, and reproducible output data."""
    args = parse_arguments()
    # Avoid opening windows by default; --animate uses the system GUI backend.
    import matplotlib
    if not args.animate:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from visualization import animate_pendulum, plot_energy, plot_motion, plot_sensitivity, save_gif

    parameters = {name: getattr(args, name) for name in ("m1", "m2", "L1", "L2", "g")}
    settings = dict(parameters, duration=args.duration, dt=args.dt, rtol=args.rtol, atol=args.atol)
    initial = np.array([np.deg2rad(args.theta1), args.omega1,
                        np.deg2rad(args.theta2), args.omega2])
    print(f"Double pendulum: {args.duration:g} s; theta1 = {args.theta1:g} degrees.")
    try:
        times, states_a, states_b, separation = compare_initial_conditions(
            initial, np.deg2rad(args.perturbation), **settings,
        )
    except (ValueError, RuntimeError) as error:
        raise SystemExit(f"Simulation error: {error}") from error

    figures = args.output_dir / "figures"
    results = args.output_dir / "results"
    figures.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    energy_a = total_energy(states_a, **parameters)
    energy_b = total_energy(states_b, **parameters)
    energy_scale = (args.m1 + args.m2) * args.g * args.L1 + args.m2 * args.g * args.L2
    drift_a = relative_energy_drift(energy_a, energy_scale)
    drift_b = relative_energy_drift(energy_b, energy_scale)
    plot_motion(times, states_a, args.L1, args.L2, figures)
    plot_energy(times, energy_a, drift_a, figures)
    plot_sensitivity(times, separation, args.perturbation, figures)

    summary = {
        "parameters_SI": parameters,
        "initial_state_A_SI": states_a[:, 0].tolist(),
        "initial_state_B_SI": states_b[:, 0].tolist(),
        "perturbation_degrees": args.perturbation,
        "duration_s": args.duration,
        "requested_sample_spacing_s": args.dt,
        "actual_sample_spacing_s": float(times[1] - times[0]),
        "samples": len(times),
        "solver": "DOP853", "rtol": args.rtol, "atol": args.atol,
        "initial_energy_A_J": float(energy_a[0]),
        "max_absolute_energy_drift_A_J": float(np.max(np.abs(energy_a - energy_a[0]))),
        "max_relative_energy_drift_A": None if drift_a is None else float(np.max(np.abs(drift_a))),
        "max_relative_energy_drift_B": None if drift_b is None else float(np.max(np.abs(drift_b))),
        "initial_separation_m": float(separation[0]),
        "final_separation_m": float(separation[-1]),
        "maximum_separation_m": float(np.max(separation)),
        "versions": {"numpy": np.__version__, "matplotlib": matplotlib.__version__},
    }
    import scipy
    summary["versions"]["scipy"] = scipy.__version__
    (results / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(results / "simulation.npz", time=times, state_a=states_a,
                        state_b=states_b, energy_a=energy_a, energy_b=energy_b,
                        separation=separation)
    print(f"Saved five plots to {figures.resolve()}")
    print(f"Initial energy: {energy_a[0]:.8g} J")
    if drift_a is None:
        print("Initial energy is near zero; reporting absolute drift instead of relative drift.")
        print(f"Maximum absolute energy drift: {np.max(np.abs(energy_a - energy_a[0])):.3e} J")
    else:
        print(f"Maximum relative energy drift (A): {np.max(np.abs(drift_a)):.3e}")
    print(f"Second-bob separation: {separation[0]:.3e} m initially; {separation[-1]:.3e} m finally.")
    print("Sensitivity is an illustration, not a Lyapunov-exponent measurement.")

    if args.animate or args.save_gif:
        figure, animation = animate_pendulum(times, states_a, args.L1, args.L2, fps=args.fps)
        if args.save_gif:
            path = results / "double_pendulum.gif"
            if save_gif(animation, path, args.fps):
                print(f"Saved animation to {path.resolve()}")
        if args.animate:
            plt.show()
        plt.close(figure)


if __name__ == "__main__":
    main()
