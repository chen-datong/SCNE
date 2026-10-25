"""Draw the eight current paper figures from existing numerical data."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Stable command name, output filename, implementation directory.
FIGURES = (
    ("cbne-scaling", "SE_scaling.pdf", "cbne_observables"),
    ("cbne-sample-lower", "sample_lower.pdf", "cbne_moment"),
    ("virtual-cooling", "vcooling_models.pdf", "virtual_cooling"),
    ("cbne-observables", "SE_all_observables.pdf", "cbne_observables"),
    ("ptme-scaling", "SE_vs_NM_and_nA.pdf", "ptme"),
    ("gibbs-detection", "Gibbs_ent_detect.pdf", "ptme"),
    ("depol-detection", "Depol_ent_detect.pdf", "ptme"),
    ("tfim-detection", "TFIM_detection_prob.svg", "ptme"),
)
BY_NAME = {row[0]: row for row in FIGURES}


def draw_figure(name, output):
    """Called in a fresh process so experiment-local imports stay isolated."""
    import matplotlib
    matplotlib.use("Agg")
    sys.path.insert(0, str(ROOT / BY_NAME[name][2]))
    params = json.loads((ROOT / "paper_parameters.json").read_text(encoding="utf-8"))["figures"]
    output.parent.mkdir(parents=True, exist_ok=True)

    if name == "cbne-scaling":
        from plot_combined_scaling import main as draw
        draw(["--output", str(output)])
    elif name == "cbne-sample-lower":
        from plot import plot_sample_complexity_lower
        p = params["sample_lower.pdf"]
        plot_sample_complexity_lower(
            nu=p["N_U"], target_error=p["target_error"],
            repetitions=p["trials"], outpath=output,
        )
    elif name == "virtual-cooling":
        from plot_combined_cooling import main as draw
        draw(["--output", str(output)])
    elif name == "cbne-observables":
        from plot import plot_gibbs_observables_with_rho3
        from generate_rho3_data import experiment_parameters
        plot_gibbs_observables_with_rho3(
            params["SE_all_observables.pdf"], experiment_parameters("gibbs"),
            data_dir=ROOT / "data/cbne_observables", outpath=output,
        )
    elif name == "ptme-scaling":
        from Plot import plot_mae_t4_nA_convergence
        p = params["SE_vs_NM_and_nA.pdf"]
        plot_mae_t4_nA_convergence(
            data_dir=ROOT / "data/ptme", nB=p["n_B"], right_nm=p["right"]["N_M"],
            left_nA_to_plot=p["left"]["n_A"], left_max_nm=p["left"]["N_M_max"],
            outpath=output,
        )
    elif name == "gibbs-detection":
        from Plot import gibbs_ent_detect
        p = params["Gibbs_ent_detect.pdf"]
        gibbs_ent_detect(
            nA=p["n_A"], nB=p["n_B"], Nu=p["N_U"], NM=p["N_M"],
            csv_path=ROOT / "data/ptme/gibbs_dbasis_paper.csv",
            expected_trials=p["trials"], outpath=output,
        )
    elif name == "depol-detection":
        from heisenberg_plot import heisenberg_depol_paper
        p = params["Depol_ent_detect.pdf"]
        heisenberg_depol_paper(
            nA=p["n_A"], nB=p["n_B"], Nu=p["N_U"], NM=p["N_M"],
            csv_path=ROOT / "data/ptme/heisenberg_dbasis_prob_paper.csv",
            expected_trials=p["trials"], spread=p["error_band"], outpath=output,
        )
    elif name == "tfim-detection":
        from Plot import detecting_ent_tfim
        detecting_ent_tfim(csv_filename=ROOT / "data/ptme/processed_results2.csv", outpath=output)


def render_figure(name, output_dir):
    command = [sys.executable, str(Path(__file__).resolve()), "--render", name,
               "--output-dir", str(output_dir)]
    return subprocess.run(command, cwd=ROOT, check=False).returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--list", action="store_true", help="show current paper figures")
    selection.add_argument("--all", action="store_true", help="try all eight current paper outputs")
    selection.add_argument("--figure", choices=BY_NAME, action="append", metavar="NAME",
                           help="select a figure; repeat to select several")
    selection.add_argument("--render", choices=BY_NAME, help=argparse.SUPPRESS)
    parser.add_argument("--check-only", action="store_true", help="audit inputs without drawing")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures",
                        help="override the shared figure destination")
    args = parser.parse_args(argv)
    output_dir = args.output_dir.resolve()
    if args.render:
        draw_figure(args.render, output_dir / BY_NAME[args.render][1])
        return 0
    if args.list or (not args.all and not args.figure):
        for name, filename, _ in FIGURES:
            print(f"{name:20} {filename:26} figures/")
        return 0

    from check_paper_data import audit_figure
    selected = FIGURES if args.all else [BY_NAME[name] for name in dict.fromkeys(args.figure)]
    failures = []
    for name, filename, _ in selected:
        print(f"\n=== {name}: {filename} ===", flush=True)
        issues, notes = audit_figure(filename)
        for note in notes:
            print(f"NOTE {filename}: {note}", flush=True)
        if issues:
            failures.append(name)
            print(f"BLOCKED {filename}")
            for issue in issues[:8]:
                print(f"  {issue}")
            if len(issues) > 8:
                print(f"  ... and {len(issues) - 8} more issues")
        elif args.check_only:
            print(f"READY {filename}")
        elif render_figure(name, output_dir):
            failures.append(name)
            print(f"FAILED {filename}")
        else:
            print(f"SAVED {output_dir / filename}")
    if failures:
        print("\nBlocked or failed: " + ", ".join(failures))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
