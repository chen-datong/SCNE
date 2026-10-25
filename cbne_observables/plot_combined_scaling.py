"""Stack the CBNE and rho^3 scaling panels in one vector PDF."""

import argparse
import json
import sys
import tempfile
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from pypdf import PdfReader, PdfWriter, Transformation

from plot_rho3 import collect_errors, plot_rho3
from generate_rho3_data import experiment_parameters


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from check_paper_data import select_cbne_right_counts

FIGURES = ROOT / "figures"
DATA = ROOT / "data" / "cbne_observables"


def plot_upper_from_retained_data(output_path, figure_height):
    """Redraw the original upper panels at the requested canvas height."""
    source = ROOT / "cbne_moment" / "plot.py"
    spec = spec_from_file_location("cbne_moment_plot_for_combined", source)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.plot_Haar_N(
        right_nm_by_n=select_cbne_right_counts(json.loads((ROOT / "paper_parameters.json").read_text(encoding="utf-8"))["figures"]),
        outpath=output_path,
        figure_height=figure_height,
    )


def combine_pages(upper_path, lower_path, output_path):
    upper = PdfReader(str(upper_path)).pages[0]
    lower = PdfReader(str(lower_path)).pages[0]
    upper_width, upper_height = float(upper.cropbox.width), float(upper.cropbox.height)
    lower_width, lower_height = float(lower.cropbox.width), float(lower.cropbox.height)
    # Keep the manuscript figure's original horizontal page size.
    width = max(float(upper.mediabox.width), float(lower.mediabox.width))
    gap = -12.0

    writer = PdfWriter()
    page = writer.add_blank_page(width=width, height=upper_height + lower_height + gap)
    # The lower plot uses explicit bounds matching the upper row.
    page.merge_transformed_page(
        upper,
        Transformation().translate(
            tx=(width - upper_width) / 2 - float(upper.cropbox.left),
            ty=lower_height + gap - float(upper.cropbox.bottom),
        ),
    )
    page.merge_transformed_page(
        lower,
        Transformation().translate(
            tx=-float(lower.cropbox.left), ty=-float(lower.cropbox.bottom)
        ),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as output:
        writer.write(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upper", type=Path, default=None, help="Use an existing upper PDF instead of redrawing it")
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Write the freshly computed error summary here",
    )
    parser.add_argument(
        "--output", type=Path, default=FIGURES / "SE_scaling.pdf"
    )
    parser.add_argument("--state", choices=("depolarized", "gibbs"), default="depolarized")
    parser.add_argument("--source", choices=("auto", "paper", "rho3"), default="auto",
                        help="Recompute errors from current raw CSVs on every run")
    parser.add_argument("--panel-height", type=float, default=3.8)
    parser.add_argument("--lower-height", type=float, default=None)
    parser.add_argument("--anc-left", type=int, nargs=4, default=[0, 4, 8, 12])
    parser.add_argument("--anc-right-max", type=int, default=12)
    parser.add_argument("--nm-fit-range", type=int, nargs=2, default=[5000, 50000])
    args = parser.parse_args(argv)

    if args.upper is not None and not args.upper.is_file():
        parser.error(f"Upper-row PDF not found: {args.upper}")
    params = experiment_parameters(args.state)
    anc_right = list(range(0, args.anc_right_max + 1, 2))
    right_ticks = anc_right.copy()
    requested_ancilla = sorted(
        set(params["right"]["n_a"]) | set(args.anc_left) | set(anc_right)
    )
    params["right"]["n_a"] = requested_ancilla
    errors, source, exact = collect_errors(params, DATA, args.source)
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        errors.to_csv(args.summary, index=False)
    print(f"Data source: {source}; exact tr(rho^3): {exact:.12g}")
    if args.summary is not None:
        print(f"Summary: {args.summary}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temporary:
        upper = args.upper
        upper_x_limits = (10**1.75, 10**7.25)
        if upper is None:
            upper = Path(temporary) / "cbne_upper.pdf"
            upper_x_limits = plot_upper_from_retained_data(upper, args.panel_height)
        lower = Path(temporary) / "rho3_lower.pdf"
        height_delta = ((args.lower_height or args.panel_height) - 3.8) * 72
        lower_layout = {
            "page_width": 518.4,
            "page_height": 257.4206875 + height_delta,
            "bottom": 58.210625,
            "height": 194.890062 + height_delta,
            "spans": ((78.45938, 286.802954), (297.567252, 505.910826)),
        }
        plot_rho3(
            errors,
            params,
            lower,
            anc_left=args.anc_left,
            anc_right=anc_right,
            anc_right_ticks=right_ticks,
            panel_labels=("(c)", "(d)"),
            show_nm_fit=True,
            nm_fit_anc=max(args.anc_left),
            nm_fit_range=tuple(args.nm_fit_range),
            figure_height=args.lower_height or args.panel_height,
            y_limits=(2e-4, 1.5),
            nm_x_limits=upper_x_limits,
            frame_layout=lower_layout,
        )
        combine_pages(upper, lower, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
