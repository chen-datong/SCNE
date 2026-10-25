"""Draw a compact 2x2 TFIM and Heisenberg virtual-cooling comparison."""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from pypdf import PdfReader, PdfWriter, Transformation

from plot import plot_virtual_cooling_fidelity_scatter
from plot_heisenberg import plot as plot_heisenberg

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from check_paper_data import select_cooling_input_label

FIGURES = ROOT / "figures"
DATA = ROOT / "data" / "virtual_cooling"

# Physical subplot bounds (PDF points) match SE_scaling.pdf.
PAGE_WIDTH = 518.4
UPPER_PAGE_HEIGHT = 263.900689
LOWER_PAGE_HEIGHT = 264.119999
FRAME_BOTTOM = 58.210625
FRAME_HEIGHT = 194.890062
FRAME_SPANS = ((78.45938, 286.802954), (297.567252, 505.910826))


def matched_row_layout(page_height, height_delta):
    return [
        (left / PAGE_WIDTH, FRAME_BOTTOM / page_height,
         (right - left) / PAGE_WIDTH, (FRAME_HEIGHT + height_delta) / page_height)
        for left, right in FRAME_SPANS
    ]


def combine_pages(upper_path, lower_path, output_path, gap=-12.0):
    upper = PdfReader(str(upper_path)).pages[0]
    lower = PdfReader(str(lower_path)).pages[0]
    upper.cropbox = upper.mediabox
    lower.cropbox = lower.mediabox
    width = float(upper.mediabox.width)
    if abs(width - float(lower.mediabox.width)) > 0.1:
        raise ValueError("Both model rows must have the same width")
    lower_height = float(lower.mediabox.height)
    writer = PdfWriter()
    page = writer.add_blank_page(
        width=width, height=float(upper.mediabox.height) + lower_height + gap
    )
    page.merge_transformed_page(
        upper, Transformation().translate(tx=0, ty=lower_height + gap)
    )
    page.merge_page(lower)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as stream:
        writer.write(stream)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-height", type=float, default=3.8)
    parser.add_argument(
        "--output", type=Path, default=FIGURES / "vcooling_models.pdf"
    )
    args = parser.parse_args(argv)
    height_delta = (args.panel_height - 3.8) * 72
    upper_height = UPPER_PAGE_HEIGHT + height_delta
    lower_height = LOWER_PAGE_HEIGHT + height_delta
    config = json.loads((ROOT / "paper_parameters.json").read_text(encoding="utf-8"))["figures"]
    tfim = config["vcooling.pdf"]
    nm_label = select_cooling_input_label(config)
    heisenberg = json.loads((ROOT / "qvc_heisenberg_parameters.json").read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temporary:
        upper = Path(temporary) / "tfim_upper.pdf"
        lower = Path(temporary) / "heisenberg_lower.pdf"
        # Prefer a complete caption-budget series. Historical fallback:
        # NM1000 labels used N_M/d in the original generator:
        # total N_M=1000*2^14=16384000. Keep the original file names.
        # Current generation uses total shots and targets N_M=10^6.
        plot_virtual_cooling_fidelity_scatter(
            n=tfim["n"], Nu=tfim["N_U"], NM=nm_label, J=tfim["J"],
            h_list=tfim["right"]["h"], h=tfim["left"]["h"],
            beta_list=tfim["left"]["beta"], data_dir=DATA, truth_dir=DATA,
            h_fmt="{:.2f}", beta_fmt="{:.1f}", err_mode=tfim["error_band"],
            expected_trials=tfim["trials"], outpath=upper,
            figure_height=upper_height / 72, show_legend=False,
            axes_bounds=matched_row_layout(upper_height, height_delta),
            swap_panels=True,
            panel_label_position=(0.03, 0.9),
        )
        plot_heisenberg(
            heisenberg, DATA / "heisenberg", DATA / "spectra" / "heisenberg", lower,
            panel_labels=("(c)", "(d)"), w_pad=0.05,
            show_legend=False, shared_legend=True,
            swap_panels=True,
            panel_label_position=(0.03, 0.9), figure_height=lower_height / 72,
            axes_bounds=matched_row_layout(lower_height, height_delta),
        )
        combine_pages(upper, lower, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
