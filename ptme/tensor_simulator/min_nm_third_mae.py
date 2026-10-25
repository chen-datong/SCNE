"""Find the smallest NM with grouped third-moment MAE below the threshold.

Average M2 and M3 in blocks of NU, then compute thirdM.
Drop incomplete blocks and scan Sample counts in ascending order.

Run: python ptme/tensor_simulator/min_nm_third_mae.py --nu 4"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parents[2] / "data" / "ptme" / "tensor_simulator" / "exp"

import numpy as np
from scipy.stats import linregress
from process_exact_results import exact_results


def _parse_sample_from_filename(filepath: str) -> int:
    filename = os.path.basename(filepath)
    m = re.search(r"_Sample(\d+)\.jsonl$", filename)
    if not m:
        raise ValueError(f"无法从文件名解析 Sample: {filename}")
    return int(m.group(1))


def _iter_jsonl(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _third_from_m2_m3(m2: float, m3: float) -> float:
    purity = m2 - 1.0
    return (m3 - 1.0 - 3.0 * purity) / 2.0


def _load_m2_m3(filepath: str) -> tuple[list[float], list[float]]:
    m2_list: list[float] = []
    m3_list: list[float] = []
    for rec in _iter_jsonl(filepath):
        m2_list.append(float(rec.get("M2", 0.0)))
        m3_list.append(float(rec.get("M3", 0.0)))
    return m2_list, m3_list


def grouped_third_mae(
    m2: list[float], m3: list[float], target_t3: float, nu: int
) -> tuple[float, int]:
    """Return grouped MAE and block count, or (nan, 0) if no full block exists."""
    total = len(m2)
    num_groups = total // nu
    if num_groups == 0:
        return float("nan"), 0

    errs: list[float] = []
    for i in range(num_groups):
        sl = slice(i * nu, (i + 1) * nu)
        m2_g = m2[sl]
        m3_g = m3[sl]
        m2_bar = sum(m2_g) / len(m2_g)
        m3_bar = sum(m3_g) / len(m3_g)
        third_g = _third_from_m2_m3(m2_bar, m3_bar)
        errs.append(abs(third_g - target_t3))
    return sum(errs) / len(errs), num_groups


def _collect_sample_files(
    exp_dir: str,
    n: int,
    depth: int,
    J: float,
    h: float,
    dprob: float,
) -> list[tuple[int, str]]:
    pattern = os.path.join(
        exp_dir,
        f"n{n}_depth{depth}_J{J:.2f}_h{h:.2f}_p{dprob:.2f}_Sample*.jsonl",
    )
    files = glob.glob(pattern)
    pairs: list[tuple[int, str]] = []
    for fp in files:
        sample = _parse_sample_from_filename(fp)
        pairs.append((sample, fp))
    pairs.sort(key=lambda x: x[0])
    return pairs


def min_nm_third_mae_for_n(
    n: int,
    nu: int,
    dprob: float,
    thresh: float = 0.1,
    depth: int = 60,
    J: float = 1.0,
    h: float = 1.0,
    exp_dir: str = EXP_DIR,
) -> dict[str, int | float | str | None]:
    """Find the smallest NM with grouped third-moment MAE below thresh."""
    _, target_t3, _, _ = exact_results(n=n, depth=depth, J=J, h=h, dprob=dprob)
    sample_files = _collect_sample_files(exp_dir, n, depth, J, h, dprob)
    if not sample_files:
        return {"NM": None, "error": "missing_files"}

    for sample, filepath in sample_files:
        m2, m3 = _load_m2_m3(filepath)
        mae_t3, num_groups = grouped_third_mae(m2, m3, target_t3, nu)
        if num_groups > 0 and mae_t3 < thresh:
            return {
                "NM": sample,
                "MAE_thirdM": mae_t3,
                "num_groups": num_groups,
                "file": os.path.basename(filepath),
            }

    return {"NM": None, "error": "no_solution_under_thresh"}


def min_nm_third_mae_dict(
    n_min: int,
    n_max: int,
    nu: int,
    dprob: float,
    thresh: float = 0.1,
    depth: int = 60,
    J: float = 1.0,
    h: float = 1.0,
    exp_dir: str = EXP_DIR,
) -> dict[int, dict[str, int | float | str | None]]:
    if n_min > n_max:
        raise ValueError("n_min 不能大于 n_max")
    result: dict[int, dict[str, int | float | str | None]] = {}
    for n in range(n_min, n_max + 1):
        result[n] = min_nm_third_mae_for_n(
            n=n,
            nu=nu,
            dprob=dprob,
            thresh=thresh,
            depth=depth,
            J=J,
            h=h,
            exp_dir=exp_dir,
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="negativity: 固定 NU，扫描最小 NM（third moment grouped MAE）"
    )
    parser.add_argument("--nu", type=int, required=True, help="固定分组大小 NU（正整数）")
    parser.add_argument("--n-min", type=int, default=8, help="扫描起始 n（含）")
    parser.add_argument("--n-max", type=int, default=20, help="扫描结束 n（含）")
    parser.add_argument("--dprob", type=float, default=0.30, help="退相干参数 p")
    parser.add_argument("--thresh", type=float, default=0.1, help="third moment MAE 阈值")
    parser.add_argument("--depth", type=int, default=60, help="文件名中的 depth")
    parser.add_argument("--J", type=float, default=1.0, help="文件名中的 J")
    parser.add_argument("--h", type=float, default=1.0, help="文件名中的 h")
    parser.add_argument(
        "--exp-dir",
        type=str,
        default=str(EXP_DIR),
        help="实验 jsonl 目录（默认 data/ptme/tensor_simulator/exp）",
    )
    args = parser.parse_args()
    if args.nu <= 0:
        raise ValueError("--nu 必须是正整数")
    if args.n_min > args.n_max:
        raise ValueError("--n-min 不能大于 --n-max")

    exp_dir = args.exp_dir
    if not os.path.isabs(exp_dir):
        exp_dir = str(Path.cwd() / exp_dir)

    result = min_nm_third_mae_dict(
        n_min=args.n_min,
        n_max=args.n_max,
        nu=args.nu,
        dprob=args.dprob,
        thresh=args.thresh,
        depth=args.depth,
        J=args.J,
        h=args.h,
        exp_dir=exp_dir,
    )

    for n in range(args.n_min, args.n_max + 1):
        print(n, result[n])

    print(
        f"\n# MIN_NM (fixed NU={args.nu}, negativity third-moment grouped MAE < {args.thresh}, dprob={args.dprob:.2f})"
    )
    print(
        "MIN_NM_THIRD_MAE = {"
        + ", ".join(
            f"{n}: {result[n]['NM']}"
            for n in range(args.n_min, args.n_max + 1)
            if isinstance(result[n].get("NM"), int)
        )
        + "}"
    )

    x_vals: list[float] = []
    y_vals: list[float] = []
    for n in range(args.n_min, args.n_max + 1):
        nm = result[n].get("NM")
        if isinstance(nm, int):
            x_vals.append(float(n))
            y_vals.append(float(args.nu * nm))

    if len(x_vals) >= 2:
        x = np.array(x_vals, dtype=float)
        y = np.array(y_vals, dtype=float)
        slope, intercept, r_value, _, _ = linregress(x, np.log2(y))
        print(f"slope = {slope}")
        print(f"intercept = {intercept}")
        print(f"r_value = {r_value}")
    else:
        print("拟合失败：有效数据点不足 2 个（需要至少两个有解 n）。")


if __name__ == "__main__":
    main()
