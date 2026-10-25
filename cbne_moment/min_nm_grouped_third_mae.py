"""Find the smallest NM with grouped third-moment MAE below the threshold.

Average each block of NU records before computing the moment.
Drop incomplete blocks and scan NM in ascending order.

Run: python cbne_moment/min_nm_grouped_third_mae.py --nu 4"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy.stats import linregress


def cal3moment(n: int, p: float) -> tuple[float, float]:
    """Return exact purity and tr(rho^3), as in plot.py."""
    d = 2**n
    aux = p * p - 2 * p + 1 + 2 / d * p * (1 - p)
    purity = aux + p * p / d
    third = (
        aux * (1 - p)
        + aux * p / d
        + p * p * (1 - p) / d / d
        + p * p * p / d / d
    )
    return purity, third


def load_purity_f(path: Path) -> tuple[np.ndarray, np.ndarray]:
    p_list: list[float] = []
    f_list: list[float] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            p_list.append(float(d["Purity"]))
            f_list.append(float(d["F"]))
    return np.array(p_list, dtype=np.float64), np.array(f_list, dtype=np.float64)


def grouped_third_mae(
    purity: np.ndarray,
    f: np.ndarray,
    n: int,
    p: float,
    nu: int,
) -> tuple[float, int]:
    """Return grouped MAE and block count, or (nan, 0) if no full block exists."""
    _, exact_t = cal3moment(n, p)
    total = len(purity)
    num_groups = total // nu
    if num_groups == 0:
        return float("nan"), 0

    errs: list[float] = []
    for i in range(num_groups):
        sl = slice(i * nu, (i + 1) * nu)
        pbar = float(purity[sl].mean())
        fbar = float(f[sl].mean())
        purity_g = pbar - 1.0
        third_m_g = (fbar - 1.0 - 3.0 * purity_g) / 2.0
        errs.append(abs(third_m_g - exact_t))

    return float(np.mean(errs)), num_groups


def collect_nm_files(
    tim_lower_dir: Path,
    n: int,
    depth: int,
    j: float = 1.0,
    h: float = 1.0,
    p_val: float = 0.2,
) -> list[tuple[int, Path]]:
    """Return matching (NM, path) pairs sorted by NM."""
    j_str, h_str, p_str = f"{j:.2f}", f"{h:.2f}", f"{p_val:.2f}"
    pattern = f"n{n}_depth{depth}_J{j_str}_h{h_str}_p{p_str}_Sample*.jsonl"
    sample_re = re.compile(r"_Sample(\d+)\.jsonl$")

    pairs: list[tuple[int, Path]] = []
    for fp in tim_lower_dir.glob(pattern):
        m = sample_re.search(fp.name)
        if not m:
            continue
        nm = int(m.group(1))
        pairs.append((nm, fp))

    pairs.sort(key=lambda x: x[0])
    return pairs


def min_nm_third_mae(
    tim_lower_dir: Path,
    n: int,
    nu: int,
    depth: int,
    p: float,
    thresh: float = 0.1,
) -> dict[str, int | float | str | None]:
    """Find the smallest NM with grouped MAE below thresh.
Return NM, MAE_thirdM, num_groups, and file, or a no-solution error."""
    nm_files = collect_nm_files(tim_lower_dir, n, depth, p_val=p)
    for nm, fp in nm_files:
        purity, f = load_purity_f(fp)
        mae, num_groups = grouped_third_mae(purity, f, n, p, nu)
        if num_groups > 0 and mae < thresh:
            return {
                "NM": nm,
                "MAE_thirdM": mae,
                "num_groups": num_groups,
                "file": fp.name,
            }
    return {"NM": None, "error": "no_solution_or_missing_files"}


def min_nm_third_mae_dict(
    nu: int,
    thresh: float = 0.1,
    depth: int = 50,
    p: float = 0.2,
    n_min: int = 6,
    n_max: int = 17,
) -> dict[int, dict[str, int | float | str | None]]:
    """Find the minimum NM for each n from n_min through n_max."""
    if nu <= 0:
        raise ValueError("nu 必须是正整数")
    if n_min > n_max:
        raise ValueError("n_min 不能大于 n_max")

    root = Path(__file__).resolve().parent.parent
    tim_dir = root / "data" / "cbne_moment" / "TIM_lower"

    result: dict[int, dict[str, int | float | str | None]] = {}
    for n in range(n_min, n_max + 1):
        result[n] = min_nm_third_mae(
            tim_lower_dir=tim_dir,
            n=n,
            nu=nu,
            depth=depth,
            p=p,
            thresh=thresh,
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="固定 NU，扫描最小 NM（仅三阶矩 MAE）")
    parser.add_argument("--nu", type=int, required=True, help="固定分组大小 NU（正整数）")
    parser.add_argument("--thresh", type=float, default=0.1, help="三阶矩分组 MAE 阈值")
    parser.add_argument("--depth", type=int, default=50, help="文件名中的 depth")
    parser.add_argument("--p", type=float, default=0.2, help="物理参数 p（与文件名一致）")
    parser.add_argument("--n-min", type=int, default=6, help="扫描起始 n（含）")
    parser.add_argument("--n-max", type=int, default=17, help="扫描结束 n（含）")
    args = parser.parse_args()

    if args.nu <= 0:
        raise ValueError("--nu 必须是正整数")
    if args.n_min > args.n_max:
        raise ValueError("--n-min 不能大于 --n-max")

    result = min_nm_third_mae_dict(
        nu=args.nu,
        thresh=args.thresh,
        depth=args.depth,
        p=args.p,
        n_min=args.n_min,
        n_max=args.n_max,
    )
    for n in range(args.n_min, args.n_max + 1):
        print(n, result[n])

    print(f"\n# MIN_NM (fixed NU={args.nu}, third-moment grouped MAE < {args.thresh})")
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
