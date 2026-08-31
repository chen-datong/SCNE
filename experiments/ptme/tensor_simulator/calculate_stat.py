import csv
import glob
import json
import os
import re

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


def _moments_to_processed(m2: float, m3: float, m4: float, m5: float):
    purity = m2 - 1.0
    third_m = (m3 - 1.0 - 3.0 * purity) / 2.0
    fourth_m = (m4 - 1.0 - 6.0 * purity - 8.0 * third_m - 3.0 * (purity ** 2)) / 6.0
    fifth_m = (
        m5
        - 1.0
        - 10.0 * purity
        - 20.0 * third_m
        - 15.0 * (purity ** 2)
        - 30.0 * fourth_m
        - 20.0 * purity * third_m
    ) / 24.0
    return purity, third_m, fourth_m, fifth_m


def _mae(values: list[float], target: float) -> float:
    if not values:
        return float("nan")
    return sum(abs(v - target) for v in values) / len(values)


def _block_means(values: list[float], block_size: int) -> list[float]:
    if block_size <= 0:
        raise ValueError("block_size 必须为正整数")
    means = []
    for i in range(0, len(values), block_size):
        block = values[i : i + block_size]
        if not block:
            continue
        means.append(sum(block) / len(block))
    return means


def calculate_mae_for_n_dprob(
    n: int,
    dprob: float,
    depth: int = 60,
    J: float = 1.0,
    h: float = 1.0,
    exp_dir: str = "exp",
    output_csv: str | None = None,
):
    """
    对给定 n 和 dprob，计算 exp 下每个 Sample 文件的 MAE。
    先由 M2~M5 变换到 (t=2,3,4,5) 对应量：
    t2=purity, t3=thirdM, t4=fourthM, t5=fifthM，
    再分别统计每个 t 的 MAE，并额外统计两种分块 MAE：
    - blk10: 每 10 个 rep 先平均，再算 MAE
    - blk50: 每 50 个 rep 先平均，再算 MAE
    """
    target_t2, target_t3, target_t4, target_t5 = exact_results(n=n, depth=depth, J=J, h=h, dprob=dprob)

    pattern = os.path.join(
        exp_dir, f"n{n}_depth{depth}_J{J:.2f}_h{h:.2f}_p{dprob:.2f}_Sample*.jsonl"
    )
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"未找到匹配文件: {pattern}")

    results = []
    for filepath in files:
        sample = _parse_sample_from_filename(filepath)
        vals_t2 = []
        vals_t3 = []
        vals_t4 = []
        vals_t5 = []

        for rec in _iter_jsonl(filepath):
            m2 = float(rec.get("M2", 0.0))
            m3 = float(rec.get("M3", 0.0))
            m4 = float(rec.get("M4", 0.0))
            m5 = float(rec.get("M5", 0.0))
            t2, t3, t4, t5 = _moments_to_processed(m2, m3, m4, m5)
            vals_t2.append(t2)
            vals_t3.append(t3)
            vals_t4.append(t4)
            vals_t5.append(t5)

        mae_t2 = _mae(vals_t2, target_t2)
        mae_t3 = _mae(vals_t3, target_t3)
        mae_t4 = _mae(vals_t4, target_t4)
        mae_t5 = _mae(vals_t5, target_t5)

        mae_t2_blk10 = _mae(_block_means(vals_t2, 10), target_t2)
        mae_t3_blk10 = _mae(_block_means(vals_t3, 10), target_t3)
        mae_t4_blk10 = _mae(_block_means(vals_t4, 10), target_t4)
        mae_t5_blk10 = _mae(_block_means(vals_t5, 10), target_t5)

        mae_t2_blk50 = _mae(_block_means(vals_t2, 50), target_t2)
        mae_t3_blk50 = _mae(_block_means(vals_t3, 50), target_t3)
        mae_t4_blk50 = _mae(_block_means(vals_t4, 50), target_t4)
        mae_t5_blk50 = _mae(_block_means(vals_t5, 50), target_t5)

        results.append(
            {
                "sample": sample,
                "mae_t2": mae_t2,
                "mae_t3": mae_t3,
                "mae_t4": mae_t4,
                "mae_t5": mae_t5,
                "mae_t2_blk10": mae_t2_blk10,
                "mae_t3_blk10": mae_t3_blk10,
                "mae_t4_blk10": mae_t4_blk10,
                "mae_t5_blk10": mae_t5_blk10,
                "mae_t2_blk50": mae_t2_blk50,
                "mae_t3_blk50": mae_t3_blk50,
                "mae_t4_blk50": mae_t4_blk50,
                "mae_t5_blk50": mae_t5_blk50,
            }
        )

    results.sort(key=lambda x: x["sample"])

    if output_csv is None:
        output_csv = f"mae_n{n}_p{dprob:.2f}.csv"

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample",
                "mae_t2",
                "mae_t3",
                "mae_t4",
                "mae_t5",
                "mae_t2_blk10",
                "mae_t3_blk10",
                "mae_t4_blk10",
                "mae_t5_blk10",
                "mae_t2_blk50",
                "mae_t3_blk50",
                "mae_t4_blk50",
                "mae_t5_blk50",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    return results, output_csv


if __name__ == "__main__":
    # 示例：python calculate_stat.py
    for n in [8, 10, 12, 14, 16, 18, 20]:
        out = calculate_mae_for_n_dprob(n=n, dprob=0.30)
        print(f"已输出: {out[1]}")
