import json
import pandas as pd
import os
import glob
import sympy as sp

def tr_depol_t(trA, t, alpha, beta):
    expr = 0
    for k in range(t + 1):
        expr += sp.binomial(t, k) * alpha**k * beta**(t - k) * trA[k]
    return sp.simplify(expr)

def exact_results(n, depth, J, h, dprob):
    # M2, M3, M4, M5 取自 TIM_exact 文件；该文件各行相同，这里取第一条有效记录
    file_pattern = os.path.join(
        "TIM_exact",
        f"n{n}_depth{depth}_J{J:.2f}_h{h:.2f}_exact.jsonl",
    )
    files = glob.glob(file_pattern)
    if not files:
        raise FileNotFoundError(f"未找到 exact 文件: {file_pattern}")

    M2 = M3 = M4 = M5 = None
    with open(files[0], "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            M2 = float(rec.get("M2", 0.0))
            M3 = float(rec.get("M3", 0.0))
            M4 = float(rec.get("M4", 0.0))
            M5 = float(rec.get("M5", 0.0))
            break

    if M2 is None:
        raise ValueError(f"文件中没有可用记录: {files[0]}")

    d = 2 ** n
    trace = {0: d, 1: 1.0, 2: M2, 3: M3, 4: M4, 5: M5}
    p = dprob
    p2 = tr_depol_t(trace, 2, 1 - p, p / d)
    p3 = tr_depol_t(trace, 3, 1 - p, p / d)
    p4 = tr_depol_t(trace, 4, 1 - p, p / d)
    p5 = tr_depol_t(trace, 5, 1 - p, p / d)
    return float(p2), float(p3), float(p4), float(p5)


def main():
    pass


if __name__ == "__main__":
    main()
