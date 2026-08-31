"""Check retained data against the manuscript parameter contract."""

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"


def result(name, missing):
    if missing:
        print(f"MISSING  {name}")
        for item in missing[:12]:
            print(f"         {item}")
        if len(missing) > 12:
            print(f"         ... and {len(missing) - 12} more")
        return False
    print(f"OK       {name}")
    return True


def check_cbne_moment(params):
    data = EXPERIMENTS / "cbne_moment" / "data" / "TIM_complete"
    missing = []
    depth = {int(n): value for n, value in params["left"]["circuit_depth_by_n"].items()}
    for n in params["left"]["n"]:
        for nm in params["left"]["N_M"]:
            path = data / f"n{n}_depth{depth[n]}_J1.00_h1.00_p0.20_Sample{nm}.jsonl"
            if not path.exists():
                missing.append(path.name)
    right_depth = {int(n): value for n, value in params["right"]["circuit_depth_by_n"].items()}
    nm = params["right"]["N_M"]
    for n in params["right"]["n"]:
        path = data / f"n{n}_depth{right_depth[n]}_J1.00_h1.00_p0.20_Sample{nm}.jsonl"
        if not path.exists():
            missing.append(path.name)
    return missing


def check_virtual_cooling(params):
    data = EXPERIMENTS / "virtual_cooling" / "data"
    prefix = f"fidelity_gibbs_n{params['n']}_Nu{params['N_U']}_NM{params['N_M']}"
    names = [f"{prefix}_beta{beta:.1f}.csv" for beta in params["left"]["beta"]]
    names += [
        f"{prefix}_h{h:.1f}.csv" if h in (0.5, 1.0, 1.5) else f"{prefix}_h{h:.2f}.csv"
        for h in params["right"]["h"]
    ]
    missing = []
    for name in names:
        path = data / name
        if not path.exists():
            missing.append(name)
            continue
        if len(pd.read_csv(path)) != params["trials"]:
            missing.append(f"{name}: expected {params['trials']} trials")
    return missing


def check_observables(params):
    data = EXPERIMENTS / "cbne_observables" / "data"
    nm_grid = params["left"]["N_M"]
    missing = []
    for stem in ("gibbs_pauli_anc", "gibbs_psibH_anc"):
        for anc in params["right"]["n_a"]:
            for nm in nm_grid:
                name = f"{stem}_n6_Nu1_NM{nm}_beta2.0_anc{anc}.csv"
                path = data / name
                if not path.exists():
                    missing.append(name)
                elif len(pd.read_csv(path)) != params["trials"]:
                    missing.append(f"{name}: expected {params['trials']} trials")
    return missing


def check_ptme_scaling(params):
    data = EXPERIMENTS / "ptme" / "data"
    missing = []
    for n_a in params["right"]["n_A"]:
        n_total = n_a + params["n_B"]
        name = f"mae_n{n_total}_p0.30.csv"
        path = data / name
        if not path.exists():
            missing.append(name)
            continue
        df = pd.read_csv(path, usecols=["sample"])
        if int(params["right"]["N_M"]) not in set(df["sample"].astype(int)):
            missing.append(f"{name}: missing N_M={params['right']['N_M']}")
    return missing


def check_grouped_csv(filename, params, sweep_column, sweep_values):
    path = EXPERIMENTS / "ptme" / filename
    if not path.exists():
        return [filename]
    df = pd.read_csv(path)
    missing = []
    base = (
        (df["nA"] == params["n_A"])
        & (df["nB"] == params["n_B"])
        & (df["Nu"] == params["N_U"])
        & (df["NM"] == params["N_M"])
    )
    for value in sweep_values:
        count = int((base & (df[sweep_column] == value)).sum())
        if count != params["trials"]:
            missing.append(
                f"{filename}: {sweep_column}={value}, expected {params['trials']} trials, found {count}"
            )
    return missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="report missing caption-parameter data without returning a failure status",
    )
    args = parser.parse_args()

    config_path = ROOT / "configs" / "paper_parameters.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))["figures"]

    checks = [
        result("SE_vs_NM_and_n.pdf", check_cbne_moment(config["SE_vs_NM_and_n.pdf"])),
        result("sample_lower.pdf", []),
        result("vcooling.pdf", check_virtual_cooling(config["vcooling.pdf"])),
        result("SE_all_observables.pdf", check_observables(config["SE_all_observables.pdf"])),
        result("SE_vs_NM_and_nA.pdf", check_ptme_scaling(config["SE_vs_NM_and_nA.pdf"])),
        result(
            "Gibbs_ent_detect.pdf",
            check_grouped_csv(
                "gibbs_dbasis_paper.csv",
                config["Gibbs_ent_detect.pdf"],
                "beta",
                config["Gibbs_ent_detect.pdf"]["beta"],
            ),
        ),
        result(
            "Depol_ent_detect.pdf",
            check_grouped_csv(
                "heisenberg_dbasis_prob_paper.csv",
                config["Depol_ent_detect.pdf"],
                "dprob",
                config["Depol_ent_detect.pdf"]["depolarizing_probability"],
            ),
        ),
    ]
    if not all(checks) and not args.allow_missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
