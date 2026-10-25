"""Audit plotting inputs separately from raw simulation provenance."""
import argparse
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
MOMENT_COLUMNS = ["purity", "thirdM", "fourthM", "fifthM"]


def read_csv_checked(path, columns, trials=None):
    """Return (frame, issues); malformed data must not pass as available."""
    if not path.exists():
        return None, [f"{path.name}: missing file"]
    try:
        df = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        return None, [f"{path.name}: cannot read CSV: {exc}"]
    issues = []
    if trials is not None and len(df) != trials:
        issues.append(f"{path.name}: expected {trials} trials, found {len(df)}")
    for column in columns:
        if column not in df:
            issues.append(f"{path.name}: missing column {column}")
            continue
        try:
            values = np.array([complex(str(v).replace("i", "j").replace(" ", ""))
                               for v in df[column]], dtype=complex)
            if not np.isfinite(values).all():
                issues.append(f"{path.name}: non-finite values in {column}")
        except (TypeError, ValueError):
            issues.append(f"{path.name}: non-numeric values in {column}")
    return df, issues


def check_cbne_moment(params):
    data = DATA_ROOT / "cbne_moment/TIM_complete"
    issues = []
    conditions = []
    for panel in ("left", "right"):
        p = params[panel]
        samples = p["N_M"] if isinstance(p["N_M"], list) else [p["N_M"]]
        conditions.extend((n, p["circuit_depth_by_n"][str(n)], nm)
                          for n in p["n"] for nm in samples)
    for n, depth, nm in conditions:
        path = data / f"n{n}_depth{depth}_J{params['J']:.2f}_h{params['h']:.2f}_p{params['depolarizing_probability']:.2f}_Sample{nm}.jsonl"
        if not path.exists():
            issues.append(f"{path.name}: missing file")
            continue
        count = 0
        try:
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if not np.isfinite([float(row["Purity"]), float(row["F"])]).all():
                        raise ValueError("non-finite Purity/F")
                    count += 1
            block_size = params.get("block_size", 1)
            if min(count, 5000) % block_size:
                issues.append(f"{path.name}: records cannot form complete blocks of {block_size}")
            if count < params["trials"]:
                issues.append(f"{path.name}: requires at least {params['trials']} records per repository config, found {count}")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            issues.append(f"{path.name}: invalid record: {exc}")
    return issues


def check_virtual_cooling(params):
    data = DATA_ROOT / "virtual_cooling"
    prefix = f"fidelity_gibbs_n{params['n']}_Nu{params['N_U']}_NM{params['N_M']}"
    names = [f"{prefix}_beta{beta:.1f}.csv" for beta in params["left"]["beta"]]
    names += [f"{prefix}_h{h:.1f}.csv" if h in (0.5, 1.0, 1.5)
              else f"{prefix}_h{h:.2f}.csv" for h in params["right"]["h"]]
    issues = []
    for name in names:
        _, found = read_csv_checked(data / name, ["pf_rhoO2_fidelity", "pf_rhoO3_fidelity"], params["trials"])
        issues.extend(found)
    # Smooth exact curves are also required by the plot.
    for axis, value, size in [("h", params["left"]["h"], 31), ("beta", params["right"]["beta"], 21)]:
        for stem in ("without_cooling", "cooling_second", "cooling_third"):
            path = data / f"tfim_{stem}_N{params['n']}_J{params['J']}_{axis}{value}_detail.npy"
            try:
                values = np.load(path, allow_pickle=False)
                if values.shape != (size,) or not np.isfinite(values).all():
                    issues.append(f"{path.name}: expected {size} finite truth values")
            except (OSError, ValueError) as exc:
                issues.append(f"{path.name}: cannot load truth curve: {exc}")
    return issues


def check_observables(params):
    data = DATA_ROOT / "cbne_observables"
    issues = []
    cache = data / f"exact_rho3O_n{params['n']}_beta{params['beta']}.csv"
    frame, found = read_csv_checked(cache, ["value"])
    issues.extend(found)
    if not found:
        if "observable" not in frame:
            issues.append(f"{cache.name}: missing observable column")
        else:
            for observable in ("X1toXn", "Z1Z2", "psib"):
                if int((frame["observable"] == observable).sum()) != 1:
                    issues.append(f"{cache.name}: expected one exact value for {observable}")
    sources = {
        "gibbs_pauli_anc": ["pf_rhoO3_X1toXn", "pf_rhoO3_Z1Z2", "third_moment"],
        "gibbs_psibH_anc": ["pf_rhoO3_psib", "third_moment"],
    }
    for stem, columns in sources.items():
        for anc in params["right"]["n_a"]:
            for nm in params["left"]["N_M"]:
                name = f"{stem}_n{params['n']}_Nu{params['left']['N_U']}_NM{nm}_beta{params['beta']:.1f}_anc{anc}.csv"
                _, found = read_csv_checked(data / name, columns, params["trials"])
                issues.extend(found)
    return issues


def check_dedicated_rho3(params):
    data = DATA_ROOT / "cbne_observables"
    issues = []
    for anc in params["right"]["n_a"]:
        if params.get("state", "gibbs") == "depolarized":
            name = (f"rho3_depol_anc_n{params['n']}_Nu{params['left']['N_U']}_"
                    f"NMmulti_p{float(params['depolarizing_probability'])}_anc{anc}.csv")
        else:
            name = (f"rho3_anc_n{params['n']}_Nu{params['left']['N_U']}_"
                    f"NMmulti_beta{float(params['beta'])}_anc{anc}.csv")
        frame, found = read_csv_checked(data / name, ["idx", "NM", "third_moment"])
        issues.extend(found)
        if found:
            continue
        for nm in params["left"]["N_M"]:
            group = frame.loc[frame["NM"] == nm].sort_values("idx")
            if group["idx"].tolist() != list(range(params["trials"])):
                issues.append(f"{name}: incomplete or duplicate trials at N_M={nm}")
    return issues


def check_ptme_scaling(params):
    data = DATA_ROOT / "ptme"
    issues = []
    for n_a in params["right"]["n_A"]:
        path = data / f"mae_n{n_a + params['n_B']}_p{params['depolarizing_probability']:.2f}.csv"
        df, found = read_csv_checked(path, ["sample", "mae_t4", "mae_t4_blk10", "mae_t4_blk50"])
        issues.extend(found)
        if found:
            continue
        if df["sample"].duplicated().any():
            issues.append(f"{path.name}: duplicate sample sizes")
        if int((df["sample"] == params["right"]["N_M"]).sum()) != 1:
            issues.append(f"{path.name}: expected one row at N_M={params['right']['N_M']}")
        if (df[["sample", "mae_t4", "mae_t4_blk10", "mae_t4_blk50"]] <= 0).any().any():
            issues.append(f"{path.name}: non-positive values cannot be plotted/fitted on log axes")
        if n_a in params["left"]["n_A"] and len(df[df["sample"] <= params["left"]["N_M_max"]]) < 2:
            issues.append(f"{path.name}: insufficient left-panel points")
    return issues


def check_grouped_csv(filename, params, sweep_column, sweep_values):
    path = DATA_ROOT / "ptme" / filename
    df, issues = read_csv_checked(path, ["nA", "nB", "Nu", "NM", sweep_column, *MOMENT_COLUMNS])
    if issues:
        return issues
    base = ((df["nA"] == params["n_A"]) & (df["nB"] == params["n_B"])
            & (df["Nu"] == params["N_U"]) & (df["NM"] == params["N_M"]))
    for value in sweep_values:
        count = int((base & np.isclose(df[sweep_column], value, rtol=0, atol=1e-12)).sum())
        if count != params["trials"]:
            issues.append(f"{filename}: {sweep_column}={value}, expected {params['trials']} trials, found {count}")
    return issues


def select_cbne_right_counts(config):
    """Use a complete exact-budget series, otherwise the configured actual budgets."""
    params = config["SE_vs_NM_and_n.pdf"]
    right = params["right"]
    target = {n: right["N_M"] for n in right["n"]}
    paths = [DATA_ROOT / "cbne_moment/TIM_complete" /
             f"n{n}_depth{right['circuit_depth_by_n'][str(n)]}_J{params['J']:.2f}_h{params['h']:.2f}_p{params['depolarizing_probability']:.2f}_Sample{nm}.jsonl"
             for n, nm in target.items()]
    if all(path.is_file() for path in paths):
        return target
    return {n: right.get("N_M_by_n", {}).get(str(n), right["N_M"]) for n in right["n"]}


def select_cooling_input_label(config):
    """Return a filename label: current total shots or the historical N_M/d label."""
    params = config["vcooling.pdf"]
    prefix = f"fidelity_gibbs_n{params['n']}_Nu{params['N_U']}_NM{params['N_M']}"
    names = [f"{prefix}_beta{beta:.1f}.csv" for beta in params["left"]["beta"]]
    names += [f"{prefix}_h{h:.1f}.csv" if h in (0.5, 1.0, 1.5)
              else f"{prefix}_h{h:.2f}.csv" for h in params["right"]["h"]]
    if all((DATA_ROOT / "virtual_cooling" / name).is_file() for name in names):
        return params["N_M"]
    return config["vcooling_models.pdf"]["retained_tfim_N_M"]


def audit_figure(name):
    config = json.loads((ROOT / "paper_parameters.json").read_text())["figures"]
    notes = []
    if name == "sample_lower.pdf":
        # Validate that the stored values exist without importing plotting code.
        import ast
        source = ast.parse((ROOT / "cbne_moment/plot.py").read_text(encoding="utf-8-sig"))
        tables = {node.targets[0].id: ast.literal_eval(node.value)
                  for node in source.body if isinstance(node, ast.Assign)
                  and isinstance(node.targets[0], ast.Name)
                  and node.targets[0].id in ("SAMPLE_LOWER_CBNE", "SAMPLE_LOWER_PTME")}
        issues = []
        for key in ("SAMPLE_LOWER_CBNE", "SAMPLE_LOWER_PTME"):
            table = tables.get(key, {})
            if set(table) != set(config[name]["n_or_n_A"]) or not all(np.isfinite(v) and v > 0 for v in table.values()):
                issues.append(f"{key}: missing or invalid stored minima")
        return issues, ["Plot uses hard-coded minima; raw-data recomputation is NOT verified."]
    if name == "SE_scaling.pdf":
        combined = config[name]
        upper = deepcopy(config[combined["upper_parameters"]])
        upper["trials"] = 1
        upper["right"]["n"] = []
        issues = check_cbne_moment(upper)
        for n, nm in select_cbne_right_counts(config).items():
            right = deepcopy(config[combined["upper_parameters"]])
            right["trials"] = right["right"].get("records_by_n", {}).get(str(n), 50)
            right["block_size"] = 50
            right["left"]["n"] = []
            right["right"]["n"] = [int(n)]
            right["right"]["N_M"] = nm
            issues.extend(check_cbne_moment(right))
        issues.extend(check_dedicated_rho3(config[combined["lower_parameters"]]))
                
        counts = select_cbne_right_counts(config)
        target_nm = config[combined["upper_parameters"]]["right"]["N_M"]
        if all(nm == target_nm for nm in counts.values()):
            note = f"Upper (b) uses complete fixed total N_M={target_nm} series."
        else:
            budgets = ", ".join(f"n={n}: {nm}" for n, nm in counts.items())
            note = f"Upper (b) total N_M by size: {budgets}. See README for record counts."
        return issues, [note]
    if name == "vcooling_models.pdf":
        combined = config[name]
        tfim = {**config[combined["tfim_parameters"]], "N_M": select_cooling_input_label(config)}
        issues = check_virtual_cooling(tfim)
        result = subprocess.run([sys.executable, str(ROOT / "check_qvc_heisenberg.py")],
                                cwd=ROOT, capture_output=True, text=True, check=False)
        if result.returncode:
            issues.extend((result.stdout + result.stderr).strip().splitlines() or ["Heisenberg data audit failed"])
        note = (f"TFIM uses complete total N_M={tfim['N_M']} series."
                if tfim["N_M"] == config[combined["tfim_parameters"]]["N_M"] else
                "TFIM historical NM1000 labels mean N_M/d (total N_M=16384000); exact 10^6 series is incomplete. Heisenberg uses 10^6.")
        return issues, [note]
    if name == "SE_rho3_gibbs.pdf":
        return check_dedicated_rho3(config[name]), []
    if name == "SE_rho3_anc.pdf":
        return check_dedicated_rho3(config[name]), []
    if name == "TFIM_detection_prob.svg":
        path = DATA_ROOT / "ptme/processed_results2.csv"
        df, issues = read_csv_checked(path, ["nA", "nB", "Nu", "NM", "dprob", *MOMENT_COLUMNS])
        if not issues:
            base = (df["nA"] == 18) & (df["nB"] == 2) & (df["NM"] == 100000000)
            for p in np.round(np.arange(0.2, 0.71, 0.05), 2):
                if not (base & np.isclose(df["dprob"], p, rtol=0, atol=1e-12)).any():
                    issues.append(f"{path.name}: missing dprob={p}")
        return issues, ["Main Fig. 4(b): numerical detection panel; the paper schematic is assembled outside this repository."]
    params = config[name]
    if name == "SE_vs_NM_and_n.pdf":
        issues = check_cbne_moment(params)
        notes.append("The 5000-record requirement comes from repository config; the figure caption does not specify this count.")
    elif name == "vcooling.pdf":
        issues = check_virtual_cooling(params)
    elif name == "SE_all_observables.pdf":
        issues = check_observables(params)
        if "rho3_parameters" in params:
            rho3_display = {**config[params["rho3_parameters"]],
                            "left": params["left"], "right": params["right"]}
            issues.extend(check_dedicated_rho3(rho3_display))
            notes.append("2x4 Gibbs figure: dedicated rho^3 column plus the three retained observables.")
    elif name == "SE_vs_NM_and_nA.pdf":
        issues = check_ptme_scaling(params)
        notes.append("Processed MAE inputs; right panel selects exactly N_M=10^7. Raw simulations are not audited.")
    elif name == "Gibbs_ent_detect.pdf":
        issues = check_grouped_csv("gibbs_dbasis_paper.csv", params, "beta", params["beta"])
    elif name == "Depol_ent_detect.pdf":
        issues = check_grouped_csv("heisenberg_dbasis_prob_paper.csv", params, "dprob", params["depolarizing_probability"])
    else:
        raise ValueError(f"Unknown figure: {name}")
    return issues, notes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--figure", help="audit one figure by its PDF filename")
    args = parser.parse_args()
    config = json.loads((ROOT / "paper_parameters.json").read_text())["figures"]
    if args.figure and args.figure not in config:
        parser.error("--figure must be a filename in paper_parameters.json")
    failed = False
    from reproduce_paper_figures import FIGURES
    for name in ([args.figure] if args.figure else [row[1] for row in FIGURES]):
        issues, notes = audit_figure(name)
        status = "BLOCKED" if issues else ("VALUES_ONLY" if name == "sample_lower.pdf" else "READY")
        print(f"{status:12} {name}")
        for note in notes:
            print(f"             {note}")
        for issue in issues[:12]:
            print(f"             {issue}")
        if len(issues) > 12:
            print(f"             ... and {len(issues) - 12} more issues")
        failed |= bool(issues)
    return 1 if failed and not args.allow_missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
