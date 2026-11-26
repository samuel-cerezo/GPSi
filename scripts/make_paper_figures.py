#!/usr/bin/env python
"""Regenerate the figures and tables of the paper from the result CSVs.

Run the experiments first (see README), e.g.::

    python run_experiments.py --dataset euroc --mode sweep
    python run_experiments.py --dataset euroc --mode criterion
    python scripts/make_paper_figures.py --results results --out paper_outputs

Outputs (per dataset found under ``--results``):

* ``<dataset>-ate-vs-gnss.{pdf,png}``       ATE vs. # GNSS measurements before T_wb optimization
* ``<dataset>-rotation-vs-gnss.{pdf,png}``  rotation RMSE vs. # GNSS measurements
* ``<dataset>-criterion.{pdf,png}``         observability criterion curves
* ``<dataset>-table.{md,tex}``              ATE / rotation RMSE: baseline (k=1) vs. criterion,
                                            plus the best k of the sweep
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pandas as pd  # noqa: E402

from gpsi.plotting import plot_criterion_curve, plot_error_vs_num_gnss, read_results  # noqa: E402


def _seq_name(path, suffix):
    return os.path.basename(path)[: -len(suffix)]


def _groups(dataset, names):
    """Split EuRoC into Machine Hall / Vicon Room panels (<= 8 series per panel)."""
    if dataset == "euroc":
        mh = [n for n in names if n.upper().startswith("MH")]
        v = [n for n in names if n not in mh]
        return [(t, g) for t, g in (("Machine Hall", mh), ("Vicon Room", v)) if g] or None
    return None


def _ate_label(offset):
    return "ATE [m]" if offset == 0 else f"ATE + {offset:g} [m]"


def make_tables(dataset, sweeps, criteria, out_dir, rot_col):
    rows = []
    for name in sorted(set(sweeps) | set(criteria)):
        row = {"sequence": name}
        if name in criteria:
            df = criteria[name]
            base = df[~df["obs_criterion"]]
            crit = df[df["obs_criterion"]]
            if len(base):
                row["ATE k=1 [m]"] = base["ate_m"].mean()
                row[f"rot k=1 [{rot_col[-3:]}]"] = base[rot_col].mean()
            if len(crit):
                row["k (criterion)"] = int(crit["min_gps_for_alignment"].iloc[0])
                row["ATE criterion [m]"] = crit["ate_m"].mean()
                row[f"rot criterion [{rot_col[-3:]}]"] = crit[rot_col].mean()
        if name in sweeps:
            agg = sweeps[name].groupby("min_gps_for_alignment")["ate_m"].mean()
            row["best k (sweep)"] = int(agg.idxmin())
            row["ATE best k [m]"] = agg.min()
        rows.append(row)
    if not rows:
        return None
    table = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    cells = [[_fmt(v) for v in r] for r in table.itertuples(index=False)]
    cols = list(table.columns)
    with open(os.path.join(out_dir, f"{dataset}-table.md"), "w") as f:
        f.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        for r in cells:
            f.write("| " + " | ".join(r) + " |\n")
    with open(os.path.join(out_dir, f"{dataset}-table.tex"), "w") as f:
        f.write("\\begin{tabular}{l" + "r" * (len(cols) - 1) + "}\n\\hline\n")
        f.write(" & ".join(c.replace("_", "\\_") for c in cols) + " \\\\\n\\hline\n")
        for r in cells:
            f.write(" & ".join(c.replace("_", "\\_") for c in r) + " \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")
    return table


def _fmt(v):
    if isinstance(v, float):
        return "--" if pd.isna(v) else f"{v:.4f}"
    return str(v)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="results", help="folder with <dataset>/ sub-folders of CSVs")
    p.add_argument("--out", default="paper_outputs")
    p.add_argument("--datasets", nargs="*", default=["euroc", "gvins", "mars"])
    p.add_argument("--rotation-unit", choices=["deg", "rad"], default="deg")
    p.add_argument("--ate-offset", type=float, default=0.0,
                   help="constant added to the plotted ATE (the original plotting script used one; default 0)")
    p.add_argument("--formats", nargs="*", default=["pdf", "png"])
    args = p.parse_args(argv)
    rot_col = f"rot_rmse_{args.rotation_unit}"

    produced = False
    for dataset in args.datasets:
        folder = os.path.join(args.results, dataset)
        if not os.path.isdir(folder):
            continue
        sweeps, criteria, curves, activation = {}, {}, {}, {}
        for path in sorted(glob.glob(os.path.join(folder, "*.csv"))):
            if path.endswith("-obs-criteria.csv"):
                name = _seq_name(path, "-obs-criteria.csv")
                criteria[name] = read_results(path)
                crit = criteria[name][criteria[name]["obs_criterion"]]
                if len(crit):
                    activation[name] = int(crit["min_gps_for_alignment"].iloc[0])
            elif path.endswith("-criterion-curve.csv"):
                curves[_seq_name(path, "-criterion-curve.csv")] = pd.read_csv(path)
            elif path.endswith("-single.csv"):
                continue
            else:
                sweeps[_seq_name(path, ".csv")] = read_results(path)

        if sweeps:
            if args.ate_offset:
                for df in sweeps.values():
                    df["ate_m"] = df["ate_m"] + args.ate_offset
            groups = _groups(dataset, sorted(sweeps))
            for fmt in args.formats:
                plot_error_vs_num_gnss(sweeps, "ate_m", _ate_label(args.ate_offset),
                                       os.path.join(args.out, f"{dataset}-ate-vs-gnss.{fmt}"), groups)
                if sweeps and rot_col in next(iter(sweeps.values())):
                    plot_error_vs_num_gnss(sweeps, rot_col, f"rotation RMSE [{args.rotation_unit}]",
                                           os.path.join(args.out, f"{dataset}-rotation-vs-gnss.{fmt}"), groups)
            if args.ate_offset:
                for df in sweeps.values():
                    df["ate_m"] = df["ate_m"] - args.ate_offset
        if curves:
            for fmt in args.formats:
                plot_criterion_curve(curves, os.path.join(args.out, f"{dataset}-criterion.{fmt}"),
                                     activation=activation)
        table = make_tables(dataset, sweeps, criteria, args.out, rot_col)
        if table is not None:
            print(f"== {dataset} ==")
            print(table.to_string(index=False))
            produced = True

    if not produced:
        print(f"No result CSVs found under '{args.results}'. Run run_experiments.py first.", file=sys.stderr)
        return 1
    print(f"Figures and tables written to '{args.out}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
