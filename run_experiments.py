#!/usr/bin/env python
"""Single entry point to reproduce the experiments of the paper.

Examples::

    # Error vs. number of GNSS measurements on all EuRoC sequences found
    python run_experiments.py --dataset euroc --data-root datasets/EuRoC --mode sweep

    # Observability criterion vs. baseline on two GVINS sequences
    python run_experiments.py --dataset gvins --data-root datasets/GVINS \
        --sequences sports_field urban_driving --mode criterion

    # Quick sanity check on a synthetic trajectory (no dataset needed)
    python run_experiments.py --dataset synthetic --max-gps 20 --max-epochs 150
"""

import argparse
import os
import sys
import warnings

# Silence noisy third-party warnings (torch.jit deprecation, PyPose shape notices).
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", message="Tensor Shape Invalid")

from gpsi.data import DEFAULT_TIME_MODE, TIME_MODES  # noqa: E402
from gpsi.device import get_device  # noqa: E402
from gpsi.experiments import resolve_sequences, run_sequence, write_results_csv  # noqa: E402
from gpsi.method import InitConfig, run_initialization  # noqa: E402

DEFAULT_ROOTS = {"euroc": "datasets/EuRoC", "gvins": "datasets/GVINS", "mars": "datasets/MARS"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=["euroc", "gvins", "mars", "synthetic"])
    p.add_argument("--data-root", default=None, help="dataset root folder (default: datasets/<DATASET>)")
    p.add_argument("--sequences", nargs="*", default=None,
                   help="sequence folder names under --data-root (default: all available)")
    p.add_argument("--mode", default="sweep", choices=["sweep", "criterion", "single"])
    p.add_argument("--out-dir", default="results")
    p.add_argument("--device", default="auto", help="auto (CUDA if available, else CPU), cpu, cuda, ...")
    p.add_argument("--time-mode", default=None, choices=TIME_MODES,
                   help="timestamp handling; default: " + ", ".join(f"{k}={v}" for k, v in DEFAULT_TIME_MODE.items()))

    g = p.add_argument_group("method parameters (defaults = paper)")
    d = InitConfig()
    g.add_argument("--max-gps", type=int, default=d.max_gps_measurements, help="GNSS epochs in the window")
    g.add_argument("--min-gps", type=int, default=d.min_gps_for_alignment, help="k for --mode single")
    g.add_argument("--no-twb", action="store_true", help="disable T_wb estimation (--mode single)")
    g.add_argument("--obs-criterion", action="store_true", help="use the observability criterion (--mode single)")
    g.add_argument("--gps-noise", type=float, default=d.gps_noise_std, help="simulated GNSS noise std [m]")
    g.add_argument("--gravity-init", type=float, nargs=3, default=list(d.gravity_init))
    g.add_argument("--rot-init-noise", type=float, default=d.rot_init_noise)
    g.add_argument("--max-epochs", type=int, default=d.max_epochs)
    g.add_argument("--lr", type=float, default=d.lr)
    g.add_argument("--seed", type=int, default=d.seed)
    g.add_argument("--repetitions", type=int, default=1, help="runs per configuration (seed, seed+1, ...)")
    g.add_argument("--sweep-step", type=int, default=5)
    g.add_argument("--log-every", type=int, default=0, help="print losses every N epochs (0 = quiet)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = get_device(args.device)
    print(f"[gpsi] device: {device}")

    cfg = InitConfig(
        max_gps_measurements=args.max_gps, use_twb=not args.no_twb, min_gps_for_alignment=args.min_gps,
        use_obs_criterion=args.obs_criterion, gps_noise_std=args.gps_noise,
        gravity_init=tuple(args.gravity_init), rot_init_noise=args.rot_init_noise,
        max_epochs=args.max_epochs, lr=args.lr, seed=args.seed, log_every=args.log_every,
    )

    if args.dataset == "synthetic":
        from gpsi.synthetic import generate_synthetic_sequence
        data = generate_synthetic_sequence(n_epochs=args.max_gps, gnss_rate=10.0, imu_rate=50.0,
                                           gravity=cfg.gravity_init, device=device)
        res = run_initialization(data, cfg, device=device)
        write_results_csv(os.path.join(args.out_dir, "synthetic", "synthetic-single.csv"),
                          [{"dataset": "synthetic", "sequence": "synthetic", **res.metrics}])
        return 0

    root = args.data_root or DEFAULT_ROOTS[args.dataset]
    sequences = resolve_sequences(args.dataset, root, args.sequences)
    if not sequences:
        print(f"[gpsi] no sequences found under '{root}'. See README.md (Datasets).", file=sys.stderr)
        return 1

    for name, path in sequences:
        run_sequence(args.dataset, path, name, args.mode, cfg, device=device, time_mode=args.time_mode,
                     repetitions=args.repetitions, sweep_step=args.sweep_step, out_dir=args.out_dir)
    print(f"[gpsi] results written to '{os.path.join(args.out_dir, args.dataset)}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
