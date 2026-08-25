#!/usr/bin/env python3
"""Run several numeric trainings and plot train loss / R2 versus epochs.

The script intentionally uses the Rust CLI as a black box:

    transformer train <data.tnum> <epochs> <model.bin>

It updates a live matplotlib chart when matplotlib is available. In headless
mode it still saves a PNG and always writes a CSV with the collected metrics.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_DATA = SCRIPT_DIR / "example_complex.tnum"
DEFAULT_OUT_DIR = SCRIPT_DIR / "runs"

LOSS_RE = re.compile(r"train loss.*=\s*([-+0-9.eE]+)")
RMSE_RE = re.compile(r"RMSE\s*=\s*([-+0-9.eE]+)")
MAE_RE = re.compile(r"MAE\s*=\s*([-+0-9.eE]+)")
REL_RE = re.compile(r"rel\. error\s*=\s*([-+0-9.eE]+)%")
R2_RE = re.compile(r"R²\s*=\s*([-+0-9.eE]+)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sweep epoch counts for train and plot train loss / R2."
    )
    p.add_argument(
        "--data",
        default=str(DEFAULT_DATA),
        help=f"Path to .tnum dataset. Default: {DEFAULT_DATA}",
    )
    p.add_argument(
        "--epochs",
        default="1,2,5,10,20,40",
        help="Comma-separated epoch counts, e.g. 1,2,5,10,20,40",
    )
    p.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help=f"Directory for models, CSV, and PNG. Default: {DEFAULT_OUT_DIR}",
    )
    p.add_argument(
        "--binary",
        default="",
        help="Path to transformer binary. Default: target/release/transformer",
    )
    p.add_argument(
        "--no-live",
        action="store_true",
        help="Do not open/update an interactive chart; still save PNG if possible.",
    )
    p.add_argument(
        "--min-r2-gain",
        type=float,
        default=0.02,
        help=(
            "Recommended-stop threshold: when the next R2 gain is below this, "
            "the previous epoch count is marked. Default: 0.02"
        ),
    )
    p.add_argument(
        "--target-r2",
        type=float,
        default=0.95,
        help="Recommend the first epoch count that reaches this R2. Default: 0.95",
    )
    p.add_argument(
        "--plateau-min-r2",
        type=float,
        default=0.80,
        help=(
            "Only use the small-gain plateau rule after R2 reaches this value. "
            "Default: 0.80"
        ),
    )
    return p.parse_args()


def parse_epoch_list(raw: str) -> list[int]:
    epochs = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        n = int(part)
        if n <= 0:
            raise SystemExit("Epoch counts must be > 0")
        epochs.append(n)
    if not epochs:
        raise SystemExit("No epochs provided")
    return epochs


def ensure_binary(path: Path) -> Path:
    if path.exists():
        return path
    print(f"[build] {path} not found; running cargo build --release")
    subprocess.run(["cargo", "build", "--release"], cwd=REPO_ROOT, check=True)
    if not path.exists():
        raise SystemExit(f"Release binary was not created: {path}")
    return path


def maybe_import_pyplot(no_live: bool):
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "transformer-matplotlib")
    )
    try:
        import matplotlib

        if no_live or (not os.environ.get("DISPLAY") and sys.platform != "darwin"):
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception as exc:  # pragma: no cover - depends on local Python setup.
        print(f"[plot] matplotlib unavailable, using table only: {exc}")
        return None


def run_training(binary: Path, data: Path, epochs: int, out_dir: Path) -> dict[str, float]:
    model_path = out_dir / f"model_e{epochs}.bin"
    cmd = [str(binary), "train", str(data), str(epochs), str(model_path)]
    print("\n" + "=" * 80)
    print("$ " + " ".join(cmd))
    print("=" * 80)

    proc = subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    assert proc.stdout is not None

    last_loss = None
    rmse = mae = rel_error = r2 = None
    for line in proc.stdout:
        print(line, end="")
        if m := LOSS_RE.search(line):
            last_loss = float(m.group(1))
        elif m := RMSE_RE.search(line):
            rmse = float(m.group(1))
        elif m := MAE_RE.search(line):
            mae = float(m.group(1))
        elif m := REL_RE.search(line):
            rel_error = float(m.group(1)) / 100.0
        elif m := R2_RE.search(line):
            # There are two R2 lines: test and after load. They should match.
            r2 = float(m.group(1))

    rc = proc.wait()
    if rc != 0:
        raise SystemExit(f"Training failed for epochs={epochs} with exit code {rc}")
    if None in (last_loss, rmse, mae, rel_error, r2):
        raise SystemExit(f"Could not parse all metrics for epochs={epochs}")

    return {
        "epochs": float(epochs),
        "train_loss": float(last_loss),
        "rmse": float(rmse),
        "mae": float(mae),
        "rel_error": float(rel_error),
        "r2": float(r2),
    }


def write_results(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["epochs", "train_loss", "rmse", "mae", "rel_error", "r2"]
        )
        w.writeheader()
        for row in rows:
            out = dict(row)
            out["epochs"] = int(out["epochs"])
            w.writerow(out)


def recommended_stop(
    rows: list[dict[str, float]],
    min_r2_gain: float,
    target_r2: float,
    plateau_min_r2: float,
) -> tuple[dict[str, float], str] | None:
    if not rows:
        return None
    for row in rows:
        if row["r2"] >= target_r2:
            return row, f"target R²≥{target_r2:g}"
    for prev, cur in zip(rows, rows[1:]):
        gain = cur["r2"] - prev["r2"]
        if prev["r2"] >= plateau_min_r2 and gain < min_r2_gain:
            return prev, f"plateau ΔR²<{min_r2_gain:g}"
    return rows[-1], "best so far"


def update_plot(
    plt,
    fig,
    ax_loss,
    ax_r2,
    rows: list[dict[str, float]],
    png: Path,
    live: bool,
    min_r2_gain: float,
    target_r2: float,
    plateau_min_r2: float,
) -> None:
    xs = [int(r["epochs"]) for r in rows]
    losses = [r["train_loss"] for r in rows]
    r2s = [r["r2"] for r in rows]

    ax_loss.clear()
    ax_r2.clear()
    ax_loss.plot(xs, losses, marker="o", color="#1f77b4", label="train loss")
    ax_r2.plot(xs, r2s, marker="s", color="#d62728", label="R²")

    ax_loss.set_xlabel("epochs")
    ax_loss.set_ylabel("train loss (normalized)", color="#1f77b4")
    ax_r2.set_ylabel("R² on test", color="#d62728")
    ax_r2.yaxis.set_label_position("right")
    ax_r2.yaxis.tick_right()
    ax_loss.tick_params(axis="y", labelcolor="#1f77b4")
    ax_r2.tick_params(axis="y", labelcolor="#d62728")
    ax_loss.grid(True, alpha=0.3)
    ax_r2.set_ylim(min(-0.05, min(r2s) - 0.05), 1.02)

    rec = recommended_stop(rows, min_r2_gain, target_r2, plateau_min_r2)
    if rec is not None:
        row, reason = rec
        epoch = int(row["epochs"])
        ax_loss.axvline(epoch, color="#555555", linestyle="-.", linewidth=1.6, alpha=0.9)
        y_top = ax_loss.get_ylim()[1]
        max_epoch = max(xs)
        place_left = epoch >= max_epoch * 0.75
        ax_loss.annotate(
            f"recommended stop\n{epoch} epochs\n{reason}",
            xy=(epoch, y_top),
            xytext=(-8 if place_left else 8, -14),
            textcoords="offset points",
            ha="right" if place_left else "left",
            va="top",
            fontsize=9,
            color="#333333",
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#777777", "alpha": 0.85},
        )

    fig.suptitle("Epoch sweep: train loss down, R² up")
    fig.subplots_adjust(left=0.12, right=0.88, bottom=0.13, top=0.88)
    fig.savefig(png, dpi=160)
    if live:
        plt.pause(0.05)


def print_summary(rows: list[dict[str, float]]) -> None:
    print("\nSummary")
    print("epochs  train_loss  RMSE     MAE      rel.error  R²")
    for r in rows:
        print(
            f"{int(r['epochs']):>6}  {r['train_loss']:>10.5f}  "
            f"{r['rmse']:>7.5f}  {r['mae']:>7.5f}  "
            f"{r['rel_error'] * 100:>8.2f}%  {r['r2']:>7.5f}"
        )


def print_recommendation(
    rows: list[dict[str, float]],
    min_r2_gain: float,
    target_r2: float,
    plateau_min_r2: float,
) -> None:
    rec = recommended_stop(rows, min_r2_gain, target_r2, plateau_min_r2)
    if rec is None:
        return
    row, reason = rec
    print(
        f"Recommended stop: {int(row['epochs'])} epochs "
        f"(R²={row['r2']:.5f}, rel.error={row['rel_error'] * 100:.2f}%, {reason}; "
        f"target R²={target_r2:g}, plateau after R²≥{plateau_min_r2:g} "
        f"with ΔR²<{min_r2_gain:g})"
    )


def main() -> None:
    args = parse_args()
    data = Path(args.data).resolve()
    if not data.exists():
        raise SystemExit(f"Dataset not found: {data}")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    binary = Path(args.binary).resolve() if args.binary else REPO_ROOT / "target/release/transformer"
    binary = ensure_binary(binary)
    epoch_counts = parse_epoch_list(args.epochs)

    plt = maybe_import_pyplot(args.no_live)
    fig = ax_loss = ax_r2 = None
    png = out_dir / "epoch_sweep.png"
    if plt is not None:
        if args.no_live:
            plt.ioff()
        else:
            plt.ion()
        fig, ax_loss = plt.subplots(figsize=(8, 4.8))
        ax_r2 = ax_loss.twinx()

    rows: list[dict[str, float]] = []
    csv_path = out_dir / "epoch_sweep_results.csv"
    for epochs in epoch_counts:
        rows.append(run_training(binary, data, epochs, out_dir))
        write_results(csv_path, rows)
        print_summary(rows)
        print_recommendation(rows, args.min_r2_gain, args.target_r2, args.plateau_min_r2)
        if plt is not None:
            update_plot(
                plt,
                fig,
                ax_loss,
                ax_r2,
                rows,
                png,
                live=not args.no_live,
                min_r2_gain=args.min_r2_gain,
                target_r2=args.target_r2,
                plateau_min_r2=args.plateau_min_r2,
            )

    if plt is not None:
        update_plot(
            plt,
            fig,
            ax_loss,
            ax_r2,
            rows,
            png,
            live=not args.no_live,
            min_r2_gain=args.min_r2_gain,
            target_r2=args.target_r2,
            plateau_min_r2=args.plateau_min_r2,
        )
        print(f"\nSaved plot: {png}")
    print(f"Saved metrics: {csv_path}")


if __name__ == "__main__":
    main()
