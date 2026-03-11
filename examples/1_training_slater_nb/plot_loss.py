#!/usr/bin/env python
"""Plot training loss curve from log file produced by train_dimer_backend.py."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ── parser ────────────────────────────────────────────────────────────────────

def parse_log(log_path: str) -> tuple[list[int], list[float], dict[str, list[float]]]:
    """Parse epoch log lines and return (epochs, avg_losses, per_pair_losses).

    Supports both formats:
      - plain:     epoch=0001  avg=5999.7041  time=0.5s
      - with ts:   2026-03-02 09:53:08 epoch=0001  avg=5999.7041  time=0.5s
    The per-pair line that follows looks like:
      [DMC_DMC=498.158  EC_EC=583.478  ...]
    """
    epoch_re = re.compile(r"epoch=(\d+)\s+avg=([\d.eE+\-]+)")
    pair_re  = re.compile(r"(\w+)=([\d.eE+\-]+)")

    epochs: list[int] = []
    avg_losses: list[float] = []
    pair_losses: dict[str, list[float]] = {}
    pending_epoch: int | None = None

    with open(log_path) as f:
        for line in f:
            # strip optional timestamp prefix
            line = line.strip()
            # check for epoch line
            m = epoch_re.search(line)
            if m:
                pending_epoch = int(m.group(1))
                epochs.append(pending_epoch)
                avg_losses.append(float(m.group(2)))
                continue
            # check for per-pair line
            if pending_epoch is not None and "[" in line:
                for pm in pair_re.finditer(line):
                    name, val = pm.group(1), float(pm.group(2))
                    pair_losses.setdefault(name, []).append(val)
                pending_epoch = None

    return epochs, avg_losses, pair_losses


# ── plot ──────────────────────────────────────────────────────────────────────

def plot_loss(epochs, avg_losses, pair_losses, out_path: Path, log_scale: bool):
    n_pairs = len(pair_losses)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Training loss  ({epochs[-1]+1} epochs)", fontsize=13, fontweight="bold")

    # ── left: avg loss ────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(epochs, avg_losses, color="black", linewidth=1.5)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Avg loss (kcal/mol)²", fontsize=11)
    ax.set_title("Average loss across all pairs", fontsize=11)
    if log_scale:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # ── right: per-pair loss ──────────────────────────────────────────────────
    ax = axes[1]
    cmap = plt.get_cmap("tab10")
    for i, (name, losses) in enumerate(sorted(pair_losses.items())):
        # align lengths (some early epochs may be missing if log was truncated)
        n = min(len(epochs), len(losses))
        ax.plot(epochs[:n], losses[:n], label=name, color=cmap(i % 10), linewidth=1.2)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss (kcal/mol)²", fontsize=11)
    ax.set_title("Per-pair loss", fontsize=11)
    if log_scale:
        ax.set_yscale("log")
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def parse_args():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Plot training loss from log file")
    parser.add_argument(
        "--log-file",
        default=str(script_dir / "output" / "train_backend.log"),
        help="Path to the training log file",
    )
    parser.add_argument(
        "--out",
        default=str(script_dir / "output" / "loss_curve.png"),
        help="Output image path",
    )
    parser.add_argument("--no-log-scale", action="store_true",
                        help="Use linear Y axis instead of log scale")
    return parser.parse_args()


def main():
    args = parse_args()
    log_path = args.log_file

    if not Path(log_path).exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    epochs, avg_losses, pair_losses = parse_log(log_path)
    if not epochs:
        raise ValueError("No epoch lines found in the log file.")

    print(f"Parsed {len(epochs)} epochs  |  pairs: {list(pair_losses.keys())}")
    print(f"  First avg loss: {avg_losses[0]:.4f}")
    print(f"  Last  avg loss: {avg_losses[-1]:.4f}")

    plot_loss(epochs, avg_losses, pair_losses,
              Path(args.out), log_scale=not args.no_log_scale)


if __name__ == "__main__":
    main()
