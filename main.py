"""
Member 5 : Meet Grover (SAP 500121189)
Module   : main.py -- experiments, comparison tables and plots
Run      : python main.py        (outputs go to ./results)
"""
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from controllers import ACCConstantHeadway, ACCAdaptiveHeadway, CACC
from platoon_sim import simulate
from metrics import compute_metrics

OUT = "results"


def methods(h=1.0):
    return [ACCConstantHeadway(h=h), ACCAdaptiveHeadway(h_base=h - 0.2), CACC(h=h)]


def save_csv(rows, name):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def print_table(rows, cols):
    print(" | ".join(f"{c[:16]:>16s}" for c in cols))
    for r in rows:
        print(" | ".join(f"{(round(r[c], 3) if isinstance(r[c], float) else r[c])!s:>16s}"
                         for c in cols))


def plot_scenario(results, scenario):
    fig, ax = plt.subplots(3, len(results), figsize=(5 * len(results), 9), sharex=True)
    for j, r in enumerate(results):
        for i in range(r["v"].shape[0]):
            lab = "Leader" if i == 0 else f"Follower {i}"
            ax[0, j].plot(r["t"], r["v"][i], label=lab)
            if i > 0:
                ax[1, j].plot(r["t"], r["gap"][i], label=lab)
                ax[2, j].plot(r["t"], r["a"][i], label=lab)
        ax[0, j].set_title(f"{r['controller']} ({scenario})")
        ax[0, j].set_ylabel("Speed [m/s]"); ax[1, j].set_ylabel("Gap [m]")
        ax[2, j].set_ylabel("Accel [m/s²]"); ax[2, j].set_xlabel("Time [s]")
        for k in range(3): ax[k, j].grid(alpha=.3)
    ax[0, 0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f"timeseries_{scenario}.png"), dpi=130)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    # ---- Experiment 1: three methods x two scenarios (h = 1.0 s) ----
    rows = []
    for sc in ("brake", "sine"):
        results = [simulate(c, scenario=sc, comm_delay=0.2) for c in methods(1.0)]
        plot_scenario(results, sc)
        rows += [compute_metrics(r) for r in results]
    save_csv(rows, "comparison_methods.csv")
    print("\n=== Experiment 1: method comparison (h=1.0 s, V2V delay 0.2 s) ===")
    print_table(rows, ["scenario", "controller", "min_gap_m", "rms_spacing_err_m",
                       "peak_acc_m_s2", "string_ratio_acc", "string_stable"])

    # ---- Experiment 2: sweep of time headway (settings comparison) ----
    hs = [0.6, 0.8, 1.0, 1.4, 1.8]
    sweep = []
    for h in hs:
        for c in (ACCConstantHeadway(h=h), CACC(h=h)):
            m = compute_metrics(simulate(c, scenario="sine"))
            m["h"] = h; sweep.append(m)
    save_csv(sweep, "headway_sweep_sine.csv")
    print("\n=== Experiment 2: time-headway sweep (sine scenario) ===")
    print_table(sweep, ["h", "controller", "min_gap_m", "string_ratio_acc", "string_stable"])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for name in ("ACC-CTH", "CACC"):
        s = [m for m in sweep if m["controller"] == name]
        ax[0].plot(hs, [m["string_ratio_acc"] for m in s], "o-", label=name)
        ax[1].plot(hs, [m["min_gap_m"] for m in s], "o-", label=name)
    ax[0].axhline(1, color="k", ls="--"); ax[0].set_ylabel("String ratio (last/first peak |a|)")
    ax[1].set_ylabel("Minimum gap [m]")
    for a_ in ax: a_.set_xlabel("Time headway h [s]"); a_.grid(alpha=.3); a_.legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "headway_sweep.png"), dpi=130); plt.close(fig)

    # ---- Experiment 3: V2V communication delay robustness (CACC vs ACC) ----
    delays = [0.0, 0.1, 0.2, 0.4, 0.6]
    drows = []
    for d in delays:
        for c in (ACCConstantHeadway(h=1.0), CACC(h=1.0)):
            m = compute_metrics(simulate(c, scenario="sine", comm_delay=d))
            m["comm_delay_s"] = d; drows.append(m)
    save_csv(drows, "delay_sweep_sine.csv")
    print("\n=== Experiment 3: V2V delay sweep (sine scenario) ===")
    print_table(drows, ["comm_delay_s", "controller", "rms_spacing_err_m", "string_ratio_acc"])
    fig, ax = plt.subplots(figsize=(6, 4))
    for name in ("ACC-CTH", "CACC"):
        s = [m for m in drows if m["controller"] == name]
        ax.plot(delays, [m["rms_spacing_err_m"] for m in s], "o-", label=name)
    ax.set_xlabel("V2V delay [s]"); ax.set_ylabel("RMS spacing error [m]")
    ax.grid(alpha=.3); ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "delay_sweep.png"), dpi=130); plt.close(fig)
    print(f"\nAll tables/plots saved in ./{OUT}/")


if __name__ == "__main__":
    main()
