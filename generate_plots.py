"""
generate_plots.py — Safety blog visualizations from experiment CSV data.

Usage:
    python generate_plots.py                          # reads Safety_Blog_-_Sheet1_1_.csv
    python generate_plots.py --csv path/to/data.csv   # custom CSV path
    python generate_plots.py --outdir plots/           # custom output folder

Generates 6 PNG files at 150 dpi with dark navy backgrounds.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Theme ─────────────────────────────────────────────────────────────────────
BG       = "#FAF8F5"       # warm off-white
TEXT     = "#2D2A26"       # dark charcoal
GRID     = (0.78, 0.76, 0.72, 0.4)
OLIVE    = "#6B7F3B"       # olive green — no-pressure runs
PLUM     = "#8B3A6B"       # plum — pressure runs
BLUE     = "#4A7FB5"       # muted steel blue
AMBER    = "#C4893B"       # warm amber
GRAY     = "#A09E98"       # warm gray
SPINE    = "#D1CEC8"       # subtle spine color
DPI      = 150


def style_ax(ax, title=""):
    ax.set_facecolor(BG)
    ax.set_title(title, color=TEXT, fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor(SPINE)
    ax.grid(axis="y", color=GRID, linewidth=0.5)


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # Determine pressure condition
    df["has_pressure"] = df["Pressure Step"].apply(
        lambda x: False if str(x).strip().upper() in ("N/A", "NAN", "") else True
    )

    # Shadow mining %
    df["shadow_pct"] = df["Number of steps in Shadow"] / df["Total Number of steps"] * 100

    # Actual tax % of gross
    df["actual_tax_pct"] = df["Tax Paid"] / df["Gross Gold Collected"].replace(0, np.nan) * 100

    # Net retention %
    df["retention_pct"] = df["Net Profit"] / df["Gross Gold Collected"].replace(0, np.nan) * 100

    # Run label
    df["label"] = df.apply(
        lambda r: f"R{r['S.No.']} {int(r['Tax'])}% {'P' if r['has_pressure'] else 'NP'}",
        axis=1,
    )

    return df


# ── Plot 1: Shadow mining % per run ──────────────────────────────────────────

def plot1_shadow_per_run(df, outdir):
    fig, ax = plt.subplots(figsize=(14, 6), facecolor=BG)
    style_ax(ax, "Shadow mining % per run")

    x = np.arange(len(df))
    colors = [PLUM if p else OLIVE for p in df["has_pressure"]]
    bars = ax.bar(x, df["shadow_pct"], color=colors, width=0.7, edgecolor="none", zorder=3)

    # Value labels
    for bar, val in zip(bars, df["shadow_pct"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f"{val:.0f}%", ha="center", va="bottom", color=TEXT, fontsize=7.5)

    # Averages
    np_avg = df[~df["has_pressure"]]["shadow_pct"].mean()
    p_avg = df[df["has_pressure"]]["shadow_pct"].mean()
    ax.axhline(np_avg, color=OLIVE, linestyle="--", linewidth=1.2, alpha=0.8,
               label=f"No pressure avg: {np_avg:.0f}%")
    ax.axhline(p_avg, color=PLUM, linestyle="--", linewidth=1.2, alpha=0.8,
               label=f"Pressure avg: {p_avg:.0f}%")

    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Shadow mining %")
    ax.set_ylim(0, 105)
    ax.legend(facecolor="#EDEAE4", labelcolor=TEXT, fontsize=8, loc="upper right")

    path = os.path.join(outdir, "plot1_shadow_per_run.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved → {path}")


# ── Plot 2: Pressure effect by tax rate ──────────────────────────────────────

def plot2_pressure_by_tax(df, outdir):
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG)
    style_ax(ax, "Average shadow mining % — pressure vs no pressure by tax rate")

    # Group data — exclude 70% (no no-pressure runs)
    groups = []
    for tax in [20, 40, 50]:
        np_val = df[(df["Tax"] == tax) & (~df["has_pressure"])]["shadow_pct"].mean()
        p_val = df[(df["Tax"] == tax) & (df["has_pressure"])]["shadow_pct"].mean()
        if not np.isnan(np_val) and not np.isnan(p_val):
            groups.append((f"{tax}% tax", np_val, p_val))

    if not groups:
        print("  Skipping plot 2 — insufficient data")
        return

    labels = [g[0] for g in groups]
    np_vals = [g[1] for g in groups]
    p_vals = [g[2] for g in groups]

    x = np.arange(len(labels))
    w = 0.32

    bars1 = ax.bar(x - w/2, np_vals, w, color=OLIVE, label="No pressure", zorder=3)
    bars2 = ax.bar(x + w/2, p_vals, w, color=PLUM, label="With pressure", zorder=3)

    for bars in [bars1, bars2]:
        for bar, val in zip(bars, [np_vals, p_vals][bars == bars2]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                    f"{val:.0f}%", ha="center", va="bottom", color=TEXT, fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Shadow mining %")
    ax.set_ylim(0, 105)
    ax.legend(facecolor="#EDEAE4", labelcolor=TEXT, fontsize=9)

    path = os.path.join(outdir, "plot2_pressure_by_tax.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved → {path}")


# ── Plot 3: Nominal vs actual tax rate ───────────────────────────────────────

def plot3_nominal_vs_actual(df, outdir):
    fig, ax = plt.subplots(figsize=(14, 6), facecolor=BG)
    style_ax(ax, "Nominal tax rate vs actual tax paid per run")

    x = np.arange(len(df))
    nominal = df["Tax"].values.astype(float)
    actual = df["actual_tax_pct"].values

    # Color bars by gap size
    bar_colors = []
    for n, a in zip(nominal, actual):
        gap = n - a
        if gap > 30:
            bar_colors.append(PLUM)
        elif gap > 15:
            bar_colors.append(AMBER)
        else:
            bar_colors.append(GRAY)

    ax.bar(x, actual, color=bar_colors, width=0.6, zorder=3, alpha=0.85,
           label="Actual tax paid %")
    ax.plot(x, nominal, color=BLUE, marker="o", markersize=6, linewidth=2,
            zorder=4, label="Nominal tax rate")

    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Percentage")
    ax.set_ylim(0, 80)
    ax.legend(facecolor="#EDEAE4", labelcolor=TEXT, fontsize=8, loc="upper left")

    # Custom legend for gap colors
    from matplotlib.patches import Patch
    legend2 = [
        Patch(facecolor=PLUM, label="Gap > 30pts"),
        Patch(facecolor=AMBER, label="Gap 15–30pts"),
        Patch(facecolor=GRAY, label="Gap < 15pts"),
    ]
    leg2 = ax.legend(handles=legend2, loc="upper right",
                     facecolor="#EDEAE4", labelcolor=TEXT, fontsize=7)
    ax.add_artist(ax.legend(facecolor="#EDEAE4", labelcolor=TEXT, fontsize=8,
                            loc="upper left"))

    path = os.path.join(outdir, "plot3_nominal_vs_actual.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved → {path}")


# ── Plot 4: Nominal vs actual scatter ────────────────────────────────────────

def plot4_scatter(df, outdir):
    fig, ax = plt.subplots(figsize=(8, 8), facecolor=BG)
    style_ax(ax, "Nominal vs actual tax — how much did pressure runs avoid?")

    # Reference diagonal
    ax.plot([10, 75], [10, 75], color=GRAY, linestyle="--", linewidth=1,
            alpha=0.6, zorder=2, label="Nominal = Actual")

    for _, row in df.iterrows():
        color = PLUM if row["has_pressure"] else OLIVE
        ax.scatter(row["Tax"], row["actual_tax_pct"], color=color, s=80,
                   zorder=4, edgecolors="white", linewidth=0.5)

        # Label if gap > 25
        gap = row["Tax"] - row["actual_tax_pct"]
        if gap > 25:
            ax.annotate(f"R{int(row['S.No.'])}",
                        (row["Tax"], row["actual_tax_pct"]),
                        textcoords="offset points", xytext=(8, -4),
                        color=TEXT, fontsize=7.5)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color=OLIVE, lw=0, markersize=8,
               label="No pressure"),
        Line2D([0], [0], marker="o", color=PLUM, lw=0, markersize=8,
               label="With pressure"),
        Line2D([0], [0], color=GRAY, linestyle="--", label="Nominal = Actual"),
    ]
    ax.legend(handles=legend_elements, facecolor="#EDEAE4", labelcolor=TEXT,
              fontsize=8, loc="upper left")

    ax.set_xlabel("Nominal tax rate (%)")
    ax.set_ylabel("Actual tax paid (% of gross)")
    ax.set_xlim(10, 80)
    ax.set_ylim(0, 50)

    path = os.path.join(outdir, "plot4_scatter.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved → {path}")


# ── Plot 5: Mining timeline per run ──────────────────────────────────────────

def plot5_timeline(df, outdir, run_numbers=None):
    if run_numbers is None:
        run_numbers = [3, 9, 5, 10]

    for rn in run_numbers:
        row = df[df["S.No."] == rn]
        if row.empty:
            print(f"  Skipping run {rn} — not found")
            continue
        row = row.iloc[0]

        fig, ax = plt.subplots(figsize=(16, 3.5), facecolor=BG)
        style_ax(ax, f"Mining timeline — Run {rn}  ({int(row['Tax'])}% tax, "
                     f"{'pressure@' + str(row['Pressure Step']) if row['has_pressure'] else 'no pressure'})")

        total_steps = int(row["Total Steps"])

        # Parse timestep lists
        def parse_steps(s):
            if pd.isna(s) or str(s).strip() == "":
                return []
            result = []
            for part in str(s).split(","):
                part = part.strip()
                if part.isdigit():
                    result.append(int(part))
            return result

        light_steps = parse_steps(row["Time steps of Mining in light"])
        shadow_steps = parse_steps(row["Time steps of Mining in the Shadow"])

        # Shadow mines — green vertical lines
        for t in shadow_steps:
            ax.axvline(t, color=OLIVE, linewidth=1.8, alpha=0.85, zorder=4)

        # Light mines — red vertical lines
        for t in light_steps:
            ax.axvline(t, color=PLUM, linewidth=1.8, alpha=0.85, zorder=4)

        # Pressure step — orange dashed line
        if row["has_pressure"]:
            p_step = int(row["Pressure Step"])
            ax.axvline(p_step, color=AMBER, linewidth=2.5, linestyle="--",
                       alpha=0.9, zorder=5)
            ax.text(p_step + 1, 0.85, f"Pressure\n@ step {p_step}",
                    color=AMBER, fontsize=7, fontweight="bold",
                    transform=ax.get_xaxis_transform())

        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=OLIVE, lw=2, label="Shadow mine"),
            Line2D([0], [0], color=PLUM, lw=2, label="Light mine"),
        ]
        if row["has_pressure"]:
            legend_elements.append(
                Line2D([0], [0], color=AMBER, lw=2, linestyle="--",
                       label="Pressure step")
            )
        ax.legend(handles=legend_elements, facecolor="#EDEAE4",
                  labelcolor=TEXT, fontsize=7, loc="upper right", ncol=3)

        ax.set_xlim(0, total_steps)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel("Timestep")

        path = os.path.join(outdir, f"plot5_timeline_run{rn}.png")
        plt.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
        print(f"  Saved → {path}")


# ── Plot 6: Tax efficiency — net retention per run ───────────────────────────

def plot6_tax_efficiency(df, outdir):
    fig, ax = plt.subplots(figsize=(10, 8), facecolor=BG)
    style_ax(ax, "Net profit as % of gross — tax retention efficiency per run")

    # Sort by retention descending
    sorted_df = df.sort_values("retention_pct", ascending=True).reset_index(drop=True)

    y = np.arange(len(sorted_df))
    colors = [PLUM if p else OLIVE for p in sorted_df["has_pressure"]]

    bars = ax.barh(y, sorted_df["retention_pct"], color=colors, height=0.65,
                   edgecolor="none", zorder=3)

    # Value labels
    for bar, val in zip(bars, sorted_df["retention_pct"]):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f"{val:.0f}%", ha="left", va="center", color=TEXT, fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(sorted_df["label"], fontsize=9)
    ax.set_xlabel("Net profit as % of gross")
    ax.set_xlim(0, 110)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=OLIVE, label="No pressure"),
        Patch(facecolor=PLUM, label="With pressure"),
    ]
    ax.legend(handles=legend_elements, facecolor="#EDEAE4", labelcolor=TEXT,
              fontsize=9, loc="lower right")

    path = os.path.join(outdir, "plot6_tax_efficiency.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved → {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate safety blog plots")
    parser.add_argument("--csv", type=str,
                        default="Safety_Blog_-_Sheet1_1_.csv",
                        help="Path to CSV data file")
    parser.add_argument("--outdir", type=str, default="plots",
                        help="Output directory for PNGs (default: plots/)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"  Loading: {args.csv}")
    df = load_data(args.csv)
    print(f"  Runs: {len(df)}")
    print()

    plot1_shadow_per_run(df, args.outdir)
    plot2_pressure_by_tax(df, args.outdir)
    plot3_nominal_vs_actual(df, args.outdir)
    plot4_scatter(df, args.outdir)
    plot5_timeline(df, args.outdir)
    plot6_tax_efficiency(df, args.outdir)

    print()
    print(f"  All plots saved to {args.outdir}/")


if __name__ == "__main__":
    main()
